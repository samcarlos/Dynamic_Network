import pandas as pd
import itertools
import numpy as np

data = pd.read_csv('/users/sweiss/src/Dynamic_Network/sipri.csv', header = None)

# Define the helper function to process the supplier data while ensuring the index corresponds to the original row
def process_supplier_data(df):
    exporter = None
    importer = None
    cleaned_data = []
    
    for index, row in df.iterrows():
        supplier_text = row[0]
        
        if pd.isna(supplier_text):
            continue  # Skip NaN rows
        
        if 'Supplier' in supplier_text:
            # Start a new block of exporters
            exporter = []
        elif 'Total exports to' in supplier_text:
            # Extract importer and append rows to cleaned data with their original index
            importer = supplier_text.split('Total exports to')[-1].strip()
            # Each exporter gets its original row's index
            for exp in exporter:
                cleaned_data.append({
                    'original_index': exp['index'],  # Use the index from when the exporter was encountered
                    'Exporter': exp['country'],      # The actual exporter country
                    'Importer': importer     # The importer country
                })
        elif exporter is not None:
            # Append each exporter with its original row index
            exporter.append({'index': index, 'country': supplier_text.strip()})
    
    # Convert cleaned_data to a DataFrame
    return pd.DataFrame(cleaned_data)


# Re-run the data cleaning process
df_cleaned_with_index = process_supplier_data(data)

data['original_index'] = data.index

# Merge the cleaned data with the original data
sipri_data = pd.merge(data, df_cleaned_with_index, on='original_index', how='left')
#final_data.to_csv('/users/sweiss/downloads/sipri_cleaned.csv', index=False)

importer_exporter_table = sipri_data[~sipri_data['Exporter'].isnull()]
importer_exporter_table = importer_exporter_table.drop(0, axis = 1)
years = [x + 1950 for x in range(75)]
importer_exporter_table_years = importer_exporter_table.iloc[:,:len(years)]
importer_exporter_table_years.columns = years
importer_exporter_table_years['Exporter'] = importer_exporter_table['Exporter'] 
importer_exporter_table_years['Importer'] = importer_exporter_table['Importer'] 

importer_exporter_table_years = importer_exporter_table_years.drop(2024, axis = 1)

importer_exporter_table_years_long = importer_exporter_table_years.melt(id_vars = ['Exporter', 'Importer'], var_name = 'Year', value_name = 'Value')


#create scafold for all cpermutateions of countries and years
countries = importer_exporter_table_years_long[['Exporter',"Importer"]].melt()['value'].unique()
years = importer_exporter_table_years_long['Year'].unique()

scafold = pd.DataFrame([(x,y) for x in countries for y in years], columns = ['Country', 'Year'])
scafold = scafold.merge(pd.DataFrame(countries), all = True)

#
# Create all permutations of the columns
permutations = list(itertools.product(countries, countries, years))
# Convert to a pandas DataFrame
df_permutations = pd.DataFrame(permutations, columns=['Exporter', 'Importer', 'Year'])
df_permutations = df_permutations.loc[df_permutations['Exporter'] != df_permutations['Importer']] 
df_permutations_table = df_permutations.merge(importer_exporter_table_years_long, how = 'left', on = ['Exporter', 'Importer', 'Year'])

df_permutations_table['Pair'] = df_permutations_table.apply(lambda x: frozenset([x['Exporter'], x['Importer']]), axis=1)

# Set value to 1 if there is a non-zero and non-null value, otherwise set to 0
df_permutations_table['Value'] = df_permutations_table['Value'].apply(lambda x: 1 if pd.notna(x) and x != 0 else 0)

# Group by the unordered 'Pair' and 'Year', and keep the maximum value (this ensures we get 1 if there's a valid value)
df_unique_pairs = df_permutations_table.groupby(['Pair', 'Year']).agg({'Value': 'max'}).reset_index()
df_unique_pairs[['Country1', 'Country2']] = pd.DataFrame(df_unique_pairs['Pair'].tolist(), index=df_unique_pairs.index)
df_unique_pairs.loc[(df_unique_pairs['Country1'] == "United Kingdom") & (df_unique_pairs['Country2'] == "United States")]

from sklearn.preprocessing import OneHotEncoder

# Create a one-hot encoder
encoder = OneHotEncoder()
encoder.fit(countries.reshape(-1,1))

scaled_year = np.array((df_unique_pairs['Year'] - np.mean(years)) / np.std(years))

Country_1 = encoder.transform(df_unique_pairs[['Country1']]).toarray()
Country_2 = encoder.transform(df_unique_pairs[['Country2']]).toarray()

Country_1 = np.concatenate([Country_1, scaled_year.reshape(-1,1)], axis = 1)
Country_2 = np.concatenate([Country_2, scaled_year.reshape(-1,1)], axis = 1)    

response = np.array(df_unique_pairs['Value']).reshape(-1, 1)




import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Define the base network (equivalent to Keras's Sequential)
class BaseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(BaseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Define the Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        self.base_network = BaseNetwork(input_dim)

    def forward(self, input_a, input_b):
        # Pass through the shared base network
        output_a = self.base_network(input_a)
        output_b = self.base_network(input_b)
        
        # Compute the Euclidean distance
        distance = F.pairwise_distance(output_a, output_b)
        return distance

# Define the contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label):
        # Compute the contrastive loss as per Hadsell et al.
        euclidean_distance = output
        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) + 
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

# Initialize the Siamese network and the loss function
input_dim = Country_1.shape[1]  # Your input dimension
model = SiameseNetwork(input_dim)
criterion = ContrastiveLoss()

# Optimizer
optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

# Example training loop
def train(model, criterion, optimizer, Country_1, Country_2, response, num_epochs=25, batch_size=10240):
    model.train()
    dataset = torch.utils.data.TensorDataset(torch.tensor(Country_1), torch.tensor(Country_2), torch.tensor(response))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            input_a, input_b, label = data
            
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(input_a.float(), input_b.float())

            # Compute the loss
            loss = criterion(output, label.float())

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

train(model, criterion, optimizer, Country_1, Country_2, response, num_epochs=25, batch_size=10240)

def get_embeddings(model, x_train):
    model.eval()
    with torch.no_grad():
        embeddings = model.base_network(torch.tensor(x_train).float())
    return embeddings.numpy()

# Get the embeddings for the training data
embeddings_1 = get_embeddings(model, Country_1)
embeddings_2 = get_embeddings(model, Country_2)

embeddings_1 = pd.DataFrame(embeddings_1)
embeddings_2 = pd.DataFrame(embeddings_2)

embeddings_1.columns = ['embedding_' + str(x) for x in range(embeddings_1.shape[1])]
embeddings_2.columns = ['embedding_' + str(x) for x in range(embeddings_2.shape[1])]

embeddings_1['Year'] = df_unique_pairs['Year']
embeddings_2['Year'] = df_unique_pairs['Year']

embeddings_1['Country'] = df_unique_pairs['Country1']
embeddings_2['Country'] = df_unique_pairs['Country2']

embeddings_1['Value'] = df_unique_pairs['Value']
embeddings_2['Value'] = df_unique_pairs['Value']


embeddings_df = pd.concat([embeddings_1, embeddings_2], ignore_index=True)