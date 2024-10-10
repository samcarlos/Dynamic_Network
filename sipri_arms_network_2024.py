import pandas as pd
import itertools
import numpy as np

data = pd.read_csv('/users/sweiss/downloads/sipri.csv', header = None)

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