import os
import pandas as pd 
from config_generic import GenericConfig

config = GenericConfig()

df_products = pd.read_csv(os.path.join(config.dir_ip, "nilkamal_products_data.csv"))

def search_product(
    query, 
    df_products=df_products
):
    """
    Searches the DataFrame for relevant review information based on the query.
    Returns the most relevant review details as a string.
    """
    product_columns = df_products.columns.to_list()
    columns_to_remove = ['Title', 'Images', 'URL', 'Similar Products']

    product_columns = [column for column in product_columns if column not in columns_to_remove]
    # print(product_columns)
    # Check if any product name contains keywords from the query
    matching_products = df_products[df_products['Title'].fillna("").astype(str).str.contains(query.strip(), case=False, na=False, regex=False)]
    
    if matching_products.empty:
        return "Sorry, I couldn't find a matching product."

    # Get the most relevant product (first match for simplicity)
    product = matching_products.iloc[0]

    # Prepare response with available details
    response = f"**{product['Title']}**\n"

    for column in product_columns:
        if pd.notna(product[column]):
            response += f"{column}: {product[column]}\n"
            
    return response

response = search_product(query="Nilkamal Sierra 1 Seater Manual Recliner Sofa (Brown)",
                          df_products=df_products)
print(response)