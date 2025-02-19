#!/usr/bin/env python3
"""
This script processes food data to calculate nutritional values per 100g,
generates food descriptions, obtains embeddings for these descriptions using
the OpenAI API, and saves the results to a CSV file.

Dependencies:
- pandas
- numpy
- torch
- loguru
- p_tqdm
- tqdm
- openai
"""

import pandas as pd
import numpy as np
import torch

from loguru import logger

from p_tqdm import p_map
from tqdm import tqdm
tqdm.pandas()

from openai import OpenAI

# Constants
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
DEBUG_MODE = False
EMBEDDING_VERSION = "v0.1"

DEBUG_PREFIX = "-debug" if DEBUG_MODE else ""

MFR_DATA_PATH = "./db_helpers/cache/mfr_data_v1.csv"
OUTPUT_PATH = f"./data/raw/food_embeddings_{EMBEDDING_VERSION}{DEBUG_PREFIX}.csv"
OUTPUT_PICKLE_PATH = f"./data/raw/food_embeddings_{EMBEDDING_VERSION}{DEBUG_PREFIX}.pkl.gz"

NUM_CPUS=8

logger.info(f"DEBUG_MODE: {DEBUG_MODE}")
logger.info(f"DEBUG_PREFIX: {DEBUG_PREFIX}")
logger.info(f"EMBEDDING_VERSION: {EMBEDDING_VERSION}")
logger.info(f"MFR_DATA_PATH: {MFR_DATA_PATH}")
logger.info(f"OPENAI_EMBEDDING_MODEL: {OPENAI_EMBEDDING_MODEL}")
logger.info(f"OUTPUT_PATH: {OUTPUT_PATH}")
logger.info(f"OUTPUT_PICKLE_PATH: {OUTPUT_PICKLE_PATH}")



def calculate_nutritional_components(row):
    """
    Calculate nutritional components per 100g for a given food item.
    
    Parameters:
        row (pd.Series): A row from the food data DataFrame containing the food item
                         details and nutritional values.
    
    Returns:
        dict: Dictionary containing food identifiers and their nutritional information (per 100g).
    """
    eaten_quantity_in_gram = row["eaten_quantity_in_gram"]
    base_components = {
        'food_id': row['food_id'],
        'version': EMBEDDING_VERSION,
        'display_name_en': row['display_name_en'],
        'display_name_fr': row['display_name_fr'],
        'display_name_de': row['display_name_de'],
    }
    
    if eaten_quantity_in_gram is not None and eaten_quantity_in_gram > 0:
        # Mapping of nutritional fields to their corresponding "eaten" columns.
        nutritional_fields = {
            'energy_kcal': 'energy_kcal_eaten',
            'carb': 'carb_eaten',
            'fat': 'fat_eaten',
            'protein': 'protein_eaten',
            'fiber': 'fiber_eaten',
            'alcohol': 'alcohol_eaten'
        }
        
        # Calculate the nutritional value per 100g.
        for field, eaten_field in nutritional_fields.items():
            base_components[f'{field}_per_100g'] = (row[eaten_field] / eaten_quantity_in_gram) * 100
    else:
        # Log a warning if eaten_quantity_in_gram is invalid or zero.
        logger.warning(f"Invalid or zero quantity for food_id: {row['food_id']}, {row['display_name_en']}. "
                       "Setting nutritional values to 0")
        for field in ['energy_kcal', 'carb', 'fat', 'protein', 'fiber', 'alcohol']:
            base_components[f'{field}_per_100g'] = 0
            
    return base_components


def process_food_data(mfr_df):
    """
    Process the raw food data to calculate nutritional components for each food item.
    
    Parameters:
        mfr_df (pd.DataFrame): DataFrame containing the raw food data.
        
    Returns:
        pd.DataFrame: DataFrame with each unique food item and its calculated nutritional values.
    """
    logger.info(f"Processing {len(mfr_df)} food entries")
    
    nutritional_components = []
    # Group the data by food_id and process the first entry in each group.
    for food_id, food_id_df in mfr_df.groupby("food_id"):
        row = food_id_df.iloc[0]
        components = calculate_nutritional_components(row)
        nutritional_components.append(components)
    
    logger.debug(f"Created nutritional components for {len(nutritional_components)} unique food items")
    return pd.DataFrame(nutritional_components)


def openai_embed(text):
    """
    Get OpenAI embeddings for the provided text.
    
    In case of an error during the API call, the function logs the error and returns
    a string with the error message. This ensures that the function returns a picklable
    object that can be used in parallel processing.
    
    Parameters:
        text (str): The text to generate embeddings for.
        
    Returns:
        list or str: A list containing the embedding vectors if successful, or an error 
                     string if an exception occurred.
    """
    try:
        client = OpenAI()
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text
        )
        # Return only the embedding list which is picklable.
        return torch.tensor(response.data[0].embedding, dtype=torch.float32)
    except Exception as e:
        logger.error(f"Error generating embeddings for text (first 50 chars): {text[:50]}... Error: {e}")
        return f"Error: {str(e)}"  # Alternatively, a placeholder like [0.0] * embedding_dim may be used.


def generate_food_description(row):
    """
    Generate a formatted description of the food item using its nutritional information.
    
    Parameters:
        row (dict): A dictionary with food details and nutritional values.
    
    Returns:
        str: A multi-line string description of the food item.
    """
    return f"""Food Item:
English Name: {row['display_name_en']}
French Name: {row['display_name_fr']}
German Name: {row['display_name_de']}
Nutritional Content (per 100g):
- Energy: {row['energy_kcal_per_100g']:.1f} kcal
- Carbohydrates: {row['carb_per_100g']:.1f}g
- Fat: {row['fat_per_100g']:.1f}g  
- Protein: {row['protein_per_100g']:.1f}g
- Fiber: {row['fiber_per_100g']:.1f}g
- Alcohol: {row['alcohol_per_100g']:.1f}g"""


def main():
    """
    Main function to generate food embeddings:
    
    1. Loads the raw food data.
    2. Processes the data to calculate nutritional components per 100g.
    3. Generates a description for each food item.
    4. Calls the OpenAI API in parallel to compute embeddings based on these descriptions.
    5. Saves the enriched data (with descriptions and embeddings) to a CSV file.
    """
    logger.info(f"Starting food embeddings generation from {MFR_DATA_PATH}")
    
    try:
        # Load the raw food data from CSV.
        mfr_df = pd.read_csv(MFR_DATA_PATH)
        logger.info(f"Successfully loaded {len(mfr_df)} rows from MFR data")
        
        # Process raw food data to compute nutritional values per 100g.
        nutritional_components_df = process_food_data(mfr_df)
        
        # Optionally reduce the dataset size in debug mode for faster processing.
        if DEBUG_MODE:
            logger.info("Debug mode is enabled, reducing the number of food items to 100")
            nutritional_components_df = nutritional_components_df[:100]
        
        # Generate descriptions for each food item.
        logger.info("Generating food descriptions")
        nutritional_components_df['description'] = p_map(
            generate_food_description, 
            nutritional_components_df.to_dict(orient="records")
        )
            
        # Generate embeddings in parallel using the OpenAI API.
        logger.info("Generating OpenAI embeddings in parallel")
        nutritional_components_df['embedding'] = p_map(openai_embed, nutritional_components_df['description'], num_cpus=NUM_CPUS)
        
        # Add reference to the embedding model used
        nutritional_components_df["embedding_mode"] = OPENAI_EMBEDDING_MODEL
        
        # Save the results to a CSV file.
        logger.info(f"Saving food embeddings to {OUTPUT_PICKLE_PATH}")
        nutritional_components_df.to_pickle(OUTPUT_PICKLE_PATH, compression="gzip")
        logger.success(f"Successfully saved food embeddings to {OUTPUT_PICKLE_PATH}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()

