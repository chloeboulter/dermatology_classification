from exploratory_data_analysis import df_raw
import pandas as pd

def remove_missing_values(df):
    """Removes rows with missing values and reports counts"""
    initial_null_count = df.isnull().sum().sum()

    df_cleaned = df.dropna()
    final_null_count = df_cleaned.isnull().sum().sum()
    print(f"Dropped {initial_null_count - final_null_count} rows containing null values.")

    return df_cleaned

df_processed = remove_missing_values(df_raw)