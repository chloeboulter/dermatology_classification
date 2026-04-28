from exploratory_data_analysis import df_raw
import pandas as pd
from sklearn.preprocessing import StandardScaler

def remove_missing_values(df):
    """Removes rows with missing values and reports counts"""
    initial_null_count = df.isnull().sum().sum()

    df_cleaned = df.dropna()
    final_null_count = df_cleaned.isnull().sum().sum()
    print(f"Dropped {initial_null_count - final_null_count} rows containing null values.")

    return df_cleaned


def scale_features(df):
    scaler = StandardScaler()
    scaled_data=scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    return scaled_df



def preprocess_data(df):
    print()