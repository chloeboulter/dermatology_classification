import pandas as pd

missing_values = ["?"]
df_raw = pd.read_csv("dermatology_dataset.csv")
print(df_raw.head())
print(df_raw.shape)
print(df_raw["class"].value_counts())
