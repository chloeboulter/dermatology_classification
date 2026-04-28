import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from src.data_preprocessing import remove_missing_values, scale_features
from src.models import build_cnn, configure_mlp, configure_svm
from src.exploratory_data_analysis import run_cluster_analysis

def main():
    df_raw = pd.read_csv("dermatology_dataset.csv", na_values="?")
    df_clean = remove_missing_values(df_raw)

    X = df_clean.drop(columns=["class"])
    y = to_categorical(df_clean['class'] - 1)

    X_scaled = scale_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train_scaled = scale_features(X_train)
    X_test_scaled = scale_features(X_test)

    run_cluster_analysis(X_scaled)

