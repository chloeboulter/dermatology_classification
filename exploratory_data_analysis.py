import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


condition_labels = {
    1: "Psoriasis",
    2: "Seborrheic Dermatitis",
    3: "Lichen Planus",
    4: "Pityriasis Rosea",
    5: "Chronic Dermatitis",
    6: "Pityriasis Rubra Pilaris"
}
def perform_eda(df):
    print(f"Total Patients: {df.shape[0]}")
    print(f"Total Features: {df.shape[1]}")
    print(df["class"].value_counts)

def plot_condition_distributions(df):
    df_visualisation=df.copy()
    df_visualisation["condition_name"] = df_visualisation["class"].map(condition_labels)

    plt.figure(figsize=(10, 10))

    sns.countplot(x="condition_name", data=df_visualisation)
    plt.title("Distribution of Skin Conditions")
    plt.ylabel("Count")
    plt.xlabel("Condition")
    plt.tight_layout()
    plt.show()
    plt.savefig("outputs/class_distribution.png")

def plot_age_distribution(df):
    sns.histplot(data=df, x="age", bins=15, color="#c0d5e5", edgecolor="#3c78a8")
    plt.title("Count of Ages in Dermatology Dataset")
    plt.show()
    plt.savefig("outputs/age_distribution")

def feature_correlation(df):
    plt.figure(figsize=(15, 10))
    sns.color_palette("crest", as_cmap=True)
    sns.heatmap(df.corr(), cmap='Blues');
    plt.title("Heatmap of Correlation between features")
    plt.show()
    plt.savefig("outputs/feature_correlation")

missing_values = ["?"]
df_raw = pd.read_csv("dermatology_dataset.csv",na_values="?")

perform_eda(df_raw)
plot_condition_distributions(df_raw)
plot_age_distribution(df_raw)
feature_correlation(df_raw)


