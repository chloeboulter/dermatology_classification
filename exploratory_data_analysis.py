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

def plot_distributions(df):
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

#    sns.histplot(df["age"].dropna(), kde=True)
#    plt.title("Age Distribution of Patients in Dermatology Dataset")

missing_values = ["?"]
df_raw = pd.read_csv("dermatology_dataset.csv",na_values="?")

perform_eda(df_raw)
plot_distributions(df_raw)


