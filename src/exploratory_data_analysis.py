import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


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
    plt.savefig("../outputs/class_distribution.png")

def plot_age_distribution(df):
    sns.histplot(data=df, x="age", bins=15, color="#c0d5e5", edgecolor="#3c78a8")
    plt.title("Count of Ages in Dermatology Dataset")
    plt.show()
    plt.savefig("../outputs/age_distribution")

def feature_correlation(df):
    plt.figure(figsize=(15, 10))
    sns.color_palette("crest", as_cmap=True)
    sns.heatmap(df.corr(), cmap='Blues');
    plt.title("Heatmap of Correlation between features")
    plt.show()
    plt.savefig("../outputs/feature_correlation")


def run_cluster_analysis(X):
    """Runs KMeans for multiple K values and plots metrics to find optimal K value"""
    k_values=range(2,10)
    silhouette_scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(X, labels)
        silhouette_scores.append(silhouette_avg)

    davies_bouldin_scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = davies_bouldin_score(X, labels)
        davies_bouldin_scores.append(score)

    calinski_harabasz_scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = calinski_harabasz_score(X, labels)
        calinski_harabasz_scores.append(score)

    plt.figure(figsize=(8, 4))
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel("Number of clusters, k")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores")
    plt.show()

    optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
    print(f"The optimal number of clusters is {optimal_k}")


missing_values = ["?"]
df_raw = pd.read_csv("../data/dermatology_dataset.csv", na_values="?")

perform_eda(df_raw)
plot_condition_distributions(df_raw)
plot_age_distribution(df_raw)
feature_correlation(df_raw)


