import os
import logging
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from model import KMeansClustering
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils import save_data, load_data
from sklearn.decomposition import PCA

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Here will all the execution of the KMeans pipeline take place.
if __name__ == "__main__":
    logging.info("Starting main pipeline.......")

    # Loading the data
    data = load_data(RAW_DATA_PATH)

    # Creating a KMeansClusteringInstance
    pipeline = KMeansClustering(n_clusters=5, random_state=42)

    # Data preprocessing
    feature_columns = list(data.columns[1:])  # Skipping the Index
    preprocessed_data = pipeline.preprocess_data(data, feature_columns)

    # Training my model
    pipeline.train_model()

    # Adding clustering labels
    clustered_data = pipeline.add_clusters(data)

    # Saving the data
    save_data(clustered_data, PROCESSED_DATA_PATH)

    # Model and Scaler also being saved as well
    pipeline.save_pipeline()

    # PCA 2D Visualization
    if len(feature_columns) > 2:
        logging.info("Reducing features to 2D for visualization with PCA...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(preprocessed_data)

        clustered_data_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        clustered_data_pca["Cluster"] = clustered_data["Cluster"]

        plt.figure(figsize=(8,6))
        sns.scatterplot(
            x="PC1",
            y="PC2",
            hue="Cluster",
            data=clustered_data_pca,
            palette="Set2"
        )
        plt.title("KMeans Cluster Visualisierung (PCA 2D)")
        plt.savefig("cluster_plot_pca.png")
        plt.show()

    logging.info("Pipeline successfully functioning.")
