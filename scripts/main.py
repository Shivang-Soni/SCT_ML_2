# scripts/main.py
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Relative imports I have
from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from .model import KMeansClustering
from .utils import load_data, save_data

# Logging configuration
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.info("Starting main pipeline.......")

    # Load data
    data = load_data(RAW_DATA_PATH)

    # Initialize KMeans pipeline
    pipeline = KMeansClustering(n_clusters=5, random_state=42)

    # Preprocess data (all columns except index)
    feature_columns = list(data.columns[1:])
    preprocessed_data = pipeline.preprocess_data(data, feature_columns)

    # Train model
    pipeline.train_model()

    # Add cluster labels
    clustered_data = pipeline.add_clusters(data)

    # Save processed data
    save_data(clustered_data, PROCESSED_DATA_PATH)

    # Save model and scaler
    pipeline.save_pipeline()

    # PCA 2D visualization if at least 2 features
    if len(feature_columns) >= 2:
        logging.info("Reducing features to 2D for visualization with PCA...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(preprocessed_data)

        clustered_data_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        clustered_data_pca["Cluster"] = clustered_data["Cluster"]

        plt.figure(figsize=(8, 6))
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
