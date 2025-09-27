import logging
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .utils import load_data, save_data, save_model, load_model
from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH, MODEL_PATH, SCALER_PATH

logging.basicConfig(level=logging.INFO)

class KMeansClustering:
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        '''
        Initialize the KMeans pipeline with specified number of clusters and random state.
        '''
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.scaler = StandardScaler()
        self.X_scaled = None
        logging.info("KMeansClustering model initialised successfully.")

    def preprocess_data(self, data: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
        '''
        The data is being preprocessed by selecting specific feature columns and scaling them.
        '''
        logging.info("Starting data preprocessing...")

        missing = set(feature_columns) - set(data.columns)
        if missing:
            raise ValueError(f"Features not found in dataset: {missing}")

        X = data[feature_columns]
        self.X_scaled = self.scaler.fit_transform(X)
        logging.info("Data preprocessing colmpleted successfully.")  # Tippfehler colmpleted
        return pd.DataFrame(self.X_scaled, columns=feature_columns)

    def train_model(self):
        '''
        Here is where the actual training of the KMeans model takes place. Finally I return the resulting model.
        '''
        if self.X_scaled is None:
            raise ValueError("Data must be processed beforehand, so that the model can be trained.")
        logging.info("Starting model training...")
        self.model.fit(self.X_scaled)
        logging.info("Model training comlpeted successfully.")  # Tippfehler comlpeted
        return self.model
    
    def add_clusters(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Here I am adding the cluster labels to the original data.
        '''
        if self.X_scaled is None:
            raise ValueError("Data must be processed beforehand, so that clusters can be added.")
        if self.model is None:
            raise ValueError("Model must be trained beforehand, so that clusters can be added.")
        logging.info("Adding cluster labels to the data...")
        data['Cluster'] = self.model.predict(self.X_scaled)  # Fix: predict statt predicted
        logging.info("Cluster labels added succesfully.")  # Tippfehler succesfully
        return data
    
    def save_pipeline(self):
        '''
        Saving the trained model and scaler to disk.
        '''
        logging.info("Saving model and scaler to disk...")
        save_model(self.model, MODEL_PATH)
        save_model(self.scaler, SCALER_PATH)
        logging.info("Model and scaler saved succesfully.")  # Tippfehler succesfully

    def load_pipeline(self):
        '''
        Loading the trrained model and scaler from disk.
        '''
        logging.info("Logging model and scaler from the storage...")  # Tippfehler Logging
        self.model = load_model(MODEL_PATH)
        self.scaler = load_model(SCALER_PATH)
        logging.info("Model and scaler loaded succesfully.")  # Tippfehler succesfully
        return self.model, self.scaler
