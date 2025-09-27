import os 
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.INFO)


# Loads raw data from the CSV file
def load_data(file_path: str):
    """
    Returns data file (csv) as pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data = pd.read_csv(file_path)
    logging.info("Data loaded from %s", file_path)
    return data 


# Saves the inputted data to a file
def save_data(data: pd.DataFrame, file_path: str) -> None:
    """
    Saves the inputted data to a file in CSV format.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data.to_csv(file_path, index=False)
    logging.info("Data saved to %s", file_path)


# Saves the model to a file
def save_model(model, file_path: str) -> None:
    """
    Saves the model to a file using joblib.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    logging.info("Model saved to %s", file_path)


# Loads the model from a file
def load_model(file_path: str):
    """
    Loads a model from a file using joblib.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")
    model = joblib.load(file_path)
    logging.info("Model loaded from %s", file_path)
    return model
