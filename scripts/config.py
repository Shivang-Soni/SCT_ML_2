import os
import dotenv


# Load environment variables from .env file
dotenv.load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "models")
LOG_DIR = os.getenv("LOG_DIR", "logs")
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw/data.csv")
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "data/processed/processed_data.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/my_model.pkl")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
N_CLUSTERS = int(os.getenv("N_CLUSTERS", 5))
LOG_PATH = os.getenv("LOG_PATH", "logs/app.log")
SCALER_PATH = os.getenv("SCALER_PATH", "./model/scaler.pkl")
# Test-specific paths (local to tests)
RAW_TEST_PATH = "data/test/raw_test.csv"
MODEL_TEST_PATH = "data/test/model_test.pkl"
