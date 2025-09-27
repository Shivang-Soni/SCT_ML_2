# Unit tests I am carrying out here additionally
import unittest
import os 
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.model import KMeansClustering
from scripts.utils import save_data, load_data
from scripts.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

class KMeansTestingPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(RAW_DATA_PATH):
            # Create dummy data if the CSV does not exist
            dummy_data = pd.DataFrame({
                "ID": range(1, 11),
                "Feature1": range(10, 20),
                "Feature2": range(20, 30),
                "Feature3": range(30, 40),
                "Feature4": range(40, 50),
                "Feature5": range(50, 60)
            })
            os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
            save_data(dummy_data, RAW_DATA_PATH)

    # Checking whether the pipeline actually functions correctly
    def test_pipeline_execution(self):
        data = load_data(RAW_DATA_PATH)
        self.assertFalse(data.empty, "Dataframe must not be empty!")

        # Initialize pipeline
        pipeline = KMeansClustering(n_clusters=3)
        self.assertIsNotNone(pipeline, "Pipeline has to be initalised first!")

        # Index is redundant in our case
        feature_columns = list(data.columns[1:])
        preprocessed_data = pipeline.preprocess_data(data, feature_columns)
        self.assertEqual(
            preprocessed_data.shape[1], 
            len(feature_columns), 
            "After preprocessing the number of feature columns have been changed!"
        )

        # Training the model
        pipeline.train_model()
        self.assertIsNotNone(pipeline.model, "Model hasn't been trained yet!")

        # Adding a cluster
        clustered_data = pipeline.add_clusters(data)
        self.assertIn("Cluster", clustered_data.columns, "Clustered labels must be present!")

        # Saving the data 
        save_data(clustered_data, PROCESSED_DATA_PATH)
        self.assertTrue(os.path.exists(PROCESSED_DATA_PATH), "Processed CSV must exist!")

# Main function
if __name__ == "__main__":
    unittest.main()
