import os
import sys
import unittest

# Add project root to sys.path so 'scripts' package is recognized
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from scripts.utils import save_model, load_model
from scripts.config import MODEL_TEST_PATH


class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Dummy model for unit testing purposes
        cls.dummy_model = {"weights": [0, 1, 2], "bias": 0.6}

    @classmethod
    def tearDownClass(cls):
        # Cleaning up test-generated data after tests
        if os.path.exists(MODEL_TEST_PATH):
            os.remove(MODEL_TEST_PATH)

    def test_save_model_creates_file(self):
        save_model(self.dummy_model, MODEL_TEST_PATH)
        self.assertTrue(os.path.exists(MODEL_TEST_PATH), "Model could not be saved!")

    def test_load_model_returns_correct_object(self):
        save_model(self.dummy_model, MODEL_TEST_PATH)
        loaded_model = load_model(MODEL_TEST_PATH)
        self.assertEqual(loaded_model, self.dummy_model, "Loaded model is different from the saved model!")


if __name__ == "__main__":
    unittest.main()
