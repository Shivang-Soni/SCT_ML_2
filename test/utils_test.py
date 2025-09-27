# utils_test.py - Unit testing this time with pytest
import os
import sys

# Add project root to sys.path so 'scripts' package is recognized
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from scripts.utils import load_data, save_data
from scripts.config import RAW_TEST_PATH, MODEL_TEST_PATH


# Fixture to provide dummy data
@pytest.fixture
def dummy_data():
    df = pd.DataFrame({
        "ID": range(1, 6),
        "F1": range(10, 15),
        "F2": range(15, 20)
    })
    return df


# Test for save and load data
def test_save_and_load_data(dummy_data):
    # Save the dummy data
    save_data(dummy_data, RAW_TEST_PATH)
    assert os.path.exists(RAW_TEST_PATH), "No file has been saved!"

    # Load the saved data
    loaded_data = load_data(RAW_TEST_PATH)
    assert not loaded_data.empty, "Loaded dataset is empty!"
    assert list(dummy_data.columns) == list(loaded_data.columns), "Loaded data has invalid columns!"


# Cleanup fixture to remove test files after all tests in this module
@pytest.fixture(scope="module", autouse=True)
def clean_up():
    yield  # run tests first
    for file in (RAW_TEST_PATH, MODEL_TEST_PATH):
        if os.path.exists(file):
            os.remove(file)
