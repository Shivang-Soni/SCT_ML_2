# Unit testing this time with pytest instead.
import os
import pytest
import pandas as pd
from utils import load_data, save_data
from config import RAW_TEST_PATH, MODEL_TEST_PATH


@pytest.fixture
def dummy_data():
    df = pd.DataFrame({
        "ID": range(1, 6),
        "F1": range(10, 15),
        "F2": range(15, 20)
    })
    return df


# Tests for save and load data
def test_save_and_load_data(dummy_data):
    # Save
    save_data(dummy_data, RAW_TEST_PATH)
    assert os.path.exists(RAW_TEST_PATH), "No file has been saved!"

    # Load
    loaded_data = load_data(RAW_TEST_PATH)
    assert not loaded_data.empty, "Loaded dataset is empty!"
    assert list(dummy_data.columns) == list(loaded_data.columns), "Loaded data has some invalid columns!"


# Cleanup after all tests in this module
@pytest.fixture(scope="module", autouse=True)
def clean_up():
    yield
    for file in (RAW_TEST_PATH, MODEL_TEST_PATH):
        if os.path.exists(file):
            os.remove(file)
