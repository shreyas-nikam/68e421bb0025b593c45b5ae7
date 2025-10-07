import pytest
import pandas as pd
from definition_91ebd4d2aa1b4cd4a34904868b4a57ec import validate_data_types

@pytest.fixture
def sample_dataframe():
    data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c'], 'col3': [1.1, 2.2, 3.3]}
    return pd.DataFrame(data)

def test_valid_data_types(sample_dataframe):
    expected_dtypes = {'col1': int, 'col2': str, 'col3': float}
    assert validate_data_types(sample_dataframe, expected_dtypes) == True

def test_invalid_data_types(sample_dataframe):
    expected_dtypes = {'col1': str, 'col2': int, 'col3': float}
    assert validate_data_types(sample_dataframe, expected_dtypes) == False

def test_missing_column(sample_dataframe):
    expected_dtypes = {'col1': int, 'col2': str, 'col4': float}
    assert validate_data_types(sample_dataframe, expected_dtypes) == False

def test_empty_dataframe():
    df = pd.DataFrame()
    expected_dtypes = {}
    assert validate_data_types(df, expected_dtypes) == True

def test_empty_expected_dtypes(sample_dataframe):
    expected_dtypes = {}
    assert validate_data_types(sample_dataframe, expected_dtypes) == True
