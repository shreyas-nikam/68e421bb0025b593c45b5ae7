import pytest
import pandas as pd
from definition_81622ebecb6441098994a991a5b45470 import calculate_descriptive_statistics

@pytest.fixture
def sample_dataframe():
    data = {'col1': [1, 2, 3, 4, 5],
            'col2': [6.0, 7.0, 8.0, 9.0, 10.0],
            'col3': ['a', 'b', 'c', 'd', 'e']}
    return pd.DataFrame(data)

def test_calculate_descriptive_statistics_valid_columns(sample_dataframe):
    numeric_columns = ['col1', 'col2']
    result = calculate_descriptive_statistics(sample_dataframe, numeric_columns)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (8, 2)
    assert 'mean' in result.index
    assert 'std' in result.index

def test_calculate_descriptive_statistics_empty_dataframe():
    df = pd.DataFrame()
    numeric_columns = ['col1', 'col2']
    result = calculate_descriptive_statistics(df, numeric_columns)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_calculate_descriptive_statistics_empty_column_list(sample_dataframe):
    numeric_columns = []
    result = calculate_descriptive_statistics(sample_dataframe, numeric_columns)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_calculate_descriptive_statistics_non_numeric_column(sample_dataframe):
    numeric_columns = ['col3']
    result = calculate_descriptive_statistics(sample_dataframe, numeric_columns)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_calculate_descriptive_statistics_mixed_columns(sample_dataframe):
    numeric_columns = ['col1', 'col3']
    result = calculate_descriptive_statistics(sample_dataframe, numeric_columns)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (8, 1)
    assert 'mean' in result.index
    assert 'std' in result.index
