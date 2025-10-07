import pytest
import pandas as pd
from definition_2045189c278243d5a3e4020b26d065f7 import check_missing_values

@pytest.fixture
def sample_dataframe():
    data = {'col1': [1, 2, None, 4], 
            'col2': ['a', None, 'c', 'd'], 
            'col3': [True, False, True, None]}
    return pd.DataFrame(data)

def test_check_missing_values_valid_columns(sample_dataframe):
    result = check_missing_values(sample_dataframe, ['col1', 'col2'])
    assert isinstance(result, pd.Series)
    assert result['col1'] == 1
    assert result['col2'] == 1

def test_check_missing_values_empty_dataframe():
    df = pd.DataFrame()
    result = check_missing_values(df, ['col1', 'col2'])
    assert isinstance(result, pd.Series)
    assert len(result) == 0

def test_check_missing_values_empty_column_list(sample_dataframe):
    result = check_missing_values(sample_dataframe, [])
    assert isinstance(result, pd.Series)
    assert len(result) == 0

def test_check_missing_values_all_missing(sample_dataframe):
    df = pd.DataFrame({'col1': [None, None, None]})
    result = check_missing_values(df, ['col1'])
    assert result['col1'] == 3

def test_check_missing_values_no_missing(sample_dataframe):
    df = pd.DataFrame({'col1': [1, 2, 3]})
    result = check_missing_values(df, ['col1'])
    assert result['col1'] == 0
