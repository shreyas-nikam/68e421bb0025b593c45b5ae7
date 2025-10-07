import pytest
import pandas as pd
from definition_1be65a9d9a304540b86dae7a9a88bcfe import validate_column_names

def test_validate_column_names_valid():
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    expected_columns = ['col1', 'col2']
    assert validate_column_names(df, expected_columns) == True

def test_validate_column_names_invalid():
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    expected_columns = ['col1', 'col3']
    assert validate_column_names(df, expected_columns) == False

def test_validate_column_names_empty_dataframe():
    df = pd.DataFrame()
    expected_columns = ['col1', 'col2']
    assert validate_column_names(df, expected_columns) == False

def test_validate_column_names_empty_expected_columns():
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    expected_columns = []
    assert validate_column_names(df, expected_columns) == True

def test_validate_column_names_different_order():
    df = pd.DataFrame({'col2': [3, 4], 'col1': [1, 2]})
    expected_columns = ['col1', 'col2']
    assert validate_column_names(df, expected_columns) == True
