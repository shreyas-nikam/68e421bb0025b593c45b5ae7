import pytest
from definition_4ebf5869833346bca6d8408a3a8516fd import generate_synthetic_data
import pandas as pd

def test_generate_synthetic_data_output_type():
    """Test that the function returns a Pandas DataFrame."""
    data = generate_synthetic_data()
    assert isinstance(data, pd.DataFrame)

def test_generate_synthetic_data_columns():
    """Test that the DataFrame has the expected columns."""
    data = generate_synthetic_data()
    expected_columns = ['study_hours', 'prior_knowledge', 'access_to_resources', 'gender', 'final_grade']
    assert all(col in data.columns for col in expected_columns)

def test_generate_synthetic_data_non_empty():
    """Test that the DataFrame is not empty."""
    data = generate_synthetic_data()
    assert not data.empty

def test_generate_synthetic_data_reasonable_values():
    """Test that the values in the DataFrame are within reasonable ranges."""
    data = generate_synthetic_data()
    assert data['study_hours'].min() >= 0
    assert data['study_hours'].max() <= 100  # Assuming study hours are capped at 100
    assert data['prior_knowledge'].isin([0, 1]).all()
    assert data['access_to_resources'].isin([0, 1]).all()
    assert data['gender'].isin(['Male', 'Female']).all()

def test_generate_synthetic_data_data_types():
    """Test that the columns have the correct data types."""
    data = generate_synthetic_data()
    assert data['study_hours'].dtype == 'float64'  # Or int64, depending on implementation
    assert data['prior_knowledge'].dtype == 'int64'
    assert data['access_to_resources'].dtype == 'int64'
    assert data['gender'].dtype == 'object'  # String/object type
    assert data['final_grade'].dtype == 'float64'  # Or int64, depending on implementation
