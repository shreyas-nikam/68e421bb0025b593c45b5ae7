import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from definition_7abf4de5410547038c983db2dac3c7f6 import generate_pseudo_models

@pytest.fixture
def mock_original_model():
    model = Mock()
    model.predict.return_value = np.array([0, 1, 0, 1])
    model.predict_proba.return_value = np.array([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7]])
    return model

@pytest.fixture
def sample_X_test():
    return np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

@pytest.fixture
def sample_sensitive_features():
    return np.array([0, 1, 0, 1])

def test_generate_pseudo_models_basic_functionality(mock_original_model, sample_X_test, sample_sensitive_features):
    """Test basic functionality with valid inputs"""
    result = generate_pseudo_models(mock_original_model, sample_X_test, sample_sensitive_features)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(hasattr(model, 'predict') for model in result)

def test_generate_pseudo_models_empty_X_test(mock_original_model, sample_sensitive_features):
    """Test with empty X_test array"""
    empty_X_test = np.array([]).reshape(0, 2)
    result = generate_pseudo_models(mock_original_model, empty_X_test, sample_sensitive_features)
    assert isinstance(result, list)
    assert len(result) == 0

def test_generate_pseudo_models_pandas_dataframe(mock_original_model, sample_sensitive_features):
    """Test with pandas DataFrame input"""
    X_test_df = pd.DataFrame({'feature1': [1, 3, 5, 7], 'feature2': [2, 4, 6, 8]})
    result = generate_pseudo_models(mock_original_model, X_test_df, sample_sensitive_features)
    assert isinstance(result, list)
    assert len(result) > 0

def test_generate_pseudo_models_pandas_series_sensitive(mock_original_model, sample_X_test):
    """Test with pandas Series sensitive features"""
    sensitive_series = pd.Series([0, 1, 0, 1])
    result = generate_pseudo_models(mock_original_model, sample_X_test, sensitive_series)
    assert isinstance(result, list)
    assert len(result) > 0

def test_generate_pseudo_models_mismatched_lengths(mock_original_model, sample_X_test):
    """Test with mismatched lengths between X_test and sensitive_features"""
    mismatched_sensitive = np.array([0, 1, 0])  # One less element
    with pytest.raises(ValueError):
        generate_pseudo_models(mock_original_model, sample_X_test, mismatched_sensitive)