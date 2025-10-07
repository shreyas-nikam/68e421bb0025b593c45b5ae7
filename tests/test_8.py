import pytest
import numpy as np
import pandas as pd
from definition_b85fa300669c4432ab96aafa03697f61 import calculate_predictive_parity_difference

@pytest.fixture
def sample_data():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    sensitive_features = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return y_true, y_pred, sensitive_features

def test_calculate_predictive_parity_difference_basic(sample_data):
    y_true, y_pred, sensitive_features = sample_data
    result = calculate_predictive_parity_difference(y_true, y_pred, sensitive_features)
    assert isinstance(result, float)

def test_calculate_predictive_parity_difference_no_difference(sample_data):
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    sensitive_features = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    result = calculate_predictive_parity_difference(y_true, y_pred, sensitive_features)
    assert result == 0.0

def test_calculate_predictive_parity_difference_opposite_predictions(sample_data):
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    sensitive_features = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    result = calculate_predictive_parity_difference(y_true, y_pred, sensitive_features)
    assert isinstance(result, float)

def test_calculate_predictive_parity_difference_pandas(sample_data):
    y_true, y_pred, sensitive_features = sample_data
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    sensitive_features = pd.Series(sensitive_features)
    result = calculate_predictive_parity_difference(y_true, y_pred, sensitive_features)
    assert isinstance(result, float)

def test_calculate_predictive_parity_difference_empty_arrays():
    y_true = np.array([])
    y_pred = np.array([])
    sensitive_features = np.array([])
    with pytest.raises(ValueError):
        calculate_predictive_parity_difference(y_true, y_pred, sensitive_features)
