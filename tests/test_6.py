import pytest
import numpy as np
import pandas as pd
from definition_a3ea902bff904dd9be439e835783a4a6 import calculate_statistical_parity_difference

@pytest.mark.parametrize(
    "y_true, y_pred, sensitive_features, expected",
    [
        (np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]), 0.0),
        (np.array([0, 1, 0, 1]), np.array([1, 1, 1, 1]), np.array([0, 0, 1, 1]), 0.5),
        (np.array([0, 1, 0, 1]), np.array([0, 0, 0, 0]), np.array([0, 0, 1, 1]), -0.5),
        (pd.Series([0, 1, 0, 1]), pd.Series([0, 1, 0, 1]), pd.Series([0, 0, 1, 1]), 0.0),
        (np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), 0.0), # Sensitive feature same as y_true
    ],
)
def test_calculate_statistical_parity_difference(y_true, y_pred, sensitive_features, expected):
    assert calculate_statistical_parity_difference(y_true, y_pred, sensitive_features) == expected
