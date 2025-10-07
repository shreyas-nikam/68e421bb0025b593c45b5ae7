import pytest
from definition_1bd31670e9e34fe3a24750697de70ed9 import calculate_equal_opportunity_difference
import numpy as np
import pandas as pd

@pytest.mark.parametrize(
    "y_true, y_pred, sensitive_features, expected",
    [
        (np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]), 0.0),
        (np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0]), np.array([0, 0, 1, 1]), -1.0),
        (np.array([0, 1, 0, 1]), np.array([1, 1, 1, 1]), np.array([0, 0, 1, 1]), 0.5),
        (pd.Series([0, 1, 0, 1]), pd.Series([0, 1, 0, 1]), pd.Series([0, 0, 1, 1]), 0.0),
        (np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]), 0.0),
    ],
)
def test_calculate_equal_opportunity_difference(y_true, y_pred, sensitive_features, expected):
    try:
        result = calculate_equal_opportunity_difference(y_true, y_pred, sensitive_features)
        assert np.isclose(result, expected)
    except Exception as e:
        assert type(e) == expected
