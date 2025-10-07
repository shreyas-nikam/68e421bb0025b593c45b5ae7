import pytest
from definition_49f13a5a085e4bfe97bd21c71a042695 import update_model_and_metrics

@pytest.mark.parametrize("regularization_strength, sensitive_attribute, expected_exception", [
    (0.1, "gender", None),  # Valid inputs
    (-0.5, "race", ValueError),  # Negative regularization strength
    (1.0, "invalid_attribute", ValueError),  # Invalid sensitive attribute
    (None, "gender", TypeError),  # None regularization strength
    (0.5, None, TypeError),  # None sensitive attribute
])

def test_update_model_and_metrics(regularization_strength, sensitive_attribute, expected_exception):
    try:
        result = update_model_and_metrics(regularization_strength, sensitive_attribute)
        assert expected_exception is None, f"Expected {expected_exception} but no exception was raised"
        assert isinstance(result, tuple), "Function should return a tuple of (model, metrics)"
        assert len(result) == 2, "Function should return exactly two elements: model and metrics"
    except Exception as e:
        assert expected_exception is not None, f"Unexpected exception {type(e).__name__}: {e}"
        assert isinstance(e, expected_exception), f"Expected {expected_exception} but got {type(e).__name__}"