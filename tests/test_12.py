import pytest
from definition_f57c574f9a614751970726829321ca05 import calculate_fairness_bonded_utility

@pytest.mark.parametrize("accuracy, fairness_metric, expected", [
    (0.8, 0.9, 0.72),
    (1.0, 1.0, 1.0),
    (0.0, 0.0, 0.0),
    (0.5, 0.5, 0.25),
    (0.7, 0.2, 0.14),
])
def test_calculate_fairness_bonded_utility(accuracy, fairness_metric, expected):
    assert calculate_fairness_bonded_utility(accuracy, fairness_metric) == expected
