import pytest
import pandas as pd
from fairlearn.preprocessing import Reweighing
from definition_16566215de784f3abf9203c15c102459 import apply_reweighting

@pytest.fixture
def sample_dataframe():
    data = {'feature1': [1, 2, 3, 4, 5],
            'feature2': [6, 7, 8, 9, 10],
            'sensitive_attribute': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]}
    return pd.DataFrame(data)

def test_apply_reweighting_returns_reweighing_object_and_weights(sample_dataframe):
    reweighing, weights = apply_reweighting(sample_dataframe, 'sensitive_attribute')
    assert isinstance(reweighing, Reweighing)
    assert len(weights) == len(sample_dataframe)

def test_apply_reweighting_correct_weights_sum_to_one(sample_dataframe):
    _, weights = apply_reweighting(sample_dataframe, 'sensitive_attribute')
    assert abs(sum(weights) - len(sample_dataframe)) < 1e-6

def test_apply_reweighting_sensitive_attribute_not_in_dataframe(sample_dataframe):
    with pytest.raises(KeyError):
        apply_reweighting(sample_dataframe, 'non_existent_attribute')

def test_apply_reweighting_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        apply_reweighting(df, 'sensitive_attribute')

def test_apply_reweighting_numerical_sensitive_attribute(sample_dataframe):
    df = sample_dataframe.copy()
    df['sensitive_attribute'] = [1, 2, 1, 2, 1]
    reweighing, weights = apply_reweighting(df, 'sensitive_attribute')
    assert isinstance(reweighing, Reweighing)
    assert len(weights) == len(df)
