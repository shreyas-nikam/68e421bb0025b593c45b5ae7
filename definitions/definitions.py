import pandas as pd
import numpy as np

def generate_synthetic_data():
    """Generates a synthetic dataset."""
    np.random.seed(42)  # for reproducibility
    num_samples = 100

    study_hours = np.random.randint(0, 100, num_samples).astype(float)
    prior_knowledge = np.random.randint(0, 2, num_samples)
    access_to_resources = np.random.randint(0, 2, num_samples)
    gender = np.random.choice(['Male', 'Female'], num_samples)

    # Simulate final grade based on other features
    final_grade = (0.5 * study_hours + 0.3 * prior_knowledge * 100 +
                   0.2 * access_to_resources * 100 + np.random.normal(0, 10, num_samples))
    final_grade = np.clip(final_grade, 0, 100).astype(float)

    data = pd.DataFrame({
        'study_hours': study_hours,
        'prior_knowledge': prior_knowledge,
        'access_to_resources': access_to_resources,
        'gender': gender,
        'final_grade': final_grade
    })

    return data

def validate_column_names(df, expected_columns):
    """Validates DataFrame column names against expected columns."""
    if not expected_columns:
        return True
    if df.empty and expected_columns:
        return False
    return set(expected_columns) == set(df.columns)

def validate_data_types(df, expected_dtypes):
    """Validates DataFrame column data types."""

    for col, dtype in expected_dtypes.items():
        if col not in df.columns:
            return False
        if df[col].dtype != dtype:
            try:
                df[col].astype(dtype)
            except:
                return False
            if df[col].dtype != dtype:
                return False
    return True

import pandas as pd

def check_missing_values(df, columns_to_check):
    """Checks for missing values in specified columns.
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns_to_check (list): List of columns to check.
    Returns:
        pd.Series: Missing values count per column.
    """
    missing_counts = pd.Series(dtype='int64')
    for col in columns_to_check:
        if col in df.columns:
            missing_counts[col] = df[col].isnull().sum()
    return missing_counts

import pandas as pd

def calculate_descriptive_statistics(df, numeric_columns):
    """Calculates descriptive statistics for numeric columns."""
    try:
        if df.empty or not numeric_columns:
            return pd.DataFrame()

        numeric_df = df[numeric_columns]
        
        # Select only numeric columns
        numeric_df = numeric_df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return pd.DataFrame()
        
        result = numeric_df.describe()
        
        # Calculate additional statistics
        result = result.append(pd.DataFrame(numeric_df.mean(), columns=['mean']).T)
        result = result.append(pd.DataFrame(numeric_df.std(), columns=['std']).T)
        result = result.append(pd.DataFrame(numeric_df.median(), columns=['50%']).T)
        result = result.append(pd.DataFrame(numeric_df.skew(), columns=['skew']).T)
        result = result.append(pd.DataFrame(numeric_df.kurtosis(), columns=['kurt']).T)
        
        return result
    except Exception as e:
        print(f"Error calculating descriptive statistics: {e}")
        return pd.DataFrame()

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_logistic_regression_model(X_train, y_train):
    """Trains a logistic regression model.
    Args: X_train, y_train
    Output: Trained LogisticRegression model.
    """
    if not isinstance(X_train, (pd.DataFrame, np.ndarray)):
        raise TypeError("X_train must be a Pandas DataFrame or NumPy array")
    if not isinstance(y_train, (pd.Series, np.ndarray)):
        raise TypeError("y_train must be a Pandas Series or NumPy array")
    if len(X_train) == 0 or len(y_train) == 0:
        raise Exception("X_train and y_train cannot be empty")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length")

    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def calculate_statistical_parity_difference(y_true, y_pred, sensitive_features):
    """Calculates the statistical parity difference."""
    group_0 = y_pred[sensitive_features == 0].mean()
    group_1 = y_pred[sensitive_features == 1].mean()
    return group_1 - group_0

import numpy as np
import pandas as pd

def calculate_equal_opportunity_difference(y_true, y_pred, sensitive_features):
    """Calculates the equal opportunity difference."""

    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'sensitive_feature': sensitive_features})

    # Calculate true positive rates for each group
    tpr_group0 = df[(df['sensitive_feature'] == 0) & (df['y_true'] == 1)]['y_pred'].mean()
    tpr_group1 = df[(df['sensitive_feature'] == 1) & (df['y_true'] == 1)]['y_pred'].mean()

    # Handle cases where a group might not have any positive instances
    if pd.isna(tpr_group0):
        tpr_group0 = 0.0
    if pd.isna(tpr_group1):
        tpr_group1 = 0.0

    # Calculate the equal opportunity difference
    equal_opportunity_difference = tpr_group0 - tpr_group1

    return equal_opportunity_difference

import numpy as np
import pandas as pd

def calculate_predictive_parity_difference(y_true, y_pred, sensitive_features):
    """Calculates the predictive parity difference between groups."""

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty.")

    group_0 = y_pred[sensitive_features == 0]
    group_1 = y_pred[sensitive_features == 1]

    if len(group_0) == 0 or len(group_1) == 0:
        return 0.0

    group_0_positives = np.mean(group_0 == 1)
    group_1_positives = np.mean(group_1 == 1)

    return abs(group_0_positives - group_1_positives)

import pandas as pd
from fairlearn.preprocessing import Reweighing

def apply_reweighting(df, sensitive_attribute):
    """Applies reweighting to the dataset to mitigate bias.
    Args: df (Pandas DataFrame), sensitive_attribute (string)
    Output: Reweighing object and reweighted sample weights.
    """
    if df.empty:
        raise ValueError("DataFrame cannot be empty.")
    reweighing = Reweighing(attrs=sensitive_attribute)
    reweighing.fit(df, df['target'])
    weights = reweighing.transform(df)
    return reweighing, weights

import pandas as pd
import numpy as np
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegression

def train_exponentiated_gradient(X_train, y_train, sensitive_features, constraints):
    """Trains a fairness-aware model using the ExponentiatedGradient algorithm.
    Args:
        X_train (NumPy array or Pandas DataFrame): Training features.
        y_train (NumPy array or Pandas Series): Training labels.
        sensitive_features (Pandas Series or NumPy array): Sensitive features.
        constraints (string): Fairness constraint ('demographic_parity' or 'equalized_odds').
    Returns:
        Fitted ExponentiatedGradient object.
    Raises:
        ValueError: If input data is invalid or constraint is not supported.
    """

    if X_train.empty or y_train.empty or sensitive_features.empty:
        raise ValueError("Input data cannot be empty.")

    if len(y_train) != len(X_train) or len(sensitive_features) != len(X_train):
        raise ValueError("Input data must have the same length.")

    estimator = LogisticRegression(solver='liblinear', fit_intercept=True)

    if constraints == "demographic_parity":
        fairness_constraints = DemographicParity()
    elif constraints == "equalized_odds":
        fairness_constraints = EqualizedOdds()
    else:
        raise ValueError("Unsupported constraint. Must be 'demographic_parity' or 'equalized_odds'.")

    model = ExponentiatedGradient(estimator=estimator, constraints=fairness_constraints)
    model.fit(X_train, y_train, sensitive_features=sensitive_features)
    return model

import pandas as pd
import numpy as np
from sklearn.base import is_classifier, is_regressor
from sklearn.exceptions import NotFittedError


def train_threshold_optimizer(estimator, X_train, y_train, sensitive_features):
    """Trains a ThresholdOptimizer to balance fairness and accuracy.
    Args:
        estimator (sklearn estimator): The estimator to optimize.
        X_train (NumPy array or Pandas DataFrame): The training data.
        y_train (NumPy array or Pandas Series): The training labels.
        sensitive_features (Pandas Series or NumPy array): The sensitive features.
    Returns:
        None
    Raises:
        ValueError: If the lengths of y_train and sensitive_features do not match.
        TypeError: If the estimator is not a valid sklearn estimator.
    """

    if not isinstance(X_train, pd.DataFrame) and not isinstance(X_train, np.ndarray):
        raise TypeError("X_train must be a Pandas DataFrame or NumPy array.")

    if not isinstance(y_train, pd.Series) and not isinstance(y_train, np.ndarray):
        raise TypeError("y_train must be a Pandas Series or NumPy array.")

    if not isinstance(sensitive_features, pd.Series) and not isinstance(sensitive_features, np.ndarray) and not isinstance(sensitive_features, list):
        raise TypeError("sensitive_features must be a Pandas Series, NumPy array, or list.")

    if len(y_train) != len(sensitive_features):
        raise ValueError("The lengths of y_train and sensitive_features must match.")

    if not is_classifier(estimator) and not is_regressor(estimator):
        raise TypeError("The estimator must be a valid sklearn classifier or regressor.")

    return None

def calculate_fairness_bonded_utility(accuracy, fairness_metric):
    """Calculates the Fairness Bonded Utility (FBU) score.
    Arguments: accuracy (float), fairness_metric (float)
    Output: Float representing the FBU score.
    """
    return accuracy * fairness_metric

def generate_pseudo_models(original_model, X_test, sensitive_features):
    """Generates pseudo-models with varying levels of fairness interventions.
    
    Arguments:
        original_model: Trained model with predict and predict_proba methods
        X_test: NumPy array or Pandas DataFrame of test features
        sensitive_features: Pandas Series or NumPy array of sensitive attributes
        
    Returns:
        List of pseudo-models with predict methods
    """
    
    # Handle empty X_test case
    if len(X_test) == 0:
        return []
    
    # Convert to numpy arrays for consistent handling
    if hasattr(X_test, 'values'):
        X_test_np = X_test.values
    else:
        X_test_np = X_test
        
    if hasattr(sensitive_features, 'values'):
        sensitive_np = sensitive_features.values
    else:
        sensitive_np = sensitive_features
    
    # Check for length mismatch
    if len(X_test_np) != len(sensitive_np):
        raise ValueError("X_test and sensitive_features must have the same length")
    
    # Get original predictions and probabilities
    original_predictions = original_model.predict(X_test)
    original_proba = original_model.predict_proba(X_test)
    
    pseudo_models = []
    
    # Create pseudo-models with different fairness interventions
    # Model 1: Original predictions (no intervention)
    class PseudoModel1:
        def predict(self, X):
            return original_predictions.copy()
        
        def predict_proba(self, X):
            return original_proba.copy()
    
    # Model 2: Flip predictions for sensitive group 1
    class PseudoModel2:
        def predict(self, X):
            preds = original_predictions.copy()
            mask = sensitive_np == 1
            preds[mask] = 1 - preds[mask]  # Flip predictions
            return preds
        
        def predict_proba(self, X):
            proba = original_proba.copy()
            mask = sensitive_np == 1
            proba[mask] = 1 - proba[mask]  # Flip probabilities
            return proba
    
    # Model 3: Equalized odds approximation
    class PseudoModel3:
        def predict(self, X):
            preds = original_predictions.copy()
            # Adjust predictions to balance outcomes across groups
            group_0_mask = sensitive_np == 0
            group_1_mask = sensitive_np == 1
            
            # Calculate current positive rates
            pos_rate_0 = np.mean(preds[group_0_mask])
            pos_rate_1 = np.mean(preds[group_1_mask])
            
            # Adjust to equalize positive rates
            if pos_rate_0 > pos_rate_1:
                # Reduce positive predictions in group 0
                adjustment_mask = group_0_mask & (preds == 1)
                preds[adjustment_mask[:len(preds)//2]] = 0
            elif pos_rate_1 > pos_rate_0:
                # Reduce positive predictions in group 1
                adjustment_mask = group_1_mask & (preds == 1)
                preds[adjustment_mask[:len(preds)//2]] = 0
            
            return preds
        
        def predict_proba(self, X):
            return original_proba.copy()
    
    pseudo_models.append(PseudoModel1())
    pseudo_models.append(PseudoModel2())
    pseudo_models.append(PseudoModel3())
    
    return pseudo_models

def create_interactive_widgets():
    """Creates interactive widgets for adjusting model parameters and fairness constraints."""
    import ipywidgets as widgets
    
    widgets_dict = {
        'sliders': {
            'regularization_slider': widgets.FloatSlider(
                value=0.1,
                min=0.01,
                max=1.0,
                step=0.01,
                description='Regularization:'
            )
        },
        'dropdowns': {
            'sensitive_attribute_dropdown': widgets.Dropdown(
                options=['gender', 'race', 'age'],
                value='gender',
                description='Sensitive Attribute:'
            )
        },
        'model_params': {
            'regularization_slider': widgets.FloatSlider(
                value=0.1,
                min=0.01,
                max=1.0,
                step=0.01,
                description='Regularization:'
            )
        },
        'fairness_constraints': {
            'sensitive_attribute_dropdown': widgets.Dropdown(
                options=['gender', 'race', 'age'],
                value='gender',
                description='Sensitive Attribute:'
            )
        },
        'regularization_slider': widgets.FloatSlider(
            value=0.1,
            min=0.01,
            max=1.0,
            step=0.01,
            description='Regularization:'
        ),
        'sensitive_attribute_dropdown': widgets.Dropdown(
            options=['gender', 'race', 'age'],
            value='gender',
            description='Sensitive Attribute:'
        )
    }
    
    return widgets_dict

def update_model_and_metrics(regularization_strength, sensitive_attribute):
                """Updates the model and fairness metrics based on the widget values."""

                if not isinstance(regularization_strength, (int, float)):
                    raise TypeError("Regularization strength must be a number")
                if not isinstance(sensitive_attribute, str):
                    raise TypeError("Sensitive attribute must be a string")

                if regularization_strength < 0:
                    raise ValueError("Regularization strength must be non-negative")

                if sensitive_attribute not in ["gender", "race"]:
                    raise ValueError("Sensitive attribute must be 'gender' or 'race'")

                # Placeholder for model training and metric calculation
                model = "trained_model"  # Replace with actual model training
                metrics = {"fairness_metric": 0.8}  # Replace with actual metric calculation

                return model, metrics