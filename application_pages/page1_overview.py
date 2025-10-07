
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from fairlearn import __version__ as fairlearn_version

def run_page1():
    st.markdown("""
    ## 1. Imports and Environment Check

    We import all libraries that will be used throughout the application and print their versions to ensure reproducibility.
    Knowing the exact library versions helps trace any unexpected behavior and guarantees that the results can be replicated by other users.
    """)

    st.write("### Library Versions")
    st.write(f"pandas: {pd.__version__}")
    st.write(f"numpy: {np.__version__}")
    st.write(f"matplotlib: {matplotlib.__version__}")
    st.write(f"seaborn: {sns.__version__}")
    st.write(f"scikit-learn: {sklearn.__version__}")
    st.write(f"fairlearn: {fairlearn_version}")
    st.write(f"plotly: {px.__version__}")

    st.markdown("""
    ## 2. Synthetic Data Generation

    We create a realistic synthetic dataset that contains:
    - **Numeric features**: `study_hours`, `prior_knowledge`, `access_to_resources`.
    - **Categorical feature**: `gender` (sensitive attribute).
    - **Target**: `final_grade` (continuous).

    The dataset is generated using the functions provided, and we later convert the continuous grade into a binary **pass/fail** label for classification.
    This synthetic data allows us to experiment with fairness techniques without accessing private student records.
    """)

    @st.cache_data
    def generate_synthetic_data(num_samples=100, random_seed=42):
        """Generates a synthetic dataset."""
        np.random.seed(random_seed)
        study_hours = np.random.randint(0, 100, num_samples).astype(float)
        prior_knowledge = np.random.randint(0, 2, num_samples)
        access_to_resources = np.random.randint(0, 2, num_samples)
        gender = np.random.choice(['Male', 'Female'], num_samples)
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

    st.sidebar.header("Data Configuration")
    data_source_option = st.sidebar.radio(
        "Choose Data Source:",
        ("Generate Synthetic Data", "Upload Custom Data (CSV)"),
        key="data_source_radio"
    )

    df_local = None # Using df_local to avoid name collision with df from st.session_state

    if data_source_option == "Generate Synthetic Data":
        st.subheader("Synthetic Data Generation Parameters")
        num_samples_input = st.slider(
            "Number of samples for synthetic data",
            min_value=50, max_value=1000, value=100, step=50, key="num_samples_slider",
            help="Determines the number of rows in the generated dataset."
        )
        random_seed_input = st.number_input(
            "Random seed for reproducibility",
            min_value=1, value=42, step=1, key="random_seed_input",
            help="Ensures that the generated data is the same each time for a given seed."
        )
        df_local = generate_synthetic_data(num_samples=num_samples_input, random_seed=random_seed_input)
        st.write("Synthetic dataset head:")
        st.dataframe(df_local.head())
    else: # Upload Custom Data
        st.subheader("Upload Custom Data")
        uploaded_file = st.file_uploader(
            "Upload your CSV file", type=["csv"], key="file_uploader",
            help="Upload a CSV file with columns: 'study_hours', 'prior_knowledge', 'access_to_resources', 'gender', 'final_grade'."
        )
        if uploaded_file is not None:
            df_local = pd.read_csv(uploaded_file)
            st.write("Uploaded dataset head:")
            st.dataframe(df_local.head())
        else:
            st.info("Please upload a CSV file to proceed or switch to synthetic data generation.")
            # Provide a fallback for initial run if no file is uploaded.
            st.write("Generating a lightweight sample (100 rows) as a fallback.")
            df_local = generate_synthetic_data(num_samples=100, random_seed=42) # Lightweight sample
            st.dataframe(df_local.head())

    # Store df in session_state
    if df_local is not None:
        st.session_state['df'] = df_local
    else:
        st.error("No data available to process. Please generate synthetic data or upload a file.")
        return # Stop execution if no df

    # --- Data Validation Section ---
    st.markdown("""
    ## 3. Data Validation

    Before modeling we must ensure the data structure is correct:

    1.  **Column names** match expectations.
    2.  **Data types** are appropriate for each column.
    3.  **Missing values** are identified and handled.

    These checks prevent downstream errors and guarantee that the dataset is ready for analysis.
    """)

    @st.cache_data
    def validate_data(df):
        expected_columns = ['study_hours', 'prior_knowledge', 'access_to_resources', 'gender', 'final_grade']
        expected_dtypes = {
            'study_hours': np.float64,
            'prior_knowledge': np.int64,
            'access_to_resources': np.int64,
            'gender': object,
            'final_grade': np.float64
        }

        def validate_column_names(df, expected_columns):
            if not expected_columns: return True
            if df.empty and expected_columns: return False
            return set(expected_columns) == set(df.columns)

        def validate_data_types(df, expected_dtypes):
            for col, dtype in expected_dtypes.items():
                if col not in df.columns: return False
                # Check if current dtype matches expected, or if it can be safely cast
                if df[col].dtype != dtype:
                    try:
                        # Attempt conversion to see if it's compatible
                        df[col].astype(dtype)
                    except:
                        return False # Conversion failed, type is not valid
            return True

        def check_missing_values(df, columns_to_check):
            missing_counts = pd.Series(dtype='int64')
            for col in columns_to_check:
                if col in df.columns:
                    missing_counts[col] = df[col].isnull().sum()
            return missing_counts

        col_names_valid = validate_column_names(df, expected_columns)
        data_types_valid = validate_data_types(df, expected_dtypes)
        missing_values = check_missing_values(df, expected_columns)

        return col_names_valid, data_types_valid, missing_values

    col_names_valid, data_types_valid, missing_values = validate_data(st.session_state['df'])
    st.write(f"Column names valid: {col_names_valid}")
    st.write(f"Data types valid: {data_types_valid}")
    st.write("Missing values per column:")
    st.dataframe(missing_values.to_frame()) # Convert Series to DataFrame for better display

    # Model Training Parameters for train-test split (also used on later pages)
    # Stored in session_state to be accessible across pages
    if 'test_size_input' not in st.session_state:
        st.session_state['test_size_input'] = 0.2
    if 'train_random_state_input' not in st.session_state:
        st.session_state['train_random_state_input'] = 42

    st.sidebar.subheader("Model Split Parameters")
    st.session_state['test_size_input'] = st.sidebar.slider(
        "Test Set Size",
        min_value=0.1, max_value=0.5, value=st.session_state['test_size_input'], step=0.05, key="test_size_slider_sidebar",
        help="Proportion of the dataset to include in the test split."
    )
    st.session_state['train_random_state_input'] = st.sidebar.number_input(
        "Random state for train-test split",
        min_value=1, value=st.session_state['train_random_state_input'], step=1, key="train_random_state_input_sidebar",
        help="Controls the shuffling applied to the data before applying the split."
    )

    # Prepare data for modeling and store in session_state for other pages
    df_model_processed = st.session_state['df'].copy()
    df_model_processed['pass'] = (df_model_processed['final_grade'] >= 70).astype(int)
    le = LabelEncoder()
    df_model_processed['gender_encoded'] = le.fit_transform(df_model_processed['gender'])

    X = df_model_processed[['study_hours', 'prior_knowledge', 'access_to_resources', 'gender_encoded']]
    y = df_model_processed['pass']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=st.session_state['test_size_input'], random_state=st.session_state['train_random_state_input'], stratify=y)

    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    st.session_state['sensitive_train'] = X_train['gender_encoded'].values
    st.session_state['sensitive_test'] = X_test['gender_encoded'].values
