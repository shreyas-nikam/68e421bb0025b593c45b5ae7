
# Streamlit Application Requirements Specification

## Application Overview
This Streamlit application, "Ethical AIED Design Assistant," aims to provide a structured and interactive workflow for understanding, quantifying, and mitigating biases in Artificial-Intelligence-Enabled Education (AIED) systems. It is designed for data scientists and educators.

**Learning Goals:**
-   Understand the key insights contained in the uploaded document and supporting data.
-   Identify sources of bias in educational data and AI models.
-   Quantify fairness using standard metrics like Statistical Parity Difference, Equal Opportunity Difference, and Predictive Parity Difference.
-   Apply and evaluate various bias mitigation strategies (pre-processing, in-processing, post-processing).
-   Visualize the trade-offs between model accuracy and fairness using the Fairness-Bonded Utility (FBU) framework.
-   Interactively explore the impact of model hyperparameters and fairness constraints on ethical outcomes.

## User Interface Requirements

### Layout and Navigation Structure
The application will utilize a multi-section layout, guiding the user through the workflow sequentially. A sidebar will house global controls and navigation. Main content areas will display explanations, data previews, visualizations, and model outputs. Sections will be clearly demarcated with headings and subheadings.

### Input Widgets and Controls
1.  **Data Source Selection:**
    -   `st.radio`: "Choose Data Source" (options: "Generate Synthetic Data", "Upload Custom Data (CSV)").
2.  **Synthetic Data Generation Parameters (conditional on "Generate Synthetic Data"):**
    -   `st.slider`: "Number of Samples" (e.g., `min_value=50, max_value=1000, value=100, step=50`), with a tooltip.
    -   `st.number_input`: "Random Seed" (e.g., `min_value=1, value=42, step=1`), with a tooltip.
3.  **Custom Data Upload (conditional on "Upload Custom Data"):**
    -   `st.file_uploader`: "Upload CSV File" (`type=['csv']`), with a tooltip for expected format.
4.  **Model Training Parameters:**
    -   `st.slider`: "Test Set Size" (e.g., `min_value=0.1, max_value=0.5, value=0.2, step=0.05`), with a tooltip.
    -   `st.number_input`: "Random State for Train-Test Split" (e.g., `min_value=1, value=42, step=1`), with a tooltip.
5.  **Interactive Model Tuning Parameters (in a dedicated section):**
    -   `st.slider`: "Regularization (C)" (e.g., `min_value=0.01, max_value=10.0, value=1.0, step=0.01`), with a tooltip describing C-value.
    -   `st.selectbox`: "Sensitive Attribute" (options: `['gender_encoded', 'prior_knowledge', 'access_to_resources']`), with a tooltip.

### Visualization Components
1.  **Data Previews:**
    -   `st.dataframe`: Display `df.head()` and descriptive statistics table (`desc_stats`).
2.  **Exploratory Data Analysis Plots:**
    -   `st.pyplot`: Matplotlib-based Histograms for numeric features (`study_hours`, `prior_knowledge`, `access_to_resources`, `final_grade`). Plots will use a color-blind-friendly palette and ensure font size $\ge 12$ pt for titles and labels.
    -   `st.plotly_chart`: Plotly-based Scatter plot of `study_hours` vs `final_grade` colored by `gender`, enabling interactivity.
    -   `st.plotly_chart`: Plotly-based Bar chart of `Average Final Grade by Gender`, enabling interactivity.
3.  **Fairness Metrics and FBU Visualization:**
    -   `st.dataframe`: Table comparing accuracy, statistical parity, equal opportunity, and predictive parity for all mitigation methods.
    -   `st.dataframe`: Table showing FBU scores.
    -   `st.pyplot`: Matplotlib-based Scatter plot of 'Accuracy vs. Statistical Parity Difference' with thresholds, using a color-blind-friendly palette.

### Interactive Elements and Feedback Mechanisms
-   All sliders and dropdowns will trigger re-computation and re-rendering of relevant results and visualizations.
-   Tooltips or inline help text (`help` parameter in Streamlit widgets) will be provided for all input controls as per user requirements.
-   Progress indicators (`st.spinner`, `st.progress`) will be used for longer computational steps (e.g., model training, data generation) to provide feedback to the user.
-   Error messages (`st.error`) will be displayed for invalid inputs or data issues.

## Additional Requirements

### Annotation and Tooltip Specifications
-   Every interactive input widget (sliders, dropdowns, number inputs, file uploader) will include a `help` parameter with a concise description of its purpose and impact.
-   Key concepts, especially fairness metrics and mitigation strategies, will be explained using `st.markdown` with clear, concise language and LaTeX-formatted mathematical equations, as found in the notebook's markdown cells.
-   Visualizations will have clear titles, axis labels, and legends with a font size $\ge 12$ pt.

### Save the States of the Fields Properly so that Changes are Not Lost
-   `st.session_state` will be extensively used to manage the state of all user inputs (sliders, dropdowns, text inputs). This ensures that selected parameters persist across reruns and interactions, maintaining a consistent user experience.
-   Intermediate computational results (e.g., processed DataFrame, trained models, calculated metrics) will be cached using `st.cache_data` or `st.cache_resource` to prevent unnecessary re-computation when only display properties or non-dependent inputs change.

## Notebook Content and Code Requirements

This section outlines the integration of the Jupyter Notebook's content and code into the Streamlit application. All markdown explanations will be rendered using `st.markdown`, and Python code snippets will be adapted for Streamlit's display and interactivity paradigm.

### Initial Application Setup and Introduction
*   **Markdown Content:**
    ```markdown
    # Ethical AIED Design Assistant: Streamlit Application

    ## Business Value
    Artificial‑Intelligence‑Enabled Education (AIED) systems promise personalized learning at scale, but they also risk embedding and amplifying societal biases.
    This application equips data scientists and educators with a **structured workflow** to:

    1.  **Generate synthetic educational data** that mimics real‑world student attributes.
    2.  **Validate and clean** the data, ensuring reproducibility.
    3.  **Train a baseline predictive model** (logistic regression) to forecast student success.
    4.  **Quantify fairness** across a sensitive attribute (gender).
    5.  **Apply pre‑, in‑, and post‑processing mitigation** strategies from the *fairlearn* library.
    6.  **Visualize trade‑offs** between accuracy and fairness using the Fairness‑Bonded Utility (FBU) framework.
    7.  **Interactively explore** how model hyper‑parameters and fairness constraints affect outcomes.

    By following this application, practitioners can **identify bias sources**, **mitigate them responsibly**, and **communicate trade‑offs** to stakeholders—critical for ethically responsible AIED deployments.
    ```
    *   **Streamlit Implementation:** `st.markdown()`
*   **Code for Package Installation:** (Not directly in `app.py`, but specifies `requirements.txt`)
    ```python
    # Required packages for Streamlit application (to be listed in requirements.txt)
    # pandas
    # numpy
    # scikit-learn
    # matplotlib
    # seaborn
    # fairlearn
    # plotly
    # streamlit
    ```

### 1. Imports and Environment Check
*   **Markdown Content:**
    ```markdown
    ## 1. Imports and Environment Check

    We import all libraries that will be used throughout the application and print their versions to ensure reproducibility.
    Knowing the exact library versions helps trace any unexpected behavior and guarantees that the results can be replicated by other users.
    ```
    *   **Streamlit Implementation:** `st.markdown()`
*   **Code Stub:**
    ```python
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    from fairlearn.preprocessing import Reweighing
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    from fairlearn.postprocessing import ThresholdOptimizer
    import plotly.express as px
    from fairlearn import __version__ as fairlearn_version

    st.write("### Library Versions")
    st.write(f"pandas: {pd.__version__}")
    st.write(f"numpy: {np.__version__}")
    st.write(f"matplotlib: {matplotlib.__version__}")
    st.write(f"seaborn: {sns.__version__}")
    st.write(f"scikit-learn: {sklearn.__version__}")
    st.write(f"fairlearn: {fairlearn_version}")
    st.write(f"plotly: {px.__version__}")
    # ipywidgets is not used in Streamlit directly.
    ```

### 2. Synthetic Data Generation
*   **Markdown Content:**
    ```markdown
    ## 2. Synthetic Data Generation

    We create a realistic synthetic dataset that contains:
    - **Numeric features**: `study_hours`, `prior_knowledge`, `access_to_resources`.
    - **Categorical feature**: `gender` (sensitive attribute).
    - **Target**: `final_grade` (continuous).

    The dataset is generated using the functions provided, and we later convert the continuous grade into a binary **pass/fail** label for classification.
    This synthetic data allows us to experiment with fairness techniques without accessing private student records.
    ```
    *   **Streamlit Implementation:** `st.markdown()`
*   **Code Stub:**
    ```python
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

    # User inputs for data generation
    st.sidebar.header("Data Configuration")
    data_source_option = st.sidebar.radio(
        "Choose Data Source:",
        ("Generate Synthetic Data", "Upload Custom Data (CSV)"),
        key="data_source_radio"
    )

    df = None
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
        df = generate_synthetic_data(num_samples=num_samples_input, random_seed=random_seed_input)
        st.write("Synthetic dataset head:")
        st.dataframe(df.head())
    else: # Upload Custom Data
        st.subheader("Upload Custom Data")
        uploaded_file = st.file_uploader(
            "Upload your CSV file", type=["csv"], key="file_uploader",
            help="Upload a CSV file with columns: 'study_hours', 'prior_knowledge', 'access_to_resources', 'gender', 'final_grade'."
        )
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded dataset head:")
            st.dataframe(df.head())
        else:
            st.info("Please upload a CSV file to proceed or switch to synthetic data generation.")
            # Provide a fallback for initial run if no file is uploaded.
            st.write("Generating a lightweight sample (100 rows) as a fallback.")
            df = generate_synthetic_data(num_samples=100, random_seed=42) # Lightweight sample
            st.dataframe(df.head())

    # Stop execution if df is still None (e.g., upload chosen but no file yet)
    if df is None:
        st.stop()
    ```

### 3. Data Validation
*   **Markdown Content:**
    ```markdown
    ## 3. Data Validation

    Before modeling we must ensure the data structure is correct:

    1.  **Column names** match expectations.
    2.  **Data types** are appropriate for each column.
    3.  **Missing values** are identified and handled.

    These checks prevent downstream errors and guarantee that the dataset is ready for analysis.
    ```
    *   **Streamlit Implementation:** `st.markdown()`
*   **Code Stub:**
    ```python
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

    col_names_valid, data_types_valid, missing_values = validate_data(df)
    st.write(f"Column names valid: {col_names_valid}")
    st.write(f"Data types valid: {data_types_valid}")
    st.write("Missing values per column:")
    st.dataframe(missing_values.to_frame()) # Convert Series to DataFrame for better display
    ```

### 4. Descriptive Statistics
*   **Markdown Content:**
    ```markdown
    ## 4. Descriptive Statistics

    We compute descriptive statistics for numeric columns to understand the distribution of student attributes and grades.
    The statistics include mean, median, standard deviation, quartiles, skewness, and kurtosis—useful for detecting outliers and assessing normality, which informs model selection and preprocessing steps.
    ```
    *   **Streamlit Implementation:** `st.markdown()`
*   **Code Stub:**
    ```python
    @st.cache_data
    def calculate_descriptive_statistics(df, numeric_columns):
        try:
            if df.empty or not numeric_columns:
                return pd.DataFrame()

            numeric_df = df[numeric_columns]
            numeric_df = numeric_df.select_dtypes(include=['number'])
            if numeric_df.empty:
                return pd.DataFrame()

            # Generate describe output and custom stats
            description = numeric_df.describe()
            skewness = pd.DataFrame(numeric_df.skew(), columns=['skew']).T
            kurtosis = pd.DataFrame(numeric_df.kurtosis(), columns=['kurt']).T

            # Combine them, ensuring correct index alignment
            result = pd.concat([description, skewness, kurtosis])
            return result.T # Transpose to match desired output format
        except Exception as e:
            st.error(f"Error calculating descriptive statistics: {e}")
            return pd.DataFrame()

    numeric_cols = ['study_hours', 'prior_knowledge', 'access_to_resources', 'final_grade']
    desc_stats = calculate_descriptive_statistics(df, numeric_cols)
    st.write("Descriptive statistics:")
    st.dataframe(desc_stats)
    ```

### 5. Data Visualization
*   **Markdown Content:**
    ```markdown
    ## 5. Data Visualization

    Visualizing data reveals patterns, correlations, and potential biases that may not be apparent from raw statistics.

    - **Histograms** show the distribution of numeric features.
    - A **scatter plot** between `study_hours` and `final_grade` colored by `gender` highlights any gender‑based performance gaps.
    - A **bar chart** compares average final grades across genders, providing a quick visual cue for bias.
    ```
    *   **Streamlit Implementation:** `st.markdown()`
*   **Code Stub:**
    ```python
    st.write("### Feature Distributions (Histograms)")
    fig_hist, axes = plt.subplots(1, 4, figsize=(16, 4))
    cols_to_plot = ['study_hours', 'prior_knowledge', 'access_to_resources', 'final_grade']
    for i, col in enumerate(cols_to_plot):
        sns.histplot(df[col], kde=True, ax=axes[i], palette="viridis")
        axes[i].set_title(f'Histogram of {col}', fontsize=12)
        axes[i].tick_params(axis='x', labelsize=10)
        axes[i].tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    st.write("### Study Hours vs Final Grade by Gender (Scatter Plot)")
    fig_scatter = px.scatter(df, x='study_hours', y='final_grade', color='gender',
                             title='Study Hours vs Final Grade by Gender',
                             color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_scatter)

    st.write("### Average Final Grade by Gender (Bar Chart)")
    avg_grade_by_gender = df.groupby('gender')['final_grade'].mean().reset_index()
    fig_bar = px.bar(avg_grade_by_gender, x='gender', y='final_grade',
                      title='Average Final Grade by Gender',
                      color='gender',
                      color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_bar)
    ```

### 6. Model Training (Initial)
*   **Markdown Content:**
    ```markdown
    ## 6. Model Training (Initial)

    We convert the continuous `final_grade` into a binary **pass/fail** label (`final_grade >= 70`), encode the categorical `gender`, split the data, and train a logistic regression classifier.
    Accuracy is computed on the test set, serving as the baseline performance metric before any fairness intervention.
    ```
    *   **Streamlit Implementation:** `st.markdown()`
*   **Code Stub:**
    ```python
    @st.cache_data
    def train_baseline_model(df, test_size, random_state):
        df_model = df.copy()
        df_model['pass'] = (df_model['final_grade'] >= 70).astype(int)
        le = LabelEncoder()
        df_model['gender_encoded'] = le.fit_transform(df_model['gender'])

        X = df_model[['study_hours', 'prior_knowledge', 'access_to_resources', 'gender_encoded']]
        y = df_model['pass']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)

        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        baseline_accuracy = accuracy_score(y_test, y_pred)
        return logreg, X_train, X_test, y_train, y_test, y_pred, baseline_accuracy

    st.subheader("Model Training Parameters")
    test_size_input = st.slider(
        "Test Set Size",
        min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="test_size_slider",
        help="Proportion of the dataset to include in the test split."
    )
    train_random_state_input = st.number_input(
        "Random state for train-test split",
        min_value=1, value=42, step=1, key="train_random_state_input",
        help="Controls the shuffling applied to the data before applying the split."
    )

    logreg, X_train, X_test, y_train, y_test, y_pred, baseline_accuracy = train_baseline_model(
        df, test_size=test_size_input, random_state=train_random_state_input
    )
    st.write(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    ```

### 7. Fairness Metric Calculation (Before Mitigation)
*   **Markdown Content:**
    ```markdown
    ## 7. Fairness Metric Calculation (Before Mitigation)

    We evaluate three key fairness metrics:

    1.  **Statistical Parity Difference**
        $$P(\hat{Y}=1 | A=1) - P(\hat{Y}=1 | A=0)$$

    2.  **Equal Opportunity Difference**
        $$P(\hat{Y}=1 | A=1, Y=1) - P(\hat{Y}=1 | A=0, Y=1)$$

    3.  **Predictive Parity Difference**
        $$|P(Y=1 | \hat{Y}=1, A=1) - P(Y=1 | \hat{Y}=1, A=0)|$$

    These metrics quantify disparities across the sensitive attribute `gender`. A value close to zero indicates parity. The functions below are taken from the provided code.
    ```
    *   **Streamlit Implementation:** `st.markdown()`
*   **Code Stub:**
    ```python
    @st.cache_data
    def calculate_statistical_parity_difference(y_true, y_pred, sensitive_features):
        group_0_idx = (sensitive_features == 0)
        group_1_idx = (sensitive_features == 1)
        group_0_pred_mean = y_pred[group_0_idx].mean() if np.any(group_0_idx) else 0.0
        group_1_pred_mean = y_pred[group_1_idx].mean() if np.any(group_1_idx) else 0.0
        return group_1_pred_mean - group_0_pred_mean

    @st.cache_data
    def calculate_equal_opportunity_difference(y_true, y_pred, sensitive_features):
        df_temp = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'sensitive_feature': sensitive_features})
        
        # True Positive Rate for group 0 (A=0, Y=1)
        tpr_group0_num = df_temp[(df_temp['sensitive_feature'] == 0) & (df_temp['y_true'] == 1) & (df_temp['y_pred'] == 1)].shape[0]
        tpr_group0_den = df_temp[(df_temp['sensitive_feature'] == 0) & (df_temp['y_true'] == 1)].shape[0]
        tpr_group0 = tpr_group0_num / tpr_group0_den if tpr_group0_den > 0 else 0.0

        # True Positive Rate for group 1 (A=1, Y=1)
        tpr_group1_num = df_temp[(df_temp['sensitive_feature'] == 1) & (df_temp['y_true'] == 1) & (df_temp['y_pred'] == 1)].shape[0]
        tpr_group1_den = df_temp[(df_temp['sensitive_feature'] == 1) & (df_temp['y_true'] == 1)].shape[0]
        tpr_group1 = tpr_group1_num / tpr_group1_den if tpr_group1_den > 0 else 0.0
        
        return tpr_group0 - tpr_group1 # Original notebook had tpr_group0 - tpr_group1. For diff, order matters.

    @st.cache_data
    def calculate_predictive_parity_difference(y_true, y_pred, sensitive_features):
        if len(y_true) == 0:
            return 0.0

        # P(Y=1 | Y_hat=1, A=0)
        group_0_pred_pos_idx = (sensitive_features == 0) & (y_pred == 1)
        ppv_group0 = np.mean(y_true[group_0_pred_pos_idx] == 1) if np.any(group_0_pred_pos_idx) else 0.0

        # P(Y=1 | Y_hat=1, A=1)
        group_1_pred_pos_idx = (sensitive_features == 1) & (y_pred == 1)
        ppv_group1 = np.mean(y_true[group_1_pred_pos_idx] == 1) if np.any(group_1_pred_pos_idx) else 0.0

        return abs(ppv_group1 - ppv_group0)


    sensitive_test = X_test['gender_encoded'].values
    stat_parity = calculate_statistical_parity_difference(y_test.values, y_pred, sensitive_test)
    eq_opportunity = calculate_equal_opportunity_difference(y_test.values, y_pred, sensitive_test)
    pred_parity = calculate_predictive_parity_difference(y_test.values, y_pred, sensitive_test)

    st.write(f"Statistical Parity Difference (Baseline): {stat_parity:.4f}")
    st.write(f"Equal Opportunity Difference (Baseline): {eq_opportunity:.4f}")
    st.write(f"Predictive Parity Difference (Baseline): {pred_parity:.4f}")
    ```

### 8. Fairness Mitigation (Pre‑processing – Reweighting)
*   **Markdown Content:**
    ```markdown
    ## 8. Fairness Mitigation (Pre‑processing – Reweighting)

    Reweighting assigns sample weights so that the weighted distribution of the sensitive attribute matches the overall distribution.
    We use **fairlearn.preprocessing.Reweighing** to compute weights, then retrain the logistic regression model with these weights.
    The goal is to reduce fairness gaps without drastically harming accuracy.
    ```
    *   **Streamlit Implementation:** `st.markdown()`
*   **Code Stub:**
    ```python
    @st.cache_data
    def apply_and_evaluate_reweighting(X_train, y_train, X_test, y_test, sensitive_test):
        train_df_for_reweighing = X_train.copy() # Make sure sensitive feature is in X_train for reweighing
        train_df_for_reweighing['pass'] = y_train
        # Renaming 'gender_encoded' to 'gender' to match expected 'attrs' parameter for Reweighing
        train_df_for_reweighing['gender'] = train_df_for_reweighing['gender_encoded']
        
        reweighing = Reweighing(attrs=['gender'])
        reweighing.fit(train_df_for_reweighing[['gender']], train_df_for_reweighing['pass']) # Fit with only sensitive attribute
        weights = reweighing.transform(train_df_for_reweighing[['gender']])

        logreg_rw = LogisticRegression(max_iter=1000)
        logreg_rw.fit(X_train, y_train, sample_weight=weights.iloc[:, 0])

        y_pred_rw = logreg_rw.predict(X_test)
        rw_accuracy = accuracy_score(y_test, y_pred_rw)

        stat_parity_rw = calculate_statistical_parity_difference(y_test.values, y_pred_rw, sensitive_test)
        eq_opportunity_rw = calculate_equal_opportunity_difference(y_test.values, y_pred_rw, sensitive_test)
        pred_parity_rw = calculate_predictive_parity_difference(y_test.values, y_pred_rw, sensitive_test)
        
        return rw_accuracy, stat_parity_rw, eq_opportunity_rw, pred_parity_rw

    rw_accuracy, stat_parity_rw, eq_opportunity_rw, pred_parity_rw = apply_and_evaluate_reweighting(
        X_train, y_train, X_test, y_test, sensitive_test
    )
    st.write(f"Accuracy after Reweighting: {rw_accuracy:.4f}")
    st.write(f"Statistical Parity Difference (Reweighting): {stat_parity_rw:.4f}")
    st.write(f"Equal Opportunity Difference (Reweighting): {eq_opportunity_rw:.4f}")
    st.write(f"Predictive Parity Difference (Reweighting): {pred_parity_rw:.4f}")
    ```

### 9. Fairness Mitigation (In‑processing – Exponentiated Gradient)
*   **Markdown Content:**
    ```markdown
    ## 9. Fairness Mitigation (In‑processing – Exponentiated Gradient)

    The **Exponentiated Gradient (EG)** algorithm iteratively adjusts the model to satisfy a chosen fairness constraint while optimizing accuracy.
    We demonstrate both **Demographic Parity** and **Equalized Odds** constraints. The EG solver returns a *fair* logistic regression model that can be evaluated like any other scikit‑learn estimator.
    ```
    *   **Streamlit Implementation:** `st.markdown()`
*   **Code Stub:**
    ```python
    @st.cache_data
    def apply_and_evaluate_eg(X_train, y_train, X_test, y_test, sensitive_train, sensitive_test):
        # EG with Demographic Parity
        constraint_dp = DemographicParity()
        eg_dp = ExponentiatedGradient(LogisticRegression(max_iter=1000), constraints=constraint_dp)
        eg_dp.fit(X_train, y_train, sensitive_features=sensitive_train)
        y_pred_eg_dp = eg_dp.predict(X_test)
        eg_dp_accuracy = accuracy_score(y_test, y_pred_eg_dp)
        stat_parity_eg_dp = calculate_statistical_parity_difference(y_test.values, y_pred_eg_dp, sensitive_test)
        eq_opportunity_eg_dp = calculate_equal_opportunity_difference(y_test.values, y_pred_eg_dp, sensitive_test)
        pred_parity_eg_dp = calculate_predictive_parity_difference(y_test.values, y_pred_eg_dp, sensitive_test)

        # EG with Equalized Odds
        constraint_eo = EqualizedOdds()
        eg_eo = ExponentiatedGradient(LogisticRegression(max_iter=1000), constraints=constraint_eo)
        eg_eo.fit(X_train, y_train, sensitive_features=sensitive_train)
        y_pred_eg_eo = eg_eo.predict(X_test)
        eg_eo_accuracy = accuracy_score(y_test, y_pred_eg_eo)
        stat_parity_eg_eo = calculate_statistical_parity_difference(y_test.values, y_pred_eg_eo, sensitive_test)
        eq_opportunity_eg_eo = calculate_equal_opportunity_difference(y_test.values, y_pred_eg_eo, sensitive_test)
        pred_parity_eg_eo = calculate_predictive_parity_difference(y_test.values, y_pred_eg_eo, sensitive_test)

        return (eg_dp_accuracy, stat_parity_eg_dp, eq_opportunity_eg_dp, pred_parity_eg_dp,
                eg_eo_accuracy, stat_parity_eg_eo, eq_opportunity_eg_eo, pred_parity_eg_eo)

    sensitive_train = X_train['gender_encoded'].values # Assuming gender_encoded is always the sensitive attribute for these methods
    (eg_dp_accuracy, stat_parity_eg_dp, eq_opportunity_eg_dp, pred_parity_eg_dp,
     eg_eo_accuracy, stat_parity_eg_eo, eq_opportunity_eg_eo, pred_parity_eg_eo) = apply_and_evaluate_eg(
        X_train, y_train, X_test, y_test, sensitive_train, sensitive_test
    )
    st.write(f"Accuracy (EG DP): {eg_dp_accuracy:.4f}")
    st.write(f"Statistical Parity (EG DP): {stat_parity_eg_dp:.4f}")
    st.write(f"Equal Opportunity (EG DP): {eq_opportunity_eg_dp:.4f}")
    st.write(f"Predictive Parity (EG DP): {pred_parity_eg_dp:.4f}")
    st.write(f"Accuracy (EG EO): {eg_eo_accuracy:.4f}")
    st.write(f"Statistical Parity (EG EO): {stat_parity_eg_eo:.4f}")
    st.write(f"Equal Opportunity (EG EO): {eq_opportunity_eg_eo:.4f}")
    st.write(f"Predictive Parity (EG EO): {pred_parity_eg_eo:.4f}")
    ```

### 10. Fairness Mitigation (Post‑processing – Threshold Optimizer)
*   **Markdown Content:**
    ```markdown
    ## 10. Fairness Mitigation (Post‑processing – Threshold Optimizer)

    Post‑processing adjusts the decision threshold for each sensitive group to balance fairness and accuracy.
    We use **fairlearn.postprocessing.ThresholdOptimizer** with the **Demographic Parity** constraint. The optimizer learns optimal thresholds that minimize the chosen loss while satisfying the fairness objective.
    ```
    *   **Streamlit Implementation:** `st.markdown()`
*   **Code Stub:**
    ```python
    @st.cache_data
    def apply_and_evaluate_threshold_optimizer(X_train, y_train, X_test, y_test, sensitive_train, sensitive_test):
        logreg_base = LogisticRegression(max_iter=1000)
        logreg_base.fit(X_train, y_train)

        to = ThresholdOptimizer(
            estimator=logreg_base,
            constraints=DemographicParity(),
            predict_method='predict',
            prefit=False
        )
        to.fit(X_train, y_train, sensitive_features=sensitive_train)

        y_pred_to = to.predict(X_test)
        to_accuracy = accuracy_score(y_test, y_pred_to)

        stat_parity_to = calculate_statistical_parity_difference(y_test.values, y_pred_to, sensitive_test)
        eq_opportunity_to = calculate_equal_opportunity_difference(y_test.values, y_pred_to, sensitive_test)
        pred_parity_to = calculate_predictive_parity_difference(y_test.values, y_pred_to, sensitive_test)
        
        return to_accuracy, stat_parity_to, eq_opportunity_to, pred_parity_to

    (to_accuracy, stat_parity_to, eq_opportunity_to, pred_parity_to) = apply_and_evaluate_threshold_optimizer(
        X_train, y_train, X_test, y_test, sensitive_train, sensitive_test
    )
    st.write(f"Accuracy (Threshold Optimizer): {to_accuracy:.4f}")
    st.write(f"Statistical Parity (TO): {stat_parity_to:.4f}")
    st.write(f"Equal Opportunity (TO): {eq_opportunity_to:.4f}")
    st.write(f"Predictive Parity (TO): {pred_parity_to:.4f}")
    ```

### 11. Fairness Metric Calculation (After Mitigation)
*   **Markdown Content:**
    ```markdown
    ## 11. Fairness Metric Calculation (After Mitigation)

    We gather the fairness metrics for each mitigation strategy and present them side‑by‑side.
    This comparison highlights the trade‑offs: some methods improve fairness more than others but may slightly reduce accuracy. Stakeholders can use these insights to select the most appropriate strategy for their educational context.
    ```
    *   **Streamlit Implementation:** `st.markdown()`
*   **Code Stub:**
    ```python
    @st.cache_data
    def compile_metrics_df(baseline_accuracy, stat_parity, eq_opportunity, pred_parity,
                           rw_accuracy, stat_parity_rw, eq_opportunity_rw, pred_parity_rw,
                           eg_dp_accuracy, stat_parity_eg_dp, eq_opportunity_eg_dp, pred_parity_eg_dp,
                           eg_eo_accuracy, stat_parity_eg_eo, eq_opportunity_eg_eo, pred_parity_eg_eo,
                           to_accuracy, stat_parity_to, eq_opportunity_to, pred_parity_to):
        methods = ['Baseline', 'Reweighting', 'EG DP', 'EG EO', 'Threshold Optimizer']
        accuracies = [baseline_accuracy, rw_accuracy, eg_dp_accuracy, eg_eo_accuracy, to_accuracy]
        stat_parity_vals = [stat_parity, stat_parity_rw, stat_parity_eg_dp, stat_parity_eg_eo, stat_parity_to]
        eq_opportunity_vals = [eq_opportunity, eq_opportunity_rw, eq_opportunity_eg_dp, eq_opportunity_eg_eo, eq_opportunity_to]
        pred_parity_vals = [pred_parity, pred_parity_rw, pred_parity_eg_dp, pred_parity_eg_eo, pred_parity_to]

        metrics_df = pd.DataFrame({
            'Method': methods,
            'Accuracy': accuracies,
            'Statistical Parity': stat_parity_vals,
            'Equal Opportunity': eq_opportunity_vals,
            'Predictive Parity': pred_parity_vals
        })
        return metrics_df

    metrics_df = compile_metrics_df(
        baseline_accuracy, stat_parity, eq_opportunity, pred_parity,
        rw_accuracy, stat_parity_rw, eq_opportunity_rw, pred_parity_rw,
        eg_dp_accuracy, stat_parity_eg_dp, eq_opportunity_eg_dp, pred_parity_eg_dp,
        eg_eo_accuracy, stat_parity_eg_eo, eq_opportunity_eg_eo, pred_parity_eg_eo,
        to_accuracy, stat_parity_to, eq_opportunity_to, pred_parity_to
    )
    st.dataframe(metrics_df)
    ```

### 12. Fairness Bonded Utility (FBU) Analysis
*   **Markdown Content:**
    ```markdown
    ## 12. Fairness Bonded Utility (FBU) Analysis

    The **Fairness‑Bonded Utility (FBU)** score combines accuracy and a fairness metric into a single scalar:
    $$\text{FBU} = \text{Accuracy} \times (1 - |\text{Fairness Gap}|)$$
    where the fairness gap is the absolute value of the chosen metric (e.g., statistical parity difference).
    We compute FBU for each mitigation strategy and plot the trade‑off between accuracy and fairness, helping decision‑makers visualise the cost of bias reduction.
    ```
    *   **Streamlit Implementation:** `st.markdown()`
*   **Code Stub:**
    ```python
    @st.cache_data
    def calculate_fairness_bonded_utility(accuracy, fairness_metric):
        return accuracy * (1 - abs(fairness_metric))

    fbu_scores = [calculate_fairness_bonded_utility(acc, sp)
                  for acc, sp in zip(metrics_df['Accuracy'], metrics_df['Statistical Parity'])]

    metrics_df['FBU'] = fbu_scores
    st.dataframe(metrics_df[['Method', 'Accuracy', 'Statistical Parity', 'FBU']])

    st.write("### Accuracy vs. Statistical Parity Difference (Trade-off Plot)")
    fig_fbu, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=metrics_df, x='Statistical Parity', y='Accuracy', hue='Method', s=100, ax=ax, palette="viridis")
    ax.set_title('Accuracy vs. Statistical Parity Difference', fontsize=14)
    ax.axhline(0.8, color='gray', linestyle='--', label='80% Accuracy Threshold')
    ax.axvline(0.05, color='gray', linestyle='--', label='5% Parity Gap Threshold')
    ax.legend(fontsize=10)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    st.pyplot(fig_fbu)
    plt.close(fig_fbu)
    ```

### 13. Interactive Exploration
*   **Markdown Content:**
    ```markdown
    ## 13. Interactive Exploration

    Users can adjust the regularization strength of the logistic regression and the sensitive attribute to consider.
    The application automatically retrains the model and updates fairness metrics, enabling real‑time exploration of how hyper‑parameters influence ethical outcomes. This interactivity supports *exploratory data analysis* and *stakeholder education*.
    ```
    *   **Streamlit Implementation:** `st.markdown()`
*   **Code Stub:**
    ```python
    st.subheader("Interactive Model Parameter Tuning")

    # Ensure X_test has 'gender_encoded' or other numeric columns for sensitive_attr selection
    df_interactive = df.copy()
    df_interactive['pass'] = (df_interactive['final_grade'] >= 70).astype(int)
    le = LabelEncoder()
    df_interactive['gender_encoded'] = le.fit_transform(df_interactive['gender'])

    X_interactive = df_interactive[['study_hours', 'prior_knowledge', 'access_to_resources', 'gender_encoded']]
    y_interactive = df_interactive['pass']

    X_train_int, X_test_int, y_train_int, y_test_int = train_test_split(
        X_interactive, y_interactive, test_size=test_size_input, random_state=train_random_state_input, stratify=y_interactive
    )

    C_value = st.slider(
        'Regularization (C):',
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.01,
        help='Inverse of regularization strength; smaller values specify stronger regularization.',
        key='interactive_c_slider'
    )

    sensitive_attr_options = ['gender_encoded', 'prior_knowledge', 'access_to_resources']
    sensitive_attr = st.selectbox(
        'Sensitive Attribute:',
        options=sensitive_attr_options,
        index=0, # Default to 'gender_encoded'
        help='Select the sensitive attribute for fairness calculation.',
        key='interactive_sensitive_attr_select'
    )

    @st.cache_data
    def update_and_display_streamlit(C_value, sensitive_attr_col, X_train_int, y_train_int, X_test_int, y_test_int):
        lr = LogisticRegression(C=C_value, max_iter=1000)
        lr.fit(X_train_int, y_train_int)
        y_pred_mod = lr.predict(X_test_int)
        
        acc = accuracy_score(y_test_int, y_pred_mod)
        
        # Calculate statistical parity using the selected sensitive attribute column
        # Ensure sensitive attribute column is available in X_test_int
        if sensitive_attr_col in X_test_int.columns:
            sp = calculate_statistical_parity_difference(y_test_int.values, y_pred_mod, X_test_int[sensitive_attr_col].values)
        else:
            sp = float('NaN') # Or handle as an error condition
            st.warning(f"Sensitive attribute '{sensitive_attr_col}' not found in test features for fairness calculation.")

        return acc, sp

    acc_interactive, sp_interactive = update_and_display_streamlit(C_value, sensitive_attr, X_train_int, y_train_int, X_test_int, y_test_int)
    st.write(f"**Current Model Performance (C={C_value}, Sensitive Attribute='{sensitive_attr}'):**")
    st.write(f"Accuracy: {acc_interactive:.4f}")
    if not np.isnan(sp_interactive):
        st.write(f"Statistical Parity Difference: {sp_interactive:.4f}")
    else:
        st.write("Statistical Parity Difference: Not Applicable (sensitive attribute column missing).")
    ```

### 14. Conclusion
*   **Markdown Content:**
    ```markdown
    ## 14. Conclusion

    This application has walked through the entire lifecycle of building an **ethical AIED system**:

    1.  **Data generation** and **validation** ensure a clean, bias‑aware starting point.
    2.  **Baseline modeling** establishes performance benchmarks.
    3.  **Fairness metrics** quantify disparities across sensitive groups.
    4.  **Pre‑, in‑, and post‑processing mitigation** techniques demonstrate practical ways to reduce bias.
    5.  **Fairness‑Bonded Utility** offers a holistic view of the trade‑off between accuracy and fairness.
    6.  **Interactive widgets** empower stakeholders to experiment with model hyper‑parameters and observe their ethical impact in real time.

    By iteratively applying these steps, practitioners can design AIED solutions that are not only accurate but also **fair, transparent, and accountable**—key pillars for responsible AI in education.
    ```
    *   **Streamlit Implementation:** `st.markdown()`

### 15. References
*   **Markdown Content:**
    ```markdown
    ## 15. References

    1. Fairlearn documentation: https://fairlearn.org/
    2. scikit‑learn documentation: https://scikit-learn.org/stable/
    3. Seaborn documentation: https://seaborn.pydata.org/
    4. Plotly documentation: https://plotly.com/python/
    5. ipywidgets documentation: https://ipywidgets.readthedocs.io/
    6. *Fairness‑Bonded Utility (FBU)* concept – derived from recent fairness‑in‑machine‑learning literature.
    ```
    *   **Streamlit Implementation:** `st.markdown()`
