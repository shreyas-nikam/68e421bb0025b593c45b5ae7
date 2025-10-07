
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Still needed for font settings in Plotly via Matplotlib config
import seaborn as sns # Still needed for color palettes/config even if not directly plotting
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.preprocessing import Reweighing
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer
import plotly.express as px
import plotly.graph_objects as go

def run_page3():
    st.title("Model Training & Fairness")

    # Ensure all necessary data is in session_state
    required_session_keys = ['df', 'X_train', 'X_test', 'y_train', 'y_test', 'sensitive_train', 'sensitive_test',
                             'test_size_input', 'train_random_state_input']
    for key in required_session_keys:
        if key not in st.session_state:
            st.error(f"Missing data: '{key}' not found in session state. Please go back to the 'Overview' page and ensure data is generated/uploaded.")
            return

    df = st.session_state['df']
    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']
    sensitive_train = st.session_state['sensitive_train']
    sensitive_test = st.session_state['sensitive_test']
    test_size_input = st.session_state['test_size_input']
    train_random_state_input = st.session_state['train_random_state_input']

    st.markdown("""
    ## 6. Model Training (Initial)

    We convert the continuous `final_grade` into a binary **pass/fail** label (`final_grade >= 70`), encode the categorical `gender`, split the data, and train a logistic regression classifier.
    Accuracy is computed on the test set, serving as the baseline performance metric before any fairness intervention.
    """)

    @st.cache_resource # Use cache_resource for models
    def train_baseline_model(X_train_data, y_train_data, X_test_data, y_test_data):
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_train_data, y_train_data)
        y_pred = logreg.predict(X_test_data)
        baseline_accuracy = accuracy_score(y_test_data, y_pred)
        return logreg, y_pred, baseline_accuracy

    logreg_baseline, y_pred_baseline, baseline_accuracy = train_baseline_model(
        X_train, y_train, X_test, y_test
    )
    st.write(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    
    st.markdown("""
    ## 7. Fairness Metric Calculation (Before Mitigation)

    We evaluate three key fairness metrics:

    1.  **Statistical Parity Difference**
        $$P(\hat{Y}=1 | A=1) - P(\hat{Y}=1 | A=0)$$

    2.  **Equal Opportunity Difference**
        $$P(\hat{Y}=1 | A=1, Y=1) - P(\hat{Y}=1 | A=0, Y=1)$$

    3.  **Predictive Parity Difference**
        $$|P(Y=1 | \hat{Y}=1, A=1) - P(Y=1 | \hat{Y}=1, A=0)|$$

    These metrics quantify disparities across the sensitive attribute `gender`. A value close to zero indicates parity. The functions below are taken from the provided code.
    """)

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


    stat_parity_baseline = calculate_statistical_parity_difference(y_test.values, y_pred_baseline, sensitive_test)
    eq_opportunity_baseline = calculate_equal_opportunity_difference(y_test.values, y_pred_baseline, sensitive_test)
    pred_parity_baseline = calculate_predictive_parity_difference(y_test.values, y_pred_baseline, sensitive_test)

    st.write(f"Statistical Parity Difference (Baseline): {stat_parity_baseline:.4f}")
    st.write(f"Equal Opportunity Difference (Baseline): {eq_opportunity_baseline:.4f}")
    st.write(f"Predictive Parity Difference (Baseline): {pred_parity_baseline:.4f}")

    st.markdown("""
    ## 8. Fairness Mitigation (Pre‑processing – Reweighting)

    Reweighting assigns sample weights so that the weighted distribution of the sensitive attribute matches the overall distribution.
    We use **fairlearn.preprocessing.Reweighing** to compute weights, then retrain the logistic regression model with these weights.
    The goal is to reduce fairness gaps without drastically harming accuracy.
    """)

    @st.cache_resource
    def apply_and_evaluate_reweighting(X_train_data, y_train_data, X_test_data, y_test_data, sensitive_test_data):
        train_df_for_reweighing = X_train_data.copy() # Make sure sensitive feature is in X_train for reweighing
        train_df_for_reweighing['pass'] = y_train_data
        # Renaming 'gender_encoded' to 'gender' to match expected 'attrs' parameter for Reweighing
        train_df_for_reweighing['gender'] = train_df_for_reweighing['gender_encoded']
        
        reweighing = Reweighing(attrs=['gender'])
        reweighing.fit(train_df_for_reweighing[['gender']], train_df_for_reweighing['pass']) # Fit with only sensitive attribute
        weights = reweighing.transform(train_df_for_reweighing[['gender']])

        logreg_rw = LogisticRegression(max_iter=1000)
        logreg_rw.fit(X_train_data, y_train_data, sample_weight=weights.iloc[:, 0])

        y_pred_rw = logreg_rw.predict(X_test_data)
        rw_accuracy = accuracy_score(y_test_data, y_pred_rw)

        stat_parity_rw = calculate_statistical_parity_difference(y_test_data.values, y_pred_rw, sensitive_test_data)
        eq_opportunity_rw = calculate_equal_opportunity_difference(y_test_data.values, y_pred_rw, sensitive_test_data)
        pred_parity_rw = calculate_predictive_parity_difference(y_test_data.values, y_pred_rw, sensitive_test_data)
        
        return rw_accuracy, stat_parity_rw, eq_opportunity_rw, pred_parity_rw

    rw_accuracy, stat_parity_rw, eq_opportunity_rw, pred_parity_rw = apply_and_evaluate_reweighting(
        X_train, y_train, X_test, y_test, sensitive_test
    )
    st.write(f"Accuracy after Reweighting: {rw_accuracy:.4f}")
    st.write(f"Statistical Parity Difference (Reweighting): {stat_parity_rw:.4f}")
    st.write(f"Equal Opportunity Difference (Reweighting): {eq_opportunity_rw:.4f}")
    st.write(f"Predictive Parity Difference (Reweighting): {pred_parity_rw:.4f}")

    st.markdown("""
    ## 9. Fairness Mitigation (In‑processing – Exponentiated Gradient)

    The **Exponentiated Gradient (EG)** algorithm iteratively adjusts the model to satisfy a chosen fairness constraint while optimizing accuracy.
    We demonstrate both **Demographic Parity** and **Equalized Odds** constraints. The EG solver returns a *fair* logistic regression model that can be evaluated like any other scikit‑learn estimator.
    """)

    @st.cache_resource
    def apply_and_evaluate_eg(X_train_data, y_train_data, X_test_data, y_test_data, sensitive_train_data, sensitive_test_data):
        # EG with Demographic Parity
        constraint_dp = DemographicParity()
        eg_dp = ExponentiatedGradient(LogisticRegression(max_iter=1000), constraints=constraint_dp)
        eg_dp.fit(X_train_data, y_train_data, sensitive_features=sensitive_train_data)
        y_pred_eg_dp = eg_dp.predict(X_test_data)
        eg_dp_accuracy = accuracy_score(y_test_data, y_pred_eg_dp)
        stat_parity_eg_dp = calculate_statistical_parity_difference(y_test_data.values, y_pred_eg_dp, sensitive_test_data)
        eq_opportunity_eg_dp = calculate_equal_opportunity_difference(y_test_data.values, y_pred_eg_dp, sensitive_test_data)
        pred_parity_eg_dp = calculate_predictive_parity_difference(y_test_data.values, y_pred_eg_dp, sensitive_test_data)

        # EG with Equalized Odds
        constraint_eo = EqualizedOdds()
        eg_eo = ExponentiatedGradient(LogisticRegression(max_iter=1000), constraints=constraint_eo)
        eg_eo.fit(X_train_data, y_train_data, sensitive_features=sensitive_train_data)
        y_pred_eg_eo = eg_eo.predict(X_test_data)
        eg_eo_accuracy = accuracy_score(y_test_data, y_pred_eg_eo)
        stat_parity_eg_eo = calculate_statistical_parity_difference(y_test_data.values, y_pred_eg_eo, sensitive_test_data)
        eq_opportunity_eg_eo = calculate_equal_opportunity_difference(y_test_data.values, y_pred_eg_eo, sensitive_test_data)
        pred_parity_eg_eo = calculate_predictive_parity_difference(y_test_data.values, y_pred_eg_eo, sensitive_test_data)

        return (eg_dp_accuracy, stat_parity_eg_dp, eq_opportunity_eg_dp, pred_parity_eg_dp,
                eg_eo_accuracy, stat_parity_eg_eo, eq_opportunity_eg_eo, pred_parity_eg_eo)

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

    st.markdown("""
    ## 10. Fairness Mitigation (Post‑processing – Threshold Optimizer)

    Post‑processing adjusts the decision threshold for each sensitive group to balance fairness and accuracy.
    We use **fairlearn.postprocessing.ThresholdOptimizer** with the **Demographic Parity** constraint. The optimizer learns optimal thresholds that minimize the chosen loss while satisfying the fairness objective.
    """)

    @st.cache_resource
    def apply_and_evaluate_threshold_optimizer(X_train_data, y_train_data, X_test_data, y_test_data, sensitive_train_data, sensitive_test_data):
        logreg_base = LogisticRegression(max_iter=1000)
        logreg_base.fit(X_train_data, y_train_data)

        to = ThresholdOptimizer(
            estimator=logreg_base,
            constraints=DemographicParity(),
            predict_method='predict',
            prefit=False
        )
        to.fit(X_train_data, y_train_data, sensitive_features=sensitive_train_data)

        y_pred_to = to.predict(X_test_data)
        to_accuracy = accuracy_score(y_test_data, y_pred_to)

        stat_parity_to = calculate_statistical_parity_difference(y_test_data.values, y_pred_to, sensitive_test_data)
        eq_opportunity_to = calculate_equal_opportunity_difference(y_test_data.values, y_pred_to, sensitive_test_data)
        pred_parity_to = calculate_predictive_parity_difference(y_test_data.values, y_pred_to, sensitive_test_data)
        
        return to_accuracy, stat_parity_to, eq_opportunity_to, pred_parity_to

    (to_accuracy, stat_parity_to, eq_opportunity_to, pred_parity_to) = apply_and_evaluate_threshold_optimizer(
        X_train, y_train, X_test, y_test, sensitive_train, sensitive_test
    )
    st.write(f"Accuracy (Threshold Optimizer): {to_accuracy:.4f}")
    st.write(f"Statistical Parity (TO): {stat_parity_to:.4f}")
    st.write(f"Equal Opportunity (TO): {eq_opportunity_to:.4f}")
    st.write(f"Predictive Parity (TO): {pred_parity_to:.4f}")

    st.markdown("""
    ## 11. Fairness Metric Calculation (After Mitigation)

    We gather the fairness metrics for each mitigation strategy and present them side‑by‑side.
    This comparison highlights the trade‑offs: some methods improve fairness more than others but may slightly reduce accuracy. Stakeholders can use these insights to select the most appropriate strategy for their educational context.
    """)

    @st.cache_data
    def compile_metrics_df(baseline_accuracy_val, stat_parity_val, eq_opportunity_val, pred_parity_val,
                           rw_accuracy_val, stat_parity_rw_val, eq_opportunity_rw_val, pred_parity_rw_val,
                           eg_dp_accuracy_val, stat_parity_eg_dp_val, eq_opportunity_eg_dp_val, pred_parity_eg_dp_val,
                           eg_eo_accuracy_val, stat_parity_eg_eo_val, eq_opportunity_eg_eo_val, pred_parity_eg_eo_val,
                           to_accuracy_val, stat_parity_to_val, eq_opportunity_to_val, pred_parity_to_val):
        methods = ['Baseline', 'Reweighting', 'EG DP', 'EG EO', 'Threshold Optimizer']
        accuracies = [baseline_accuracy_val, rw_accuracy_val, eg_dp_accuracy_val, eg_eo_accuracy_val, to_accuracy_val]
        stat_parity_vals = [stat_parity_val, stat_parity_rw_val, stat_parity_eg_dp_val, stat_parity_eg_eo_val, stat_parity_to_val]
        eq_opportunity_vals = [eq_opportunity_val, eq_opportunity_rw_val, eq_opportunity_eg_dp_val, eq_opportunity_eg_eo_val, eq_opportunity_to_val]
        pred_parity_vals = [pred_parity_val, pred_parity_rw_val, pred_parity_eg_dp_val, pred_parity_eg_eo_val, pred_parity_to_val]

        metrics_df = pd.DataFrame({
            'Method': methods,
            'Accuracy': accuracies,
            'Statistical Parity': stat_parity_vals,
            'Equal Opportunity': eq_opportunity_vals,
            'Predictive Parity': pred_parity_vals
        })
        return metrics_df

    metrics_df = compile_metrics_df(
        baseline_accuracy, stat_parity_baseline, eq_opportunity_baseline, pred_parity_baseline,
        rw_accuracy, stat_parity_rw, eq_opportunity_rw, pred_parity_rw,
        eg_dp_accuracy, stat_parity_eg_dp, eq_opportunity_eg_dp, pred_parity_eg_dp,
        eg_eo_accuracy, stat_parity_eg_eo, eq_opportunity_eg_eo, pred_parity_eg_eo,
        to_accuracy, stat_parity_to, eq_opportunity_to, pred_parity_to
    )
    st.dataframe(metrics_df)

    st.markdown("""
    ## 12. Fairness Bonded Utility (FBU) Analysis

    The **Fairness‑Bonded Utility (FBU)** score combines accuracy and a fairness metric into a single scalar:
    $$\text{FBU} = \text{Accuracy} \times (1 - |\text{Fairness Gap}|)$$
    where the fairness gap is the absolute value of the chosen metric (e.g., statistical parity difference).
    We compute FBU for each mitigation strategy and plot the trade‑off between accuracy and fairness, helping decision‑makers visualise the cost of bias reduction.
    """)

    @st.cache_data
    def calculate_fairness_bonded_utility(accuracy, fairness_metric):
        return accuracy * (1 - abs(fairness_metric))

    fbu_scores = [calculate_fairness_bonded_utility(acc, sp)
                  for acc, sp in zip(metrics_df['Accuracy'], metrics_df['Statistical Parity'])]

    metrics_df['FBU'] = fbu_scores
    st.dataframe(metrics_df[['Method', 'Accuracy', 'Statistical Parity', 'FBU']])

    st.write("### Accuracy vs. Statistical Parity Difference (Trade-off Plot)")
    
    fig_fbu = px.scatter(metrics_df, x='Statistical Parity', y='Accuracy', color='Method',
                         title='Accuracy vs. Statistical Parity Difference',
                         hover_name='Method', size_max=60,
                         color_discrete_sequence=px.colors.qualitative.Vivid) # Using Vivid for variety and contrast

    fig_fbu.add_hline(y=0.8, line_dash="dash", line_color="gray", annotation_text="80% Accuracy Threshold", 
                      annotation_position="bottom right", annotation_font_size=12)
    fig_fbu.add_vline(x=0.05, line_dash="dash", line_color="gray", annotation_text="5% Parity Gap Threshold",
                      annotation_position="top left", annotation_font_size=12)
    
    fig_fbu.update_layout(
        xaxis_title="Statistical Parity Difference",
        yaxis_title="Accuracy",
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        legend_title_font_size=14,
        legend_font_size=12,
        height=500
    )
    st.plotly_chart(fig_fbu)
    
    st.markdown("""
    ## 13. Interactive Exploration

    Users can adjust the regularization strength of the logistic regression and the sensitive attribute to consider.
    The application automatically retrains the model and updates fairness metrics, enabling real‑time exploration of how hyper‑parameters influence ethical outcomes. This interactivity supports *exploratory data analysis* and *stakeholder education*.
    """)

    st.subheader("Interactive Model Parameter Tuning")

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

    @st.cache_resource
    def update_and_display_streamlit(C_value_val, sensitive_attr_col, X_train_int, y_train_int, X_test_int, y_test_int):
        lr = LogisticRegression(C=C_value_val, max_iter=1000)
        lr.fit(X_train_int, y_train_int)
        y_pred_mod = lr.predict(X_test_int)
        
        acc = accuracy_score(y_test_int, y_pred_mod)
        
        # Calculate statistical parity using the selected sensitive attribute column
        # Ensure sensitive attribute column is available in X_test_int. X_test already contains all features.
        sp = calculate_statistical_parity_difference(y_test_int.values, y_pred_mod, X_test_int[sensitive_attr_col].values)

        return acc, sp

    acc_interactive, sp_interactive = update_and_display_streamlit(C_value, sensitive_attr, X_train, y_train, X_test, y_test)
    st.write(f"**Current Model Performance (C={C_value}, Sensitive Attribute='{sensitive_attr}'):**")
    st.write(f"Accuracy: {acc_interactive:.4f}")
    if not np.isnan(sp_interactive):
        st.write(f"Statistical Parity Difference: {sp_interactive:.4f}")
    else:
        st.write("Statistical Parity Difference: Not Applicable (sensitive attribute column missing).")

    st.markdown("""
    ## 14. Conclusion

    This application has walked through the entire lifecycle of building an **ethical AIED system**:

    1.  **Data generation** and **validation** ensure a clean, bias‑aware starting point.
    2.  **Baseline modeling** establishes performance benchmarks.
    3.  **Fairness metrics** quantify disparities across sensitive groups.
    4.  **Pre‑, in‑, and post‑processing mitigation** techniques demonstrate practical ways to reduce bias.
    5.  **Fairness‑Bonded Utility** offers a holistic view of the trade‑off between accuracy and fairness.
    6.  **Interactive widgets** empower stakeholders to experiment with model hyper‑parameters and observe their ethical impact in real time.

    By iteratively applying these steps, practitioners can design AIED solutions that are not only accurate but also **fair, transparent, and accountable**—key pillars for responsible AI in education.
    """)

    st.markdown("""
    ## 15. References

    1. Fairlearn documentation: https://fairlearn.org/
    2. scikit‑learn documentation: https://scikit-learn.org/stable/
    3. Seaborn documentation: https://seaborn.pydata.org/
    4. Plotly documentation: https://plotly.com/python/
    5. ipywidgets documentation: https://ipywidgets.readthedocs.io/
    6. *Fairness‑Bonded Utility (FBU)* concept – derived from recent fairness‑in‑machine‑learning literature.
    """)
