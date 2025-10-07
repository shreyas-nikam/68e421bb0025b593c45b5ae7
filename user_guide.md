id: 68e421bb0025b593c45b5ae7_user_guide
summary: FairAIED: Navigating Fairness, Bias, and Ethics in Educational AI Applications User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# FairAIED: Navigating Fairness, Bias, and Ethics in Educational AI Applications

## 1. Overview – Setting the Stage
**Duration:** 5:00

The first page introduces the purpose of the application and establishes a reproducible environment.

- **Library Versions** – The app prints the exact versions of pandas, numpy, scikit‑learn, fairlearn, and Plotly. Knowing the versions guarantees that the results can be replicated by other users and helps debug version‑specific issues.
- **Synthetic Data Generation** – A realistic dataset is created with numeric features (`study_hours`, `prior_knowledge`, `access_to_resources`), a categorical sensitive attribute (`gender`), and a continuous target (`final_grade`). The continuous grade is later binarised into a pass/fail label. This allows experimentation without handling private student records.
- **Data Validation** – The app checks that column names, data types, and missing values match expectations. This step prevents downstream errors and ensures the data is ready for analysis.
- **Model Split Parameters** – Users can set the test‑set proportion and random seed in the sidebar. These values are stored in `st.session_state` so that subsequent pages can access the same split.

The Overview page gives users a clear understanding of the data source, the validation checks, and how the data will be split for modeling.



## 2. Data & Visualizations – Exploring the Dataset
**Duration:** 7:00

The second page focuses on descriptive statistics and visual exploration.

- **Descriptive Statistics** – The app displays mean, median, standard deviation, quartiles, skewness, and kurtosis for numeric columns. These metrics help identify outliers, assess normality, and decide on preprocessing steps.
- **Histograms** – For each numeric feature, a histogram with a kernel density estimate shows the distribution shape. This visual cue reveals whether the data is skewed or contains heavy tails.
- **Scatter Plot (Study Hours vs Final Grade)** – By coloring points by `gender`, users can spot potential performance gaps between male and female students.
- **Bar Chart (Average Final Grade by Gender)** – A quick visual comparison of mean grades across genders highlights any systematic bias.

These visualizations provide an intuitive sense of the data’s structure and potential fairness issues before any modeling is performed.



## 3. Model Training & Fairness – Building and Auditing an Ethical Model
**Duration:** 15:00

The third page is the core of the application, covering baseline modeling, fairness metrics, mitigation strategies, and interactive exploration.

### 3.1 Baseline Model
- A logistic regression classifier is trained on the training split.
- Accuracy on the test set is reported as the baseline performance.

### 3.2 Fairness Metrics (Before Mitigation)
Three key metrics quantify disparities across the sensitive attribute `gender`:

1. **Statistical Parity Difference**  
   $$P(\hat{Y}=1 | A=1) - P(\hat{Y}=1 | A=0)$$

2. **Equal Opportunity Difference**  
   $$P(\hat{Y}=1 | A=1, Y=1) - P(\hat{Y}=1 | A=0, Y=1)$$

3. **Predictive Parity Difference**  
   $$|P(Y=1 | \hat{Y}=1, A=1) - P(Y=1 | \hat{Y}=1, A=0)|$$

A value close to zero indicates parity. These metrics give a baseline snapshot of bias before any intervention.

### 3.3 Pre‑Processing Mitigation – Reweighting
- **Reweighting** assigns sample weights so that the weighted distribution of `gender` matches the overall distribution.
- The weighted logistic regression is retrained, and accuracy and fairness metrics are recomputed.

### 3.4 In‑Processing Mitigation – Exponentiated Gradient (EG)
- EG iteratively adjusts the model to satisfy a chosen fairness constraint while optimizing accuracy.
- Two constraints are demonstrated:
  - **Demographic Parity** (equal positive prediction rates across groups).
  - **Equalized Odds** (equal true‑positive and false‑positive rates across groups).
- Accuracy and fairness metrics are reported for each constraint.

### 3.5 Post‑Processing Mitigation – Threshold Optimizer
- The **Threshold Optimizer** learns group‑specific decision thresholds that satisfy the Demographic Parity constraint.
- Accuracy and fairness metrics are evaluated after threshold adjustment.

### 3.6 Fairness‑Bonded Utility (FBU) Analysis
- **FBU** combines accuracy and a fairness gap into a single score:  
  $$\text{FBU} = \text{Accuracy} \times (1 - |\text{Fairness Gap}|)$$
- A scatter plot shows the trade‑off between accuracy and statistical parity difference for each mitigation strategy, helping stakeholders visualise the cost of bias reduction.

### 3.7 Interactive Exploration
- Users can adjust the logistic regression regularization strength (`C`) and select a sensitive attribute (e.g., `gender_encoded`, `prior_knowledge`, or `access_to_resources`).
- The model is retrained in real time, and updated accuracy and statistical parity difference are displayed. This live feedback encourages experimentation and deepens understanding of how hyper‑parameters influence ethical outcomes.



## 4. Conclusion – Building Ethical AIED Systems
**Duration:** 3:00

The application guides users through the entire lifecycle of an educational AI system:

1. **Data Generation & Validation** – Ensures a clean, bias‑aware starting point.
2. **Baseline Modeling** – Provides performance benchmarks.
3. **Fairness Quantification** – Measures disparities across sensitive groups.
4. **Mitigation Techniques** – Demonstrates pre‑, in‑, and post‑processing methods to reduce bias.
5. **FBU Analysis** – Offers a holistic view of the trade‑off between accuracy and fairness.
6. **Interactive Tuning** – Empowers stakeholders to experiment with model hyper‑parameters and observe their ethical impact in real time.

By iteratively applying these steps, practitioners can design AIED solutions that are not only accurate but also **fair, transparent, and accountable**—key pillars for responsible AI in education.



## 5. References
1. Fairlearn documentation: https://fairlearn.org/  
2. scikit‑learn documentation: https://scikit-learn.org/stable/  
3. Seaborn documentation: https://seaborn.pydata.org/  
4. Plotly documentation: https://plotly.com/python/  
5. *Fairness‑Bonded Utility (FBU)* concept – derived from recent fairness‑in‑machine‑learning literature.  

