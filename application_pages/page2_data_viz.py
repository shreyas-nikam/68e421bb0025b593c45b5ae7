
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def run_page2():
    st.title("Data & Visualizations")

    if 'df' not in st.session_state:
        st.error("No data available. Please go to the 'Overview' page to generate or upload data.")
        return

    df = st.session_state['df']

    st.markdown("""
    ## 4. Descriptive Statistics

    We compute descriptive statistics for numeric columns to understand the distribution of student attributes and grades.
    The statistics include mean, median, standard deviation, quartiles, skewness, and kurtosis—useful for detecting outliers and assessing normality, which informs model selection and preprocessing steps.
    """)

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

    st.markdown("""
    ## 5. Data Visualization

    Visualizing data reveals patterns, correlations, and potential biases that may not be apparent from raw statistics.

    - **Histograms** show the distribution of numeric features.
    - A **scatter plot** between `study_hours` and `final_grade` colored by `gender` highlights any gender‑based performance gaps.
    - A **bar chart** compares average final grades across genders, providing a quick visual cue for bias.
    """)

    st.write("### Feature Distributions (Histograms)")
    fig_hist, axes = plt.subplots(1, 4, figsize=(16, 4))
    cols_to_plot = ['study_hours', 'prior_knowledge', 'access_to_resources', 'final_grade']
    
    # Set a color-blind friendly palette for matplotlib
    sns.set_palette("viridis") 

    for i, col in enumerate(cols_to_plot):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Histogram of {col}', fontsize=12)
        axes[i].set_xlabel(col, fontsize=12) # Ensure labels also have font size >= 12
        axes[i].set_ylabel('Frequency', fontsize=12)
        axes[i].tick_params(axis='x', labelsize=10)
        axes[i].tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    st.write("### Study Hours vs Final Grade by Gender (Scatter Plot)")
    fig_scatter = px.scatter(df, x='study_hours', y='final_grade', color='gender',
                             title='Study Hours vs Final Grade by Gender',
                             color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_scatter.update_layout(
        xaxis_title="Study Hours",
        yaxis_title="Final Grade",
        title_font_size=16, # Ensure title font size >= 12
        xaxis_title_font_size=14, # Ensure axis label font size >= 12
        yaxis_title_font_size=14, # Ensure axis label font size >= 12
        legend_title_font_size=14, # Ensure legend title font size >= 12
        legend_font_size=12 # Ensure legend item font size >= 12
    )
    st.plotly_chart(fig_scatter)

    st.write("### Average Final Grade by Gender (Bar Chart)")
    avg_grade_by_gender = df.groupby('gender')['final_grade'].mean().reset_index()
    fig_bar = px.bar(avg_grade_by_gender, x='gender', y='final_grade',
                      title='Average Final Grade by Gender',
                      color='gender',
                      color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_bar.update_layout(
        xaxis_title="Gender",
        yaxis_title="Average Final Grade",
        title_font_size=16, # Ensure title font size >= 12
        xaxis_title_font_size=14, # Ensure axis label font size >= 12
        yaxis_title_font_size=14, # Ensure axis label font size >= 12
        legend_title_font_size=14, # Ensure legend title font size >= 12
        legend_font_size=12 # Ensure legend item font size >= 12
    )
    st.plotly_chart(fig_bar)
