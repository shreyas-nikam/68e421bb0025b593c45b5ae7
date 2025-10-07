
import streamlit as st
import pandas as pd # Added for general use, especially for df operations in pages
import numpy as np # Added for general use, especially for numerical operations in pages

st.set_page_config(page_title="FairAIED: Navigating Fairness, Bias, and Ethics in Educational AI Applications", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: FairAIED")
st.divider()
st.markdown("""
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
""")

# Your code starts here
page = st.sidebar.selectbox(
    label="Navigation",
    options=["Overview", "Data & Visualizations", "Model Training & Fairness"]
)

if page == "Overview":
    from application_pages.page1_overview import run_page1
    run_page1()
elif page == "Data & Visualizations":
    from application_pages.page2_data_viz import run_page2
    run_page2()
elif page == "Model Training & Fairness":
    from application_pages.page3_model_fairness import run_page3
    run_page3()
# Your code ends
