# FairAIED – Ethical AI‑Enabled Education Dashboard  
**Streamlit Lab Project**  
*Version 1.0.0 – 2025‑10‑07*  

---

## 📌 Project Title & Description  

**FairAIED** (Fairness‑Aided Intelligent Education Design) is a **Streamlit‑based interactive laboratory** that walks users through the entire lifecycle of building an ethically‑aware AI‑enabled education system.  

- **Synthetic data generation** that mimics real student attributes.  
- **Data validation & cleaning** to guarantee reproducibility.  
- **Baseline logistic‑regression modeling** for student success prediction.  
- **Fairness metrics** (Statistical Parity, Equal Opportunity, Predictive Parity).  
- **Pre‑, in‑, and post‑processing mitigation** (Reweighting, Exponentiated Gradient, Threshold Optimizer).  
- **Fairness‑Bonded Utility (FBU)** analysis that visualises the trade‑off between accuracy and fairness.  
- **Interactive hyper‑parameter tuning** that lets stakeholders see the impact of regularisation and sensitive‑attribute choices in real time.  

> **Goal** – Empower data scientists, educators, and policy makers to design, evaluate, and communicate fair AI‑driven educational tools.

---

## 🚀 Features  

| Feature | Description |
|---------|-------------|
| **Synthetic Data Engine** | Generates realistic student data (study hours, prior knowledge, resources, gender, grade). |
| **Data Validation** | Checks column names, dtypes, missing values, and reports results. |
| **Model Training** | Logistic regression baseline with train‑test split, reproducible via seed. |
| **Fairness Metrics** | Statistical Parity, Equal Opportunity, Predictive Parity – computed before & after mitigation. |
| **Mitigation Strategies** | 1️⃣ Reweighting (pre‑processing) <br> 2️⃣ Exponentiated Gradient (in‑processing) <br> 3️⃣ Threshold Optimizer (post‑processing) |
| **FBU Analysis** | Combines accuracy & fairness into a single score; visualises trade‑offs. |
| **Interactive Tuning** | Slider for regularisation strength (C) and dropdown for sensitive attribute. |
| **Rich Visualisations** | Histograms, scatter plots, bar charts, and interactive Plotly dashboards. |
| **Modular Page Architecture** | Three pages: Overview, Data & Visualizations, Model Training & Fairness. |
| **Reproducibility** | All library versions logged; caching used for deterministic results. |

---

## 🛠️ Getting Started  

### Prerequisites  

| Item | Minimum Version | Notes |
|------|-----------------|-------|
| Python | 3.10+ | Tested on 3.10, 3.11, 3.12 |
| pip | – | Use `pip` or `pipx` |
| Git | – | Optional – for cloning the repo |

### Installation  

```bash
# 1. Clone the repository
git clone https://github.com/your-org/fairai-ed.git
cd fairai-ed

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Tip** – If you prefer a Conda environment, create one with `conda create -n fairai-ed python=3.11` and then run `pip install -r requirements.txt`.

### Running the App  

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.  
Navigate between pages using the sidebar.

---

## 📚 Usage Overview  

1. **Overview**  
   - View library versions.  
   - Generate synthetic data or upload a CSV.  
   - Validate the dataset.  
   - Set train‑test split parameters.  

2. **Data & Visualizations**  
   - Inspect descriptive statistics.  
   - Explore histograms, scatter plots, and bar charts.  

3. **Model Training & Fairness**  
   - Train baseline logistic regression.  
   - Compute fairness metrics.  
   - Apply mitigation strategies and compare results.  
   - Visualise FBU trade‑offs.  
   - Interactively tune hyper‑parameters.

---

## 📁 Project Structure  

```
fairai-ed/
├── app.py                                 # Main Streamlit entry point
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
├── application_pages/
│   ├── __init__.py
│   ├── page1_overview.py                # Overview page logic
│   ├── page2_data_viz.py                 # Data & visualizations page
│   └── page3_model_fairness.py           # Model training & fairness page
├── assets/
│   └── logo5.jpg                         # Sidebar logo
└── .gitignore
```

- **`app.py`** – Handles navigation and imports page modules.  
- **`application_pages/`** – Each page is a self‑contained module exposing a `run_pageX()` function.  
- **`assets/`** – Static files (images, CSS, etc.).  

---

## 🧰 Technology Stack  

| Category | Library | Purpose |
|----------|---------|---------|
| **Web UI** | Streamlit | Interactive dashboard framework |
| **Data Manipulation** | pandas, numpy | Data frames & numerical ops |
| **Visualization** | matplotlib, seaborn, plotly | Static & interactive plots |
| **Machine Learning** | scikit‑learn | Logistic regression & metrics |
| **Fairness** | fairlearn | Pre‑, in‑, post‑processing mitigation & metrics |
| **Caching** | Streamlit caching decorators | Speed up repeated computations |
| **Environment** | Python 3.10+, pip | Core runtime |
| **Deployment** | Streamlit Cloud / Docker (optional) | Production hosting |

---

## 🤝 Contributing  

We welcome contributions! Please follow these steps:

1. **Fork** the repository.  
2. Create a **feature branch** (`git checkout -b feature/your-feature`).  
3. Write tests (if applicable) and ensure all existing tests pass (`pytest`).  
4. Commit with clear messages and push.  
5. Open a **Pull Request** – describe the change and link any relevant issues.  

**Code of Conduct** – Please adhere to the [Contributor Covenant](https://www.contributor-covenant.org/).  

---

## 📄 License  

MIT License – see the [LICENSE](LICENSE) file for details.  

---

## 📬 Contact  

- **Project Lead** – *Jane Doe*  
  Email: jane.doe@example.com  
  GitHub: [@janedoe](https://github.com/janedoe)  

- **Documentation & Support** –  
  GitHub Issues: https://github.com/your-org/fairai-ed/issues  
  Discord: https://discord.gg/yourproject  

---

## 📖 References  

1. Fairlearn documentation – https://fairlearn.org/  
2. scikit‑learn – https://scikit-learn.org/stable/  
3. Seaborn – https://seaborn.pydata.org/  
4. Plotly – https://plotly.com/python/  
5. *Fairness‑Bonded Utility (FBU)* – Derived from recent fairness‑in‑ML literature.  

---

Happy building! 🚀

## License

## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@quantuniversity.com](mailto:info@quantuniversity.com)
