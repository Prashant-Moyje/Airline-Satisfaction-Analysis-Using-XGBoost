# ✈️Airline Passenger Satisfaction Analysis
**[🚀 Test the Live Application Here](https://airline-satisfaction-analysis-using-xgboost-5wwfbcqwha5zs96zq8.streamlit.app/)**

Predicting passenger satisfaction, uncovering the key service drivers behind it, and turning those insights into segmentation, explainability, and an experiment design an airline can actually act on.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

**Business goal:** Predict whether a passenger is *Satisfied* or *Neutral/Dissatisfied*, identify the key drivers of satisfaction, and segment customers by their likelihood of satisfaction — so the airline can prioritize service investments and target at-risk customers.

**Dataset:** 129,880 passenger records with demographics, travel details, and 14 service ratings (0–5 scale).

## 🧭 Pipeline

1. Data loading & cleaning
2. Exploratory Data Analysis (EDA)
3. Preprocessing (encoding, scaling, train/test split)
4. Model training — Logistic Regression, Random Forest, XGBoost
5. Model evaluation & comparison
6. Feature importance / driver analysis
7. Customer segmentation by predicted satisfaction probability
8. Business recommendations
9. **Extended analysis:** hyperparameter tuning (Optuna), per-customer explainability (SHAP), A/B test design, and a retraining/drift-monitoring template

## 🔑 Key Findings

- **`Online Boarding` is the single strongest driver** of satisfaction — both in raw correlation and model feature importance.
- `In-flight Wifi Service`, `Seat Comfort`, `On-board Service`, and `Leg Room Service` round out the top drivers.
- **Flight delays matter far less than expected** — only a weak correlation with satisfaction, suggesting service quality outweighs punctuality here.
- Business travelers and Business-class passengers are markedly more satisfied than Personal-travel/Economy passengers.
- The best model (tuned XGBoost) achieves **~0.995 ROC-AUC** on held-out test data.

## 📁 Repository Structure

```
.
├── Airline_Satisfaction_Analysis.ipynb   # Main analysis notebook (EDA → modeling → SHAP → A/B test)
├── DS-DATA.csv                           # Raw dataset (not included — see Data section)
├── requirements.txt                      # Python dependencies
├── README.md
└── LICENSE
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/Prashant-Moyje/airline-satisfaction-analysis.git
cd airline-satisfaction-analysis
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Data

Place the source file `DS-DATA.csv` in the project root before running the notebook. The dataset contains passenger demographics, travel details, and 14 service ratings (0–5 scale), plus a `Satisfaction` target column. The notebook handles known data-quality issues automatically:

- Stray characters in `Flight Distance` → coerced to numeric, missing values filled with median
- Missing `Arrival Delay` values → filled with median
- Non-predictive `ID` column → dropped

### Usage

```bash
jupyter notebook Airline_Satisfaction_Analysis.ipynb
```

Run all cells top to bottom. The notebook is self-contained and produces every chart, metric, and table referenced below.

## 🧪 Models Trained

| Model | Role |
|---|---|
| Logistic Regression | Interpretable baseline (scaled features) |
| Random Forest | Non-linear, interaction-aware, robust |
| XGBoost | Best raw performance on tabular data |
| XGBoost (Optuna-tuned) | Final model of record |

Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC — plus confusion matrix and ROC curves for the best model.

## 🎯 Beyond the Baseline Model

- **Hyperparameter tuning:** Bayesian optimization with Optuna (25 trials, 3-fold CV on ROC-AUC).
- **SHAP explainability:** Global summary/dependence plots plus a per-customer waterfall plot, so a retention team can see *why* one specific at-risk passenger is predicted to churn.
- **Customer segmentation:** Predicted probabilities bucketed into `High Likelihood`, `At-Risk`, and `Low Likelihood` segments, profiled by their average top-driver ratings.
- **A/B test design:** Sample-size/power calculation for a boarding-process redesign, plus a ready-to-use significance-test script (currently run on simulated data as a template).

## 💡 Business Recommendations

1. **Fix Online Boarding first** — the single strongest driver of satisfaction; a digital-boarding UX overhaul likely has the highest ROI of any single initiative.
2. **Invest in in-flight Wifi** — consistently rated low and a top-3 driver.
3. **Differentiate the Personal-travel/Economy experience** — the widest satisfaction gap the airline directly controls.
4. **Target the "At-Risk" segment** — proactive outreach here is cheaper than winning back a fully lost customer.
5. **Don't over-index on delays** — punctuality matters for other KPIs, but it's a weak lever for satisfaction specifically.

## 🛠️ Tech Stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `optuna` · `shap` · `statsmodels` · `matplotlib` · `seaborn`

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙋 Contributing

Issues and pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.
