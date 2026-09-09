# 📊 Customer Churn & Revenue Risk Analytics Platform

**An end-to-end analytics platform that predicts customer churn, quantifies revenue at risk in dollars, and generates data-driven retention recommendations — built to answer the question every subscription business actually asks: *"who's leaving, and what will it cost us?"***

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-Render-46E3B7?style=for-the-badge&logo=render)](YOUR_RENDER_URL_HERE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-EB0028?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

---

## 🎯 The Problem

Companies typically discover churn **after** a customer has already left. Retention teams need to know — *before* it happens — who is at risk, how much revenue that represents, and which lever (contract offer, support outreach, service fix) is most likely to work. Most churn projects stop at a model with an accuracy score and no path to a business decision.

**This project builds the full workflow a real Customer Analytics team would use:** segment customers by value and behavior → predict churn probability → translate that probability into dollars → explain *why* the model thinks someone will churn → package it as a live tool a stakeholder can open and act on.

---

## 🔗 Live Demo

**Dashboard:** [YOUR_RENDER_URL_HERE](YOUR_RENDER_URL_HERE)
**Source:** [github.com/KirtanPatel30/churn-revenue-analytics](https://github.com/KirtanPatel30/churn-revenue-analytics)

> Note: hosted on Render's free tier — the app may take 30–60 seconds to wake up on first load after a period of inactivity.

---

## 🖼️ Preview

| Overview | Churn Drivers |
|---|---|
| KPIs, segment mix, revenue-at-risk by segment | SHAP-driven churn explainability |

| Deep Dive | Recommendations |
|---|---|
| Scatter, histogram, contract/service churn breakdown | Data-backed retention actions |

*(Add screenshots here — drag 4 PNGs into the repo's `assets/screenshots/` folder and reference them, e.g. `![Overview](assets/screenshots/overview.png)`)*

---

## 💡 What It Does

| Capability | Detail |
|---|---|
| **Segmentation** | CLV-adapted RFM model — 5 named business segments (Champions, Loyal High-Value, At Risk, Price-Sensitive/New, Low-Value/Disengaged) |
| **Prediction** | XGBoost classifier, ROC-AUC **0.844**, wrapped in a scikit-learn pipeline with proper train/test isolation |
| **Explainability** | SHAP `TreeExplainer` values — every prediction is auditable, not a black box |
| **Revenue Quantification** | `churn probability × annualized charges` per customer, rolled up by segment — risk expressed in dollars, not just probability |
| **Interactive Dashboard** | 4-tab Dash app with live filters, 6 chart types (pie, bar, box, scatter, histogram, grouped bar), and written recommendations |
| **Deployment** | Fully containerized (Docker) and deployed live on Render |

---

## 📈 Key Results

- **$389K+** in annualized revenue identified as at-risk in the highest-churn segments
- **Month-to-month contracts** identified as the single largest addressable churn driver via SHAP analysis
- **0.844 ROC-AUC / 0.58 F1** — a realistic, non-overfit result on a genuinely noisy real-world problem
- **4 specific, quantified retention recommendations** generated directly from model output, not generic advice

---

## 🏗️ Architecture

```
Raw CSV (7,043 customers)
        │
        ▼
┌───────────────────┐
│  preprocessing.py  │  → cleans data, engineers features (CLV, tenure buckets, avg spend)
└─────────┬──────────┘
          ▼
┌───────────────────┐
│  segmentation.py   │  → CLV-adapted RFM scoring → 5 business segments
└─────────┬──────────┘
          ▼
┌───────────────────┐
│     model.py       │  → XGBoost pipeline + SHAP explainability → scored dataset
└─────────┬──────────┘
          ▼
┌───────────────────┐
│      app.py        │  → Dash dashboard: KPIs, 4 tabs, live filters
└─────────┬──────────┘
          ▼
   Docker → Render (live, public URL)
```

---

## 🛠️ Tech Stack

**Language & Core:** Python 3.11
**Data & ML:** pandas, NumPy, scikit-learn, XGBoost, SHAP
**Visualization:** Plotly, Dash
**Deployment:** Docker, Gunicorn, Render

---

## 📁 Project Structure

```
churn-revenue-analytics/
├── data/
│   └── Telco-Customer-Churn.csv      # raw dataset (IBM Telco Churn, 7,043 customers)
├── src/
│   ├── preprocessing.py              # cleaning + feature engineering
│   ├── segmentation.py               # CLV-adapted RFM segmentation
│   ├── model.py                      # XGBoost training + SHAP
│   └── app.py                        # Dash dashboard (entry point)
├── models/                           # generated: churn_model.pkl, feature_importance.csv
├── assets/
│   └── style.css                     # dashboard styling
├── run_pipeline.py                   # orchestrates preprocessing → segmentation → model
├── requirements.txt
├── Dockerfile
├── render.yaml
└── README.md
```

---

## 🚀 Run It Locally

```bash
git clone https://github.com/KirtanPatel30/churn-revenue-analytics.git
cd churn-revenue-analytics

python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate

pip install -r requirements.txt

python run_pipeline.py         # cleans data, builds segments, trains model + SHAP
python src/app.py              # launches dashboard
```

Open **http://127.0.0.1:8050** — use the Segment / Contract Type filters to see every KPI and chart update live.

---

## ☁️ Deploy Your Own Copy (Render)

1. Fork/clone this repo
2. Render → **New → Blueprint** → connect the repo
3. Render reads `render.yaml`, builds the Docker image (which runs the pipeline at build time), and deploys automatically
4. First deploy takes ~3–5 minutes

---

## 🧠 Methodology Notes

**Why CLV-adapted RFM instead of textbook RFM?**
This dataset is an account snapshot, not a transaction log — there's no real "days since last purchase" field. Rather than fabricate one, tenure and monthly spend are used as documented, defensible proxies for Recency/Frequency, with real `TotalCharges` as Monetary value. Recognizing a data limitation and adapting the method honestly, instead of forcing a metric the data can't actually support, was a deliberate design choice.

**Why 0.844 AUC and not 0.95+?**
Churn is a genuinely hard, noisy prediction problem. A suspiciously perfect score would signal data leakage, not model quality — this result is realistic and consistent with published benchmarks on this dataset.

**Revenue-at-risk formula:**
`churn_probability × MonthlyCharges × 12`, summed per segment — a transparent, standard expected-value calculation rather than an opaque proprietary score.

---

## 📬 Contact

**Kirtan Patel**
MS Artificial Intelligence, San Jose State University
[LinkedIn](https://www.linkedin.com/in/kirtan-patel-24227a248/) · [GitHub](https://github.com/KirtanPatel30) · [Portfolio](https://kirtanpatel30.github.io/Portfolio/) · kirtannpatel2003@gmail.com

Open to Data Analyst, Data Engineer, Business Analyst, and Data Scientist internships — Summer 2027.
