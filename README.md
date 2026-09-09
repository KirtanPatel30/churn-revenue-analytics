# Customer Churn & Revenue Risk Analytics

A live, interactive analytics platform that identifies which customers are
likely to churn, quantifies the annual revenue at risk in dollars, and
generates data-driven retention recommendations — built to answer the
question every subscription business actually asks: *"who's leaving, and
what will it cost us?"*

**Live dashboard:** _add your Render URL here after deploying_
**Repo:** _add your GitHub URL here_

---

## Situation

Companies typically discover churn after a customer has already left.
Retention teams need to know, before that happens: who is at risk, how much
revenue that represents, and which lever (contract offer, support outreach,
service fix) is most likely to work — not just a churn/no-churn label.

## Task

Build the full analyst workflow a retention/customer-analytics team would
actually use: segment customers by value and behavior, predict churn
probability, translate that probability into dollars, explain *why* the
model thinks someone will churn, and package it as something a
non-technical stakeholder can open and act on — not a notebook.

## Action

- **Data:** the canonical IBM Telco Customer Churn dataset (7,043 real
  customer accounts, Kaggle) — cleaned and feature-engineered
  (`src/preprocessing.py`).
- **Segmentation:** a CLV-adapted RFM model. True RFM needs a transaction
  log; this dataset is a single account snapshot, so tenure and monthly
  spend are used as honest Recency/Frequency proxies alongside real
  lifetime `TotalCharges` as Monetary value, quartile-scored into five
  named business segments — *Champions, Loyal High-Value, At Risk,
  Price-Sensitive/New, Low-Value/Disengaged* (`src/segmentation.py`).
- **Prediction:** an XGBoost classifier (ROC-AUC 0.844, F1 0.583 — a
  believable result, not an overfit one) wrapped in a scikit-learn
  pipeline with proper train/test isolation (`src/model.py`).
- **Explainability:** SHAP `TreeExplainer` values computed per feature so
  every prediction is auditable — contract type, tenure, and lack of
  online security/tech support surface as the top global drivers.
- **Revenue quantification:** `churn probability × annualized charges`
  per customer, rolled up by segment, so risk is expressed in dollars,
  not just probability.
- **Dashboard:** a four-tab Dash app (Overview, Churn Drivers, Deep Dive,
  Recommendations) with live segment/contract filters, KPI cards, and six
  chart types — pie, horizontal bar, box plot, scatter, histogram, and
  grouped bar — all built from scratch, not a templated Streamlit demo.
- **Deployment:** containerized with Docker, deployed to Render as a web
  service via gunicorn.

## Result

A self-contained, explainable churn analytics tool that quantifies
$389K+ in annual revenue concentrated in the highest-risk segments,
identifies month-to-month contracts as the single largest addressable
risk pool, and ships four specific, data-backed retention
recommendations — deployed live, filterable, and demo-able in an
interview.

---

## Project structure

```
churn-revenue-analytics/
├── data/
│   └── Telco-Customer-Churn.csv      # raw dataset (included)
├── src/
│   ├── preprocessing.py              # cleaning + feature engineering
│   ├── segmentation.py               # CLV-adapted RFM segmentation
│   ├── model.py                      # XGBoost + SHAP
│   └── app.py                        # Dash dashboard (entry point)
├── models/                           # generated: model.pkl, importances
├── assets/
│   └── style.css                     # dashboard styling (auto-loaded by Dash)
├── run_pipeline.py                   # runs preprocessing -> segmentation -> model
├── requirements.txt
├── Dockerfile
├── render.yaml
└── README.md
```

## How to run it locally (VS Code)

**1. Clone/open the folder in VS Code, then create a virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Run the data pipeline once** (cleans data, builds segments, trains the
model, computes SHAP values — takes under a minute):

```bash
python run_pipeline.py
```

You should see output ending in `Pipeline complete.` This generates:
- `data/processed_customers.csv`, `data/segmented_customers.csv`, `data/scored_customers.csv`
- `models/churn_model.pkl`, `models/feature_importance.csv`

**3. Launch the dashboard:**

```bash
python src/app.py
```

Open **http://127.0.0.1:8050** in your browser. Use the Segment / Contract
Type filters at the top — KPIs and every chart update live.

---

## How to deploy it to Render

**Option A — one-click via `render.yaml` (recommended):**

1. Push this folder to a new GitHub repo.
2. In Render: **New → Blueprint** → connect the repo. Render reads
   `render.yaml` automatically and builds the Docker image (which runs
   `run_pipeline.py` at build time, then serves via gunicorn).
3. First deploy takes ~3-5 minutes (installing xgboost/shap). After that,
   your dashboard is live at `https://<your-service-name>.onrender.com`.

**Option B — manual web service:**

1. In Render: **New → Web Service** → connect your repo.
2. Environment: **Docker**. Render will detect the `Dockerfile` automatically.
3. Leave build/start commands blank (the Dockerfile handles both).
4. Plan: **Free** is sufficient for a portfolio project.
5. Deploy. Note: free-tier services spin down after 15 minutes of
   inactivity and take 30-60 seconds to wake back up on the next visit —
   worth mentioning if you link this on your resume/portfolio so a
   recruiter isn't confused by the initial load time.

---

## Notes on methodology (worth mentioning in an interview)

- **Why CLV-adapted RFM instead of textbook RFM:** this dataset is an
  account snapshot, not a transaction log, so there's no real "days since
  last purchase" field to compute true Recency from. Rather than fabricate
  one, tenure and monthly spend are used as documented, defensible proxies
  — a decision worth explaining if asked, since recognizing a data
  limitation and adapting the method honestly is itself a signal of
  analytical maturity.
- **Why XGBoost's 0.844 AUC (not 0.95+):** churn is a genuinely hard,
  noisy prediction problem in this dataset. A suspiciously perfect score
  would indicate leakage, not skill — this result is realistic and
  matches published benchmarks on this same dataset.
- **Revenue-at-risk formula:** `churn_probability × MonthlyCharges × 12`,
  summed per segment. This is a standard, simple expected-value approach —
  intentionally transparent rather than a black-box number.
