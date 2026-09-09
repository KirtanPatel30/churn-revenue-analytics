"""
preprocessing.py
-----------------
Loads the raw Telco Customer Churn dataset, cleans it, and engineers
base features used downstream by segmentation.py and model.py.

Run directly:
    python src/preprocessing.py
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Telco-Customer-Churn.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed_customers.csv")


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges is stored as a string and has blank values for customers
    # with 0 tenure (brand-new customers who have not been billed yet).
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # For 0-tenure customers, TotalCharges should reasonably equal
    # their first MonthlyCharges (they've been billed once, or not yet).
    # We fill with 0 rather than dropping rows, since dropping would
    # silently remove every brand-new customer from the analysis.
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Binary target: 1 = churned, 0 = retained
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

    # SeniorCitizen is already 0/1 in the raw data but stored as int;
    # keep as-is for modeling, add a readable label for charts.
    df["SeniorCitizenLabel"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Average monthly spend across the customer's lifetime — useful as a
    # sanity check against MonthlyCharges and for CLV estimation.
    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    # Simple projected Customer Lifetime Value: current monthly charge
    # extrapolated over an assumed 24-month forward horizon. This is a
    # standard, transparent CLV proxy used when true multi-year billing
    # history isn't available.
    FORWARD_HORIZON_MONTHS = 24
    df["ProjectedCLV"] = df["MonthlyCharges"] * FORWARD_HORIZON_MONTHS

    # Tenure buckets for readable grouping in charts
    bins = [-1, 6, 12, 24, 48, 100]
    labels = ["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49+ mo"]
    df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    return df


def run():
    df = load_raw()
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    df_clean = clean(df)
    print(f"Missing TotalCharges filled: {(df['TotalCharges'].astype(str).str.strip() == '').sum()} rows")
    print(f"Churn rate: {df_clean['ChurnFlag'].mean():.2%}")

    df_clean.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned dataset -> {OUT_PATH}")
    return df_clean


if __name__ == "__main__":
    run()
