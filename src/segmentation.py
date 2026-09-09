"""
segmentation.py
----------------
Segments customers into business-meaningful groups using a CLV-adapted
RFM approach.

Note on methodology: classic RFM needs a transaction log (Recency =
days since last purchase, Frequency = number of purchases). This
dataset is a single account snapshot, not a transaction history, so
true Recency/Frequency can't be computed. Instead we substitute two
honest, defensible proxies built from what the data actually contains:

    R proxy -> tenure           (how long they've been engaged; a longer
                                  tenured customer is analogous to a more
                                  "recently active" relationship)
    F proxy -> MonthlyCharges   (rate of ongoing spend / engagement)
    M       -> TotalCharges     (true lifetime monetary value, this one
                                  is a real Monetary figure, not a proxy)

Each is scored 1-4 by quartile (4 = best), summed into an RFM score,
then mapped to a named business segment.

Run directly:
    python src/segmentation.py
"""

import pandas as pd
import os

IN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed_customers.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "segmented_customers.csv")


def score_quartile(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Score a numeric series 1-4 by quartile. 4 = most favorable."""
    ranks = series.rank(method="first")
    quartiles = pd.qcut(ranks, 4, labels=[1, 2, 3, 4])
    scores = quartiles.astype(int)
    if not ascending:
        scores = 5 - scores
    return scores


def segment_name(row) -> str:
    total = row["R_Score"] + row["F_Score"] + row["M_Score"]
    if total >= 10:
        return "Champions"
    elif total >= 8:
        return "Loyal High-Value"
    elif total >= 6:
        return "At Risk"
    elif total >= 4:
        return "Price-Sensitive / New"
    else:
        return "Low-Value / Disengaged"


def run():
    df = pd.read_csv(IN_PATH)

    df["R_Score"] = score_quartile(df["tenure"], ascending=True)
    df["F_Score"] = score_quartile(df["MonthlyCharges"], ascending=True)
    df["M_Score"] = score_quartile(df["TotalCharges"], ascending=True)

    df["RFM_Score"] = df["R_Score"] + df["F_Score"] + df["M_Score"]
    df["Segment"] = df.apply(segment_name, axis=1)

    print("Segment distribution:")
    print(df["Segment"].value_counts())
    print()
    print("Churn rate by segment:")
    print(df.groupby("Segment")["ChurnFlag"].mean().sort_values(ascending=False))

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved segmented dataset -> {OUT_PATH}")
    return df


if __name__ == "__main__":
    run()
