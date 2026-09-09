"""
model.py
--------
Trains a churn prediction model (XGBoost) on the segmented dataset,
evaluates it, and computes SHAP values for explainability.

Outputs:
    models/churn_model.pkl       - trained pipeline (preprocessing + model)
    models/feature_importance.csv - global SHAP importance per feature
    data/scored_customers.csv    - full dataset with churn_probability added

Run directly:
    python src/model.py
"""

import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from xgboost import XGBClassifier
import shap

IN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "segmented_customers.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.pkl")
IMPORTANCE_OUT = os.path.join(os.path.dirname(__file__), "..", "models", "feature_importance.csv")
SCORED_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "scored_customers.csv")

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizenLabel", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
TARGET = "ChurnFlag"


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def run():
    df = pd.read_csv(IN_PATH)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    print(f"Test ROC-AUC: {auc:.3f}")
    print(f"Test F1 Score: {f1:.3f}")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    # ---- SHAP explainability ----
    # Transform the full feature set through the fitted preprocessor so
    # SHAP sees the same numeric matrix XGBoost was trained on.
    preprocessor = pipeline.named_steps["preprocessor"]
    xgb_model = pipeline.named_steps["model"]

    X_test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_transformed)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    # Clean up one-hot feature names for readability in the dashboard
    importance_df["feature"] = (
        importance_df["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    importance_df.to_csv(IMPORTANCE_OUT, index=False)
    print(f"\nTop 10 churn drivers (mean |SHAP|):")
    print(importance_df.head(10).to_string(index=False))

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nSaved model -> {MODEL_OUT}")

    # ---- Score the FULL dataset (train+test) for the dashboard ----
    full_proba = pipeline.predict_proba(X)[:, 1]
    df["ChurnProbability"] = full_proba
    df["RevenueAtRisk"] = df["ChurnProbability"] * df["MonthlyCharges"] * 12  # annualized

    df.to_csv(SCORED_OUT, index=False)
    print(f"Saved scored dataset -> {SCORED_OUT}")

    return pipeline, importance_df, df


if __name__ == "__main__":
    run()
