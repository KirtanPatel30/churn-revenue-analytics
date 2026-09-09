"""
run_pipeline.py
----------------
Runs the full data pipeline end to end:
    1. preprocessing.py  -> cleans raw data
    2. segmentation.py   -> CLV-adapted RFM segmentation
    3. model.py           -> trains XGBoost churn model + SHAP importance

Run this once before starting the dashboard:
    python run_pipeline.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import preprocessing
import segmentation
import model


def main():
    print("=" * 60)
    print("STEP 1/3: Preprocessing")
    print("=" * 60)
    preprocessing.run()

    print()
    print("=" * 60)
    print("STEP 2/3: Segmentation")
    print("=" * 60)
    segmentation.run()

    print()
    print("=" * 60)
    print("STEP 3/3: Model training + SHAP")
    print("=" * 60)
    model.run()

    print()
    print("=" * 60)
    print("Pipeline complete. Run 'python src/app.py' to launch the dashboard.")
    print("=" * 60)


if __name__ == "__main__":
    main()
