"""
Leak-Proof Model Comparison for Phishing Detection
Ensures:
 - Same preprocessing as train_model.py
 - Removes biased features (is_https, is_http)
 - Removes URL columns
 - No train/test leakage
 - Same split used for all models
 - Realistic accuracy (no 95–99% cheating)
"""

import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# 💠 STEP 1 — LOAD & CLEAN DATA EXACTLY LIKE train_model.py
# ============================================================

def load_clean_data(path="../data/final_dataset.csv"):
    df = pd.read_csv(path)

    print(f"\nLoaded dataset: {len(df)} rows")

    # Remove biased leakage features
    for biased in ["is_https", "is_http"]:
        if biased in df.columns:
            df.drop(columns=[biased], inplace=True)

    # Remove URL columns (leakage)
    for col in ["original_url", "URL", "FILENAME", "file", "source"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Ensure label is named 'phishing'
    if "phishing" not in df.columns:
        if "label" in df.columns:
            df.rename(columns={"label": "phishing"}, inplace=True)
        else:
            raise ValueError("Dataset must contain 'phishing' column")

    # Extract features and label
    X = df.drop(columns=["phishing"]).values
    y = df["phishing"].values

    # Handle NaN & Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    print("Class distribution:")
    print(" Legit:", sum(y == 0))
    print(" Phish:", sum(y == 1))

    return X, y, df.columns.drop("phishing")


# ============================================================
# 💠 STEP 2 — SPLIT DATA SAFELY (NO LEAKAGE)
# ============================================================

def safe_split(X, y):
    return train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )


# ============================================================
# 💠 STEP 3 — MODEL WRAPPER
# ============================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
    else:
        auc = 0.0

    return {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC-AUC": auc
    }


# ============================================================
# 💠 STEP 4 — RUN COMPARISON
# ============================================================

def run_comparison():
    X, y, feature_names = load_clean_data()
    X_train, X_test, y_train, y_test = safe_split(X, y)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),

        "Logistic Regression": LogisticRegression(max_iter=500),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,  # prevents overfitting
            random_state=42
        ),

        "XGBoost": xgb.XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False
        )
    }

    results = {}

    print("\n=======================================================")
    print("📊 RUNNING LEAK-PROOF MODEL COMPARISON")
    print("=======================================================\n")

    for name, model in models.items():
        print(f"Training {name}...")
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)

    print("\n=======================================================")
    print("📊 MODEL COMPARISON RESULTS (NO LEAKAGE)")
    print("=======================================================\n")

    print(f"{'Model':22}  Acc     Prec    Rec     F1      AUC")
    print("-" * 65)

    for name, m in results.items():
        print(f"{name:22} {m['Accuracy']:.4f}  {m['Precision']:.4f}  {m['Recall']:.4f}  {m['F1']:.4f}  {m['ROC-AUC']:.4f}")

    return results


if __name__ == "__main__":
    run_comparison()
