import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# =========================================================
# URL DEDUPLICATION (same as train_model logic)
# =========================================================
def remove_url_duplicates(
    df: pd.DataFrame,
    url_col: str = "original_url",
    label_col: str = "phishing",
) -> pd.DataFrame:
    """
    Remove exact duplicate URLs (same URL + same label).
    This avoids train/test leakage without killing the minority class.
    """
    if url_col not in df.columns:
        print(f"⚠️  '{url_col}' column not found, skipping URL deduplication")
        return df

    before = len(df)

    phishing_dups = df[df[label_col] == 1].duplicated(subset=[url_col]).sum()
    legit_dups = df[df[label_col] == 0].duplicated(subset=[url_col]).sum()

    df = df.drop_duplicates(subset=[url_col, label_col], keep="first")

    after = len(df)
    removed = before - after

    print("🔍 Removing exact URL duplicates (URL + label)...")
    print(f"   Phishing duplicates removed: {phishing_dups}")
    print(f"   Legitimate duplicates removed: {legit_dups}")
    print(f"   Total removed: {removed} ({(removed / before * 100):.2f}%)")
    print(f"   Remaining: {after:,} rows\n")

    return df


# =========================================================
# DATA LOADING + SPLITTING (Option 1 behaviour)
# =========================================================
def load_and_split_dataset(
    data_path: str = "../data/final_dataset.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load dataset, remove duplicates, drop biased features, split into train/test
    with URL-overlap validation.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    print("=" * 70)
    print("📊 LOADING DATASET FOR EXPERIMENTS")
    print("=" * 70)
    print(f"📂 Path: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"❌ Dataset not found: {data_path}\n"
            f"Make sure final_dataset.csv exists in ../data/"
        )

    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df):,} rows, {df.shape[1]} columns")

    # 1) Remove exact URL duplicates
    df = remove_url_duplicates(df, url_col="original_url", label_col="phishing")

    # 2) Choose label column
    if "phishing" in df.columns:
        y = df["phishing"].values
        label_col = "phishing"
    elif "label" in df.columns:
        y = df["label"].values
        label_col = "label"
    else:
        raise ValueError("❌ Dataset must contain 'phishing' or 'label' column")

    # 3) Remove biased & metadata features
    exclude = [label_col, "original_url", "source"]
    biased_features = ["is_https", "is_http"]

    feature_columns = [c for c in df.columns if c not in exclude]
    feature_columns = [c for c in feature_columns if c not in biased_features]

    print(f"🔧 Features selected (excluding biased + metadata): {len(feature_columns)}")

    X = df[feature_columns].values
    urls = df["original_url"].values if "original_url" in df.columns else np.array(
        [str(i) for i in range(len(df))]
    )

    # 4) Fix NaN / Inf
    if np.isnan(X).any():
        print("⚠️  Replacing NaN values with 0...")
        X = np.nan_to_num(X, nan=0.0)

    if np.isinf(X).any():
        print("⚠️  Replacing infinite values...")
        X = np.nan_to_num(X, posinf=1e10, neginf=-1e10)

    # 5) Class distribution
    legit = int((y == 0).sum())
    phish = int((y == 1).sum())
    total = len(y)

    print("\n📊 Class distribution:")
    print(f"   Legitimate: {legit:,} ({legit / total * 100:.1f}%)")
    print(f"   Phishing:   {phish:,} ({phish / total * 100:.1f}%)")

    if legit == 0 or phish == 0:
        raise ValueError("❌ Need BOTH phishing and legitimate samples in dataset.")

    # 6) Stratified split
    print("\n🔀 Stratified train/test split (80/20)...")
    X_train, X_test, y_train, y_test, urls_train, urls_test = train_test_split(
        X,
        y,
        urls,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
        shuffle=True,
    )

    print(f"✓ Training set: {X_train.shape[0]:,} samples")
    print(f"   - Legitimate: {(y_train == 0).sum():,}")
    print(f"   - Phishing:   {(y_train == 1).sum():,}")
    print(f"✓ Test set: {X_test.shape[0]:,} samples")
    print(f"   - Legitimate: {(y_test == 0).sum():,}")
    print(f"   - Phishing:   {(y_test == 1).sum():,}")

    # 7) URL overlap check
    print("\n🔍 Validating train/test independence...")
    overlap = set(urls_train).intersection(set(urls_test))
    if len(overlap) > 0:
        print(f"\n❌ CRITICAL: {len(overlap)} URLs appear in BOTH train and test!")
        for url in list(overlap)[:3]:
            print(f"   - {url}")
        raise ValueError("Train/test overlap detected – this causes data leakage!")
    else:
        print("✓ No overlapping URLs between train and test.")
        print("✓ Split is SAFE for experiments.\n")

    return X_train, X_test, y_train, y_test, feature_columns


# =========================================================
# GENERIC EVALUATION
# =========================================================
def evaluate_model(
    name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Fit model, evaluate on test set, print metrics, and return them as dict.
    """
    print("=" * 70)
    print(f"🚀 Running {name} experiment...")
    print("=" * 70)

    # Fit
    model.fit(X_train, y_train)

    # Predict labels
    y_pred = model.predict(X_test)

    # Predict scores / probabilities (for ROC AUC)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if y_score is not None:
        roc = roc_auc_score(y_test, y_score)
    else:
        roc = float("nan")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n===== {name.upper()} RESULTS =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    if not np.isnan(roc):
        print(f"ROC AUC  : {roc:.4f}")

    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("               Legit  Phish")
    print(f"Actual Legit   {tn:5d}  {fp:5d}")
    print(f"       Phish   {fn:5d}  {tp:5d}")
    print()

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
