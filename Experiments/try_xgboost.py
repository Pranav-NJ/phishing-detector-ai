import os
import sys

# Ensure project root is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from Experiments.experiment_utils import load_and_split_dataset, evaluate_model


def main():
    # Attempt to import XGBoost
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        print("❌ XGBoost is not installed or is incompatible.")
        print("   Install using:")
        print("   pip install xgboost")
        print("Error:", str(e))
        return

    data_path = "../data/final_dataset.csv"

    # Load dataset and get safe split
    X_train, X_test, y_train, y_test, feature_names = load_and_split_dataset(
        data_path=data_path
    )

    # ---------------------------------------------
    # ⭐ SAFE, STABLE XGBOOST CONFIGURATION
    # ---------------------------------------------
    xgb = XGBClassifier(
        n_estimators=300,          # More trees = better smoothing
        max_depth=6,               # Prevent overfitting
        learning_rate=0.08,        # Conservative LR
        subsample=0.9,             # Good generalization
        colsample_bytree=0.8,      # Feature sampling
        eval_metric="logloss",     # Required after XGBoost 1.6+
        scale_pos_weight=float((y_train == 0).sum()) / float((y_train == 1).sum()),
        random_state=42,
        n_jobs=1,                  # Prevent recursion depth issues
        tree_method="hist"         # Fast + prevents memory issues
    )

    # Run experiment
    results = evaluate_model("XGBoost", xgb, X_train, X_test, y_train, y_test)

    print("🎯 Final XGBoost Metrics:", results)


if __name__ == "__main__":
    main()
