import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Experiments.experiment_utils import load_and_split_dataset, evaluate_model


def main():
    data_path = "../data/final_dataset.csv"

    X_train, X_test, y_train, y_test, feature_names = load_and_split_dataset(
        data_path=data_path
    )

    log_reg = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    n_jobs=1,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    results = evaluate_model(
        "Logistic Regression", log_reg, X_train, X_test, y_train, y_test
    )

    print("🎯 Final Logistic Regression metrics:", results)


if __name__ == "__main__":
    main()
