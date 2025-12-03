import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.tree import DecisionTreeClassifier
from Experiments.experiment_utils import load_and_split_dataset, evaluate_model


def main():
    data_path = "../data/final_dataset.csv"

    X_train, X_test, y_train, y_test, feature_names = load_and_split_dataset(
        data_path=data_path
    )

    dt = DecisionTreeClassifier(
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )

    results = evaluate_model("Decision Tree", dt, X_train, X_test, y_train, y_test)

    print("🎯 Final Decision Tree metrics:", results)


if __name__ == "__main__":
    main()
