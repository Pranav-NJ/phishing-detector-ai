"""Inspect model metadata for debugging parity between CLI and API models."""
import joblib
from pathlib import Path

def inspect(path):
    p = Path(path)
    if not p.exists():
        print(f"NOT FOUND: {path}")
        return
    data = joblib.load(str(p))
    model = data.get('model')
    classes = getattr(model, 'classes_', None)
    thresh = data.get('optimal_threshold')
    version = data.get('version')
    print(f"{p.name}: classes={classes}, optimal_threshold={thresh}, version={version}")

if __name__ == '__main__':
    inspect('c:/Users/Pranav N J/Desktop/PRANAV NJ/phishing-detector-ai/ml-api/models/phishing_model.pkl')
    inspect('c:/Users/Pranav N J/Desktop/PRANAV NJ/phishing-detector-ai/ml-api/models/phishing_model_old.pkl')
