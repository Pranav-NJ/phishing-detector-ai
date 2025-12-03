"""
Universal Dataset Transformer for Phishing Detection
Converts any dataset with URLs into the enhanced feature format.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
import os
import sys
from pathlib import Path

# -------------------------------------------------------------
# FIX PYTHON PATH FOR FOLDER NAME WITH HYPHEN (ml-api)
# -------------------------------------------------------------

CURRENT_DIR = os.path.dirname(__file__)             # ml-api/scripts
ML_API_ROOT = os.path.dirname(CURRENT_DIR)          # ml-api/
PROJECT_ROOT = os.path.dirname(ML_API_ROOT)         # project root

# Add ml-api/ and project root so "utils" can be imported
sys.path.insert(0, ML_API_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# Import the feature extractor
from utils.enhanced_feature_extraction import CompleteFeatureExtractor


class DatasetTransformer:
    """Transform any phishing dataset to use enhanced features."""
    
    def __init__(self, feature_extractor: Optional[CompleteFeatureExtractor] = None):
        self.extractor = feature_extractor or CompleteFeatureExtractor()
        self.expected_features = self.extractor.get_feature_names()
        self.url_column = None
        self.label_column = None
        
    # -------------------------------------------------------------
    # DETECT URL + LABEL COLUMNS
    # -------------------------------------------------------------
    def detect_columns(self, df: pd.DataFrame) -> Tuple[str, str]:
        print("🔍 Detecting URL and label columns...")

        url_candidates = ['url', 'URL', 'original_url', 'link', 'domain', 'website', 'address']
        url_col = None

        for col in df.columns:
            if col in url_candidates:
                url_col = col
                break
            if df[col].dtype == object:
                sample = str(df[col].dropna().iloc[0]).lower() if len(df[col].dropna()) else ""
                if "http" in sample or "." in sample:
                    url_col = col
                    break
        
        if url_col is None:
            raise ValueError("❌ Could not detect URL column. Rename it to 'url' or 'original_url'.")
        
        # Detect label column
        label_candidates = ['label', 'phishing', 'is_phishing', 'class', 'target', 'y']
        label_col = None

        for col in df.columns:
            if col in label_candidates:
                label_col = col
                break
        
        if label_col is None:
            # Try binary columns
            for col in df.columns:
                if df[col].nunique() == 2:
                    label_col = col
                    break

        if label_col is None:
            raise ValueError("❌ Could not detect label column. Rename it to 'phishing' or 'label'.")

        print(f"✓ URL column detected: {url_col}")
        print(f"✓ Label column detected: {label_col}")

        self.url_column = url_col
        self.label_column = label_col
        return url_col, label_col

    # -------------------------------------------------------------
    # NORMALIZE LABELS
    # -------------------------------------------------------------
    def normalize_labels(self, df: pd.DataFrame, label_col: str) -> pd.Series:
        print(f"\n🔄 Normalizing labels from '{label_col}'...")

        labels = df[label_col].astype(str).str.lower().str.strip()

        mapping = {
            "0": 0, "1": 1,
            "legitimate": 0, "benign": 0,
            "phishing": 1, "malicious": 1,
            "true": 1, "false": 0,
            "yes": 1, "no": 0
        }

        labels = labels.map(mapping)

        if labels.isna().any():
            raise ValueError(f"❌ Unmapped label values found: {df[label_col].unique()}")

        labels = labels.astype(int)

        print("✓ Labels normalized to {0,1}")
        return labels

    # -------------------------------------------------------------
    # EXTRACT FEATURES FROM URLS
    # -------------------------------------------------------------
    def extract_features_from_urls(self, urls: pd.Series) -> pd.DataFrame:
        print(f"\n⚙️ Extracting features from {len(urls)} URLs...")

        features = []
        for i, url in enumerate(urls):
            try:
                feats = self.extractor.extract_all_features(url)
            except:
                feats = {f: 0.0 for f in self.expected_features}
            features.append(feats)

        print("✓ Feature extraction complete")
        return pd.DataFrame(features)

    # -------------------------------------------------------------
    # MERGE EXISTING + EXTRACTED FEATURES
    # -------------------------------------------------------------
    def merge_with_existing_features(self, df, extracted_features):
        exclude = [self.url_column, self.label_column, 'source', 'Unnamed: 0']

        existing = [c for c in df.columns if c not in exclude]
        status = {}
        final = {}

        for feat in self.expected_features:
            if feat in existing:
                final[feat] = df[feat].values
                status[feat] = "existing"
            else:
                final[feat] = extracted_features[feat].values
                status[feat] = "extracted"

        merged = pd.DataFrame(final)
        merged.insert(0, "original_url", df[self.url_column].values)
        merged["phishing"] = self.normalize_labels(df, self.label_column)

        print(f"✓ Merged features: {len(merged.columns)} columns")

        return merged, status

    # -------------------------------------------------------------
    # MAIN TRANSFORM FUNCTION
    # -------------------------------------------------------------
    def transform_dataset(self, input_df: pd.DataFrame) -> pd.DataFrame:
        print("\n===============================")
        print("🔄 TRANSFORMING DATASET")
        print("===============================")

        url_col, label_col = self.detect_columns(input_df)

        extracted = self.extract_features_from_urls(input_df[url_col])

        merged, _ = self.merge_with_existing_features(input_df, extracted)

        merged = merged.fillna(0)
        merged = merged.replace([np.inf, -np.inf], 0)

        print("\n✓ Dataset Transformation Complete")
        print(f"  Rows: {len(merged)}")
        print(f"  Columns: {len(merged.columns)}")

        return merged

    # -------------------------------------------------------------
    # TRANSFORM FILE
    # -------------------------------------------------------------
    def transform_file(self, input_path: str, output_path: str):
        print(f"\n📂 Loading input file: {input_path}")

        df = pd.read_csv(input_path)

        transformed = self.transform_dataset(df)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        transformed.to_csv(output_path, index=False)

        print(f"💾 Saved transformed dataset → {output_path}")
        return output_path


# =============================================================
# CLI SUPPORT (optional)
# =============================================================

def main():
    parser = argparse.ArgumentParser(description="Transform dataset to enhanced features")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    transformer = DatasetTransformer()
    transformer.transform_file(args.input, args.output)


if __name__ == "__main__":
    main()
