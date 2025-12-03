import os
import sys
import pandas as pd

# --- Fix module path ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

# Import DatasetTransformer
from scripts.Transformer import DatasetTransformer

# === INPUT / OUTPUT FILES ===
input_csv = "../data/final_dataset0.csv"
output_csv = "../data/final_dataset2.csv"

print("📂 Loading input file:", input_csv)
df = pd.read_csv(input_csv)
print(f"✓ Loaded {len(df)} rows")

# === Setup Transformer ===
transformer = DatasetTransformer()

# Explicitly tell which columns to use
transformer.url_column = "URL"       # <-- your dataset URL column
transformer.label_column = "label"   # <-- your dataset label column

print("\n🔄 FORCE EXTRACTING FEATURES...")
# Extract features manually
extracted = transformer.extract_features_from_urls(df[transformer.url_column])

print("🔗 MERGING WITH LABELS...")
labels = transformer.normalize_labels(df, transformer.label_column)

# --- Build final output ---
output_df = extracted.copy()
output_df.insert(0, "original_url", df[transformer.url_column].values)
output_df["phishing"] = labels

print("\n💾 Saving transformed dataset to:", output_csv)
output_df.to_csv(output_csv, index=False)
print("✓ Saved successfully!")

print("\n🎉 DONE! Your dataset is ready for model training.")
