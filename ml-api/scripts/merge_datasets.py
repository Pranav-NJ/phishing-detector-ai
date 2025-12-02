import os
import sys
import pandas as pd
from tqdm import tqdm

# ---------------------------------------
# FIX IMPORT PATH (same fix as build script)
# ---------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from utils.enhanced_feature_extraction import CompleteFeatureExtractor


# ============================
# CONFIG
# ============================

PHISHING_DATA = "../data/phishing_enhanced.csv"
SAFE_URLS_FILE = "../data/safe_urls.txt"
OUTPUT = "../data/final_dataset.csv"

SAVE_EVERY = 500   # Auto save progress for big files


def load_safe_urls():
    """Load safe URLs from safe_urls.txt"""
    if not os.path.exists(SAFE_URLS_FILE):
        raise FileNotFoundError(f"safe_urls.txt NOT FOUND at {SAFE_URLS_FILE}")

    with open(SAFE_URLS_FILE, "r", encoding="utf-8", errors="ignore") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"✓ Loaded {len(urls)} legitimate URLs from safe_urls.txt")
    return urls


def extract_features_for_legit(urls):
    """Extract features for legitimate URLs"""
    extractor = CompleteFeatureExtractor()
    rows = []

    print("\n🚀 Extracting features for legitimate (safe) URLs...\n")

    for i, url in enumerate(tqdm(urls)):
        try:
            feats = extractor.extract_all_features(url)
            feats["phishing"] = 0  # Label = legitimate
            feats["original_url"] = url
            rows.append(feats)
        except Exception as e:
            print(f"\n❌ Error processing URL: {url}")
            print("Reason:", e)

        # Auto save progress if needed
        if (i + 1) % SAVE_EVERY == 0:
            temp_df = pd.DataFrame(rows)
            temp_df.to_csv("../data/legit_partial.csv", index=False)
            print(f"💾 Auto-saved {len(rows)} legitimate records so far...")

    return pd.DataFrame(rows)


def merge_all(phishing_df, legit_df):
    """Merge phishing + legitimate datasets"""
    print("\n🔀 Merging phishing + legitimate datasets...")

    full_df = pd.concat([phishing_df, legit_df], ignore_index=True)

    # Shuffle rows
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"✓ Total rows after merge: {full_df.shape[0]}")
    print(f"✓ Total features: {full_df.shape[1]}")

    return full_df


def main():
    print("\n📘 STARTING MERGE PROCESS")
    print("============================\n")

    # 1) Load phishing dataset
    if not os.path.exists(PHISHING_DATA):
        print(f"❌ ERROR: phishing_enhanced.csv not found at {PHISHING_DATA}")
        return

    phishing_df = pd.read_csv(PHISHING_DATA)
    print(f"✓ Loaded phishing dataset: {phishing_df.shape[0]} rows")

    # 2) Load safe URLs
    safe_urls = load_safe_urls()

    # 3) Extract features for legitimate URLs
    legit_df = extract_features_for_legit(safe_urls)
    print(f"\n✓ Legitimate dataset processed: {legit_df.shape[0]} rows\n")

    # 4) Merge both datasets
    final_df = merge_all(phishing_df, legit_df)

    # 5) Save final dataset
    final_df.to_csv(OUTPUT, index=False)

    print("\n🎉 DONE!")
    print(f"Final merged dataset saved to: {OUTPUT}")
    print(f"Total rows: {final_df.shape[0]}")
    print(f"Total features: {final_df.shape[1]}")


if __name__ == "__main__":
    main()
