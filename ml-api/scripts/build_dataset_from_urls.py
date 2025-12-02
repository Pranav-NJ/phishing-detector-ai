import sys
import os

# Add ml-api root to path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from utils.enhanced_feature_extraction import CompleteFeatureExtractor

import pandas as pd
from tqdm import tqdm
import time



# ==============================
# CONFIG
# ==============================

INPUT = "../data/raw_urls.txt"            # Your big txt file
OUTPUT = "../data/phishing_enhanced.csv"  # Final CSV
TEMP_SAVE_EVERY = 1000                      # Auto-save every N URLs


def main() -> None:
    """Build enhanced phishing dataset from raw_urls.txt using CompleteFeatureExtractor."""

    # ==============================
    # CHECK FILE EXISTS
    # ==============================

    if not os.path.exists(INPUT):
        print(f"❌ ERROR: raw_urls.txt not found at: {INPUT}")
        return

    # ==============================
    # LOAD URLs
    # ==============================

    print("\n📂 Loading URLs from raw_urls.txt ...")

    with open(INPUT, "r", encoding="utf-8", errors="ignore") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"✓ Loaded {len(urls)} URLs\n")

    # ==============================
    # INIT EXTRACTOR
    # ==============================

    extractor = CompleteFeatureExtractor()
    rows = []

    # ==============================
    # EXTRACTION LOOP
    # ==============================

    print("🚀 Extracting enhanced features for each URL...\n")
    start_time = time.time()

    for i, url in enumerate(tqdm(urls, total=len(urls))):
        try:
            feats = extractor.extract_all_features(url)
            feats["phishing"] = 1  # All provided URLs = phishing
            feats["original_url"] = url
            rows.append(feats)
        except Exception as e:
            print(f"\n❌ Error extracting: {url}")
            print("Reason:", e)

        # Auto-save periodically to avoid losing progress
        if rows and (i + 1) % TEMP_SAVE_EVERY == 0:
            temp_df = pd.DataFrame(rows)
            temp_df.to_csv(OUTPUT, index=False)
            print(f"💾 Auto-saved {len(rows)} records so far to {OUTPUT} ...")

    # ==============================
    # FINAL SAVE
    # ==============================

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT, index=False)

    elapsed = time.time() - start_time

    # ==============================
    # DONE
    # ==============================

    print("\n🎉 DONE!")
    print(f"Enhanced phishing dataset saved to: {OUTPUT}")
    print(f"Total URLs processed: {len(df)}")
    print(f"Total features per URL: {df.shape[1]}")
    print(f"⏱ Total time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()

