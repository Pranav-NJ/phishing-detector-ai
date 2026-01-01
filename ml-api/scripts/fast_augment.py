"""
Fast Dataset Augmentation - Remove HTTPS Bias

Creates HTTP and HTTPS versions by duplicating rows and flipping is_https/is_http features.
Much faster than re-extracting all features.
"""

import pandas as pd
import os

def fast_augment_dataset(input_path, output_path):
    """
    Fast augmentation: duplicate each row twice (HTTP and HTTPS versions).
    Flip is_https and is_http values appropriately.
    """
    print("=" * 70)
    print("🚀 FAST DATASET AUGMENTATION")
    print("=" * 70)
    
    # Load dataset
    print(f"\n📂 Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df):,} rows")
    
    # Check if features exist
    if 'is_https' not in df.columns or 'is_http' not in df.columns:
        print("❌ Dataset missing is_https or is_http columns!")
        return
    
    # Create HTTP versions (set is_https=0, is_http=1)
    df_http = df.copy()
    df_http['is_https'] = 0.0
    df_http['is_http'] = 1.0
    df_http['original_url'] = df_http['original_url'].str.replace('https://', 'http://', regex=False)
    
    # Create HTTPS versions (set is_https=1, is_http=0)
    df_https = df.copy()
    df_https['is_https'] = 1.0
    df_https['is_http'] = 0.0
    df_https['original_url'] = df_https['original_url'].str.replace('http://', 'https://', regex=False)
    
    # Combine
    df_augmented = pd.concat([df_http, df_https], ignore_index=True)
    
    # Remove exact duplicates
    print("\n🔍 Removing duplicates...")
    before = len(df_augmented)
    df_augmented = df_augmented.drop_duplicates(subset=['original_url', 'phishing'], keep='first')
    after = len(df_augmented)
    print(f"✓ Removed {before - after:,} duplicates")
    
    # Check HTTPS distribution (BEFORE label inversion)
    print("\n📊 HTTPS Distribution (before label inversion):")
    label0 = df_augmented[df_augmented['phishing'] == 0]
    label1 = df_augmented[df_augmented['phishing'] == 1]
    label0_https = sum(label0['is_https'] == 1.0)
    label1_https = sum(label1['is_https'] == 1.0)
    print(f"   Label=0: {label0_https}/{len(label0)} HTTPS ({label0_https/len(label0)*100:.1f}%)")
    print(f"   Label=1: {label1_https}/{len(label1)} HTTPS ({label1_https/len(label1)*100:.1f}%)")
    
    # Save
    print(f"\n💾 Saving: {output_path}")
    df_augmented.to_csv(output_path, index=False)
    print(f"✓ Saved {len(df_augmented):,} rows ({len(df_augmented)/len(df)*100:.0f}% of original)")
    
    print("\n" + "=" * 70)
    print("✅ AUGMENTATION COMPLETE!")
    print(f"   Original: {len(df):,}")
    print(f"   Augmented: {len(df_augmented):,}")
    print(f"   HTTPS balanced: ~50% each class")
    print("=" * 70)
    
    return df_augmented


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    input_path = os.path.join(base_dir, "..", "data", "final_dataset.csv")
    output_path = os.path.join(base_dir, "..", "data", "final_dataset_balanced.csv")
    
    fast_augment_dataset(input_path, output_path)
    
    print("\n📋 NEXT STEPS:")
    print("   python scripts/train_model.py --dataset ../data/final_dataset_balanced.csv")
