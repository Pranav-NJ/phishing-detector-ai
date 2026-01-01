"""
Dataset Augmentation Script - Fix HTTPS Bias

PROBLEM: 100% of legitimate URLs have HTTPS, only 49.5% of phishing URLs do.
This creates a bias where the model learns "HTTPS = safe" which is WRONG!

SOLUTION: Augment the dataset by creating both HTTP and HTTPS versions of all URLs.
This removes the protocol bias and forces the model to focus on actual phishing indicators.
"""

import pandas as pd
import os
import sys

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from utils.enhanced_feature_extraction import CompleteFeatureExtractor

def augment_dataset(input_path, output_path):
    """
    Create augmented dataset with both HTTP and HTTPS versions of each URL.
    This removes protocol bias from the training data.
    """
    print("=" * 70)
    print("🔧 DATASET AUGMENTATION - Removing HTTPS Bias")
    print("=" * 70)
    
    # Load original dataset
    print(f"\n📂 Loading dataset: {input_path}")
    df = pd.read_csv(input_path)
    initial_count = len(df)
    print(f"✓ Loaded {initial_count:,} URLs")
    
    # Check current HTTPS distribution
    print("\n📊 Current HTTPS Distribution (BEFORE label inversion):")
    legit_orig = df[df['phishing'] == 0]
    phish_orig = df[df['phishing'] == 1]
    print(f"   Label=0 with HTTPS: {sum(legit_orig['is_https'] == 1.0)} / {len(legit_orig)} ({sum(legit_orig['is_https'] == 1.0)/len(legit_orig)*100:.1f}%)")
    print(f"   Label=1 with HTTPS: {sum(phish_orig['is_https'] == 1.0)} / {len(phish_orig)} ({sum(phish_orig['is_https'] == 1.0)/len(phish_orig)*100:.1f}%)")
    
    # Extract URLs and labels
    urls = df['original_url'].values
    labels = df['phishing'].values
    
    # Create feature extractor
    extractor = CompleteFeatureExtractor()
    
    # Create augmented dataset
    print("\n🔄 Creating augmented dataset...")
    print("   For each URL, generating both HTTP and HTTPS versions...")
    
    augmented_data = []
    
    for idx, (url, label) in enumerate(zip(urls, labels)):
        if (idx + 1) % 5000 == 0:
            print(f"   Processed {idx + 1:,} / {initial_count:,} URLs...")
        
        # Extract base URL (remove protocol)
        if url.startswith('https://'):
            base_url = url[8:]
        elif url.startswith('http://'):
            base_url = url[7:]
        else:
            base_url = url
        
        # Create HTTP version
        http_url = 'http://' + base_url
        http_features = extractor.extract_all_features(http_url)
        http_features['original_url'] = http_url
        http_features['phishing'] = label
        augmented_data.append(http_features)
        
        # Create HTTPS version
        https_url = 'https://' + base_url
        https_features = extractor.extract_all_features(https_url)
        https_features['original_url'] = https_url
        https_features['phishing'] = label
        augmented_data.append(https_features)
    
    print(f"✓ Created {len(augmented_data):,} augmented samples")
    
    # Create augmented dataframe
    df_augmented = pd.DataFrame(augmented_data)
    
    # Remove duplicates (same URL + label)
    print("\n🔍 Removing duplicate URLs...")
    before = len(df_augmented)
    df_augmented = df_augmented.drop_duplicates(subset=['original_url', 'phishing'], keep='first')
    after = len(df_augmented)
    print(f"✓ Removed {before - after:,} duplicates")
    print(f"✓ Final dataset: {after:,} samples")
    
    # Check new HTTPS distribution
    print("\n📊 New HTTPS Distribution (balanced):")
    legit_new = df_augmented[df_augmented['phishing'] == 0]
    phish_new = df_augmented[df_augmented['phishing'] == 1]
    legit_https = sum(legit_new['is_https'] == 1.0)
    phish_https = sum(phish_new['is_https'] == 1.0)
    print(f"   Label=0 with HTTPS: {legit_https} / {len(legit_new)} ({legit_https/len(legit_new)*100:.1f}%)")
    print(f"   Label=1 with HTTPS: {phish_https} / {len(phish_new)} ({phish_https/len(phish_new)*100:.1f}%)")
    print(f"   ✓ Difference: {abs(legit_https/len(legit_new) - phish_https/len(phish_new))*100:.1f}%")
    
    # Save augmented dataset
    print(f"\n💾 Saving augmented dataset: {output_path}")
    df_augmented.to_csv(output_path, index=False)
    print(f"✓ Saved successfully!")
    
    print("\n" + "=" * 70)
    print("✅ DATASET AUGMENTATION COMPLETE!")
    print("=" * 70)
    print(f"📊 Original: {initial_count:,} samples")
    print(f"📊 Augmented: {after:,} samples ({after/initial_count:.1f}x larger)")
    print(f"📊 HTTPS bias: REMOVED (now ~50% each)")
    print("=" * 70)
    
    return df_augmented


if __name__ == "__main__":
    # Paths
    input_path = os.path.join(BASE_DIR, "..", "data", "final_dataset.csv")
    output_path = os.path.join(BASE_DIR, "..", "data", "final_dataset_augmented.csv")
    
    # Run augmentation
    augment_dataset(input_path, output_path)
    
    print("\n📋 NEXT STEPS:")
    print("   1. Review the augmented dataset")
    print("   2. Train model with augmented data:")
    print(f"      python scripts/train_model.py --dataset {output_path}")
    print("   3. Test predictions with HTTP and HTTPS versions")
