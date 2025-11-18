"""
Data Collection and Preprocessing Script for Phishing Detection

This script downloads phishing URL datasets from various sources and preprocesses
them for machine learning training. It extracts features from URLs and creates
a clean dataset ready for model training.
"""

import pandas as pd
import numpy as np
import requests
import os
import sys
from urllib.parse import urlparse
from typing import List, Dict, Tuple
import time
import random

# Add the utils directory to the path to import feature extractor
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from feature_extractor import URLFeatureExtractor


class PhishingDataCollector:
    """Collect and preprocess phishing detection datasets."""
    
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = data_dir
        self.feature_extractor = URLFeatureExtractor()
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        print(f"📁 Data directory: {os.path.abspath(data_dir)}")
    
    def download_phishtank_data(self) -> pd.DataFrame:
        """
        Download and process PhishTank verified phishing URLs.
        
        PhishTank is a collaborative clearing house for data about phishing
        on the Internet. It provides real-world phishing URLs that are verified
        by their community.
        
        Note: This function creates sample data as PhishTank requires registration
        for API access. In production, you would use their official API.
        """
        print("🎣 Collecting PhishTank phishing data...")
        
        # Sample phishing URLs (in production, use PhishTank API)
        sample_phishing_urls = [
            "http://paypal-verification-required.tk/login.php",
            "https://amazon-security-check.ml/account/verify",
            "http://microsoft-account-suspended.ga/signin",
            "https://apple-id-locked.cf/unlock",
            "http://facebook-security-alert.tk/login",
            "https://google-account-recovery.ml/verify",
            "http://ebay-payment-issue.ga/resolve",
            "https://netflix-billing-problem.cf/update",
            "http://dropbox-storage-full.tk/upgrade",
            "https://linkedin-message-waiting.ml/check",
            "http://192.168.1.100/paypal/login.php",
            "https://secure-banking-login.tk/authenticate",
            "http://update-your-account-now.ml/verify",
            "https://click-here-to-verify.ga/action",
            "http://urgent-security-notice.cf/respond",
            "https://congratulations-winner.tk/claim",
            "http://limited-time-offer.ml/free",
            "https://immediate-action-required.ga/update",
            "http://suspended-account-restore.cf/login",
            "https://expired-password-reset.tk/change"
        ]
        
        phishing_data = []
        for url in sample_phishing_urls:
            phishing_data.append({
                'url': url,
                'label': 1,  # 1 for phishing
                'source': 'phishtank_sample'
            })
        
        print(f"✅ Collected {len(phishing_data)} phishing URLs from PhishTank sample")
        return pd.DataFrame(phishing_data)
    
    def download_legitimate_data(self) -> pd.DataFrame:
        """
        Create a dataset of legitimate URLs.
        
        These are well-known, trusted websites that are definitely not phishing.
        In production, you might use Alexa Top Sites or similar sources.
        """
        print("✅ Collecting legitimate website data...")
        
        # Sample legitimate URLs from trusted sources
        legitimate_urls = [
            "https://chatgpt.com"
            "https://chat.openai.com",
            "https://www.google.com",
            "https://www.facebook.com",
            "https://www.amazon.com",
            "https://www.microsoft.com",
            "https://www.apple.com",
            "https://www.paypal.com",
            "https://www.ebay.com",
            "https://www.netflix.com",
            "https://www.linkedin.com",
            "https://www.twitter.com",
            "https://www.instagram.com",
            "https://www.youtube.com",
            "https://www.wikipedia.org",
            "https://www.github.com",
            "https://www.stackoverflow.com",
            "https://www.reddit.com",
            "https://www.cnn.com",
            "https://www.bbc.com",
            "https://www.nytimes.com",
            "https://www.washingtonpost.com",
            "https://www.gmail.com",
            "https://www.outlook.com",
            "https://www.dropbox.com",
            "https://www.adobe.com",
            "https://www.salesforce.com",
            "https://www.zoom.us",
            "https://www.slack.com",
            "https://www.trello.com",
            "https://www.atlassian.com",
            "https://www.spotify.com",
            "https://docs.google.com/forms",
            "https://drive.google.com/file",
            "https://www.office.com/login",
            "https://account.microsoft.com/profile",
            "https://myaccount.google.com/security",
            "https://www.paypal.com/signin",
            "https://secure.amazon.com/account",
            "https://appleid.apple.com/account",
            "https://www.facebook.com/login",
            "https://secure.netflix.com/account"
        ]
        
        legitimate_data = []
        for url in legitimate_urls:
            legitimate_data.append({
                'url': url,
                'label': 0,  # 0 for legitimate
                'source': 'trusted_sites'
            })
        
        print(f"✅ Collected {len(legitimate_data)} legitimate URLs")
        return pd.DataFrame(legitimate_data)
    
    def extract_features_from_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from all URLs in the dataset.
        
        This is the most important step - converting raw URLs into numerical
        features that machine learning algorithms can understand.
        """
        print("🔧 Extracting features from URLs...")
        
        features_list = []
        failed_extractions = 0
        
        for idx, row in df.iterrows():
            try:
                url = row['url']
                features = self.feature_extractor.extract_all_features(url)
                
                # Add the label and source info
                features['label'] = row['label']
                features['source'] = row['source']
                features['original_url'] = url
                
                features_list.append(features)
                
                # Progress indicator
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{len(df)} URLs...")
                    
            except Exception as e:
                print(f"  ❌ Failed to extract features for {row['url']}: {str(e)}")
                failed_extractions += 1
                continue
        
        if failed_extractions > 0:
            print(f"⚠️  Failed to extract features from {failed_extractions} URLs")
        
        features_df = pd.DataFrame(features_list)
        print(f"✅ Successfully extracted features from {len(features_df)} URLs")
        print(f"📊 Feature dimensions: {features_df.shape}")
        
        return features_df
    
    def analyze_features(self, df: pd.DataFrame) -> None:
        """Analyze the extracted features to understand the data better."""
        print("\n📊 Dataset Analysis:")
        print("=" * 50)
        
        # Basic statistics
        print(f"Total samples: {len(df)}")
        print(f"Phishing URLs: {sum(df['label'] == 1)}")
        print(f"Legitimate URLs: {sum(df['label'] == 0)}")
        print(f"Features extracted: {len(df.columns) - 3}")  # Exclude label, source, original_url
        
        # Feature statistics for phishing vs legitimate
        feature_columns = [col for col in df.columns if col not in ['label', 'source', 'original_url']]
        
        print("\\n🔍 Key Feature Differences (Phishing vs Legitimate):")
        print("-" * 50)
        
        phishing_df = df[df['label'] == 1]
        legitimate_df = df[df['label'] == 0]
        
        interesting_features = [
            'url_length', 'dots_count', 'dashes_count', 'is_https',
            'is_ip_address', 'suspicious_tld', 'brand_keywords_count',
            'action_keywords_count', 'urgency_keywords_count'
        ]
        
        for feature in interesting_features:
            if feature in df.columns:
                phishing_mean = phishing_df[feature].mean()
                legitimate_mean = legitimate_df[feature].mean()
                
                print(f"{feature:25} | Phishing: {phishing_mean:8.2f} | Legitimate: {legitimate_mean:8.2f}")
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "phishing_dataset.csv") -> str:
        """Save the processed dataset to CSV file."""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"💾 Dataset saved to: {os.path.abspath(filepath)}")
        return filepath
    
    def create_balanced_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a balanced dataset with equal numbers of phishing and legitimate URLs.
        
        Machine learning models perform better with balanced datasets.
        """
        print("⚖️  Creating balanced dataset...")
        
        phishing_df = df[df['label'] == 1]
        legitimate_df = df[df['label'] == 0]
        
        # Use the smaller group size to balance
        min_size = min(len(phishing_df), len(legitimate_df))
        
        balanced_phishing = phishing_df.sample(n=min_size, random_state=42)
        balanced_legitimate = legitimate_df.sample(n=min_size, random_state=42)
        
        balanced_df = pd.concat([balanced_phishing, balanced_legitimate], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        print(f"✅ Balanced dataset created: {len(balanced_df)} samples ({min_size} phishing, {min_size} legitimate)")
        return balanced_df
    
    def run_full_pipeline(self) -> str:
        """Run the complete data collection and preprocessing pipeline."""
        print("🚀 Starting phishing detection data collection pipeline")
        print("=" * 60)
        
        # Step 1: Download datasets
        phishing_df = self.download_phishtank_data()
        legitimate_df = self.download_legitimate_data()
        
        # Step 2: Combine datasets
        print("\\n🔗 Combining datasets...")
        combined_df = pd.concat([phishing_df, legitimate_df], ignore_index=True)
        print(f"Combined dataset size: {len(combined_df)} URLs")
        
        # Step 3: Extract features
        features_df = self.extract_features_from_dataset(combined_df)
        
        # Step 4: Create balanced dataset
        balanced_df = self.create_balanced_dataset(features_df)
        
        # Step 5: Analyze features
        self.analyze_features(balanced_df)
        
        # Step 6: Save dataset
        output_file = self.save_dataset(balanced_df, "phishing_dataset_processed.csv")
        
        print("\\n🎉 Data collection and preprocessing completed successfully!")
        print(f"📄 Final dataset: {output_file}")
        print(f"📊 Ready for machine learning training!")
        
        return output_file


def main():
    """Main function to run data collection."""
    print("🛡️  Phishing Detection Dataset Creation")
    print("====================================")
    
    # Initialize collector
    collector = PhishingDataCollector()
    
    # Run full pipeline
    try:
        output_file = collector.run_full_pipeline()
        
        print("\\n📋 Next Steps:")
        print("1. Review the generated dataset")
        print("2. Run the model training script")
        print("3. Test the trained model with the ML API")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error in data collection pipeline: {str(e)}")
        return None


if __name__ == "__main__":
    main()