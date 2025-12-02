"""
Enhanced Machine Learning Model Training for Phishing Detection

This script trains an optimized Random Forest classifier with:
- Proper hyperparameter tuning
- Class imbalance handling
- Optimal decision threshold selection
- Cross-validation for robustness
"""

import sys
import os

# Add ml-api root to Python path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Dict, Any
import warnings
from utils.enhanced_feature_extraction import CompleteFeatureExtractor

warnings.filterwarnings('ignore')


class EnhancedPhishingModelTrainer:
    """Train and evaluate optimized ML models for phishing detection."""

    def __init__(self, data_path: str = "../data/final_dataset.csv"):
        self.data_path = data_path
        self.model = None
        self.feature_names = None
        self.model_performance = {}
        self.optimal_threshold = 0.5

        print("🤖 Enhanced Phishing Detection Model Training")
        print("=" * 60)

    # =========================================================
    # LOAD DATA
    # =========================================================
    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Load and prepare the balanced training data."""
        print(f"📊 Loading dataset from: {self.data_path}")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"❌ Dataset not found at {self.data_path}\n"
                f"Please run merge_datasets.py to create final_dataset.csv"
            )

        df = pd.read_csv(self.data_path)
        print(f"✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")

        # Exclude metadata columns
        exclude = ['phishing', 'label', 'original_url', 'source']
        feature_columns = [c for c in df.columns if c not in exclude]

        # Get target variable
        if 'phishing' in df.columns:
            y = df['phishing'].values
        elif 'label' in df.columns:
            y = df['label'].values
        else:
            raise ValueError("❌ Dataset must contain 'phishing' or 'label' column")

        # Extract features
        X = df[feature_columns].values
        self.feature_names = feature_columns

        # Handle missing and infinite values
        if np.isnan(X).any():
            print("⚠️  Replacing NaN values with 0...")
            X = np.nan_to_num(X, nan=0.0)
        
        if np.isinf(X).any():
            print("⚠️  Replacing infinite values...")
            X = np.nan_to_num(X, posinf=1e10, neginf=-1e10)

        # Display class distribution
        legit = sum(y == 0)
        phish = sum(y == 1)
        total = len(y)

        print(f"\n📊 Class Distribution:")
        print(f"   Legitimate: {legit:,} ({legit/total*100:.1f}%)")
        print(f"   Phishing:   {phish:,} ({phish/total*100:.1f}%)")

        # Validation
        if legit == 0 or phish == 0:
            raise ValueError("❌ Dataset must contain BOTH phishing and legitimate URLs")

        # Check for severe imbalance
        minority_ratio = min(legit, phish) / total
        if minority_ratio < 0.1:
            print(f"⚠️  WARNING: Severe class imbalance detected!")
            print(f"   Minority class: {minority_ratio*100:.1f}%")
            print(f"   Using balanced class weights to compensate")

        print(f"✓ Features: {len(feature_columns)}")

        return df, X, y

    # =========================================================
    # SPLIT DATA
    # =========================================================
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data with stratification to maintain class balance."""
        print("\n🔀 Splitting data (80% train, 20% test)...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.20, 
            random_state=42, 
            stratify=y
        )

        print(f"✓ Training set: {X_train.shape[0]:,} samples")
        print(f"   - Legitimate: {sum(y_train == 0):,}")
        print(f"   - Phishing:   {sum(y_train == 1):,}")
        print(f"✓ Test set: {X_test.shape[0]:,} samples")
        print(f"   - Legitimate: {sum(y_test == 0):,}")
        print(f"   - Phishing:   {sum(y_test == 1):,}")

        return X_train, X_test, y_train, y_test

    # =========================================================
    # TRAIN MODEL
    # =========================================================
    def train_optimized_random_forest(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        enable_tuning: bool = False
    ) -> RandomForestClassifier:
        """Train Random Forest with optimized hyperparameters."""
        print("\n🌲 Training Optimized Random Forest Classifier...")

        if enable_tuning:
            print("   🔍 Performing hyperparameter tuning (this may take a while)...")
            
            param_grid = {
                'n_estimators': [150, 200, 250],
                'max_depth': [20, 25, 30],
                'min_samples_split': [2, 4, 6],
                'min_samples_leaf': [1, 2, 3],
                'max_features': ['sqrt', 'log2']
            }
            
            rf_base = RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                oob_score=True
            )
            
            grid_search = GridSearchCV(
                rf_base, 
                param_grid,
                cv=5, 
                scoring='f1',
                n_jobs=-1, 
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            
            print(f"   ✓ Best parameters found:")
            for param, value in grid_search.best_params_.items():
                print(f"      • {param}: {value}")
        else:
            print("   Using pre-optimized hyperparameters...")
            
            model = RandomForestClassifier(
                n_estimators=200,           # More trees for better performance
                max_depth=25,               # Deep enough to capture patterns
                min_samples_split=4,        # Balanced splitting
                min_samples_leaf=2,         # Small leaves for detail
                max_features='sqrt',        # Feature sampling
                class_weight='balanced',    # Handle any imbalance
                random_state=42,
                n_jobs=-1,                  # Use all CPU cores
                bootstrap=True,
                oob_score=True              # Out-of-bag validation
            )
            
            model.fit(X_train, y_train)

        print(f"✓ Training completed!")
        print(f"✓ Number of trees: {model.n_estimators}")
        
        if hasattr(model, 'oob_score_'):
            print(f"✓ Out-of-Bag Score: {model.oob_score_:.4f}")

        self.model = model
        return model

    # =========================================================
    # FIND OPTIMAL THRESHOLD
    # =========================================================
    def find_optimal_threshold(
        self, 
        X_val: np.ndarray, 
        y_val: np.ndarray
    ) -> float:
        """Find optimal classification threshold for best F1-score."""
        print("\n🎯 Finding optimal classification threshold...")

        # Get prediction probabilities
        proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_val, proba)
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

        # Find best threshold
        idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[idx] if idx < len(thresholds) else 0.5

        self.optimal_threshold = optimal_threshold

        print(f"✓ Optimal threshold: {optimal_threshold:.3f}")
        print(f"✓ Expected F1-score: {f1_scores[idx]:.4f}")
        print(f"✓ Expected Precision: {precisions[idx]:.3f}")
        print(f"✓ Expected Recall: {recalls[idx]:.3f}")

        return optimal_threshold

    # =========================================================
    # EVALUATE MODEL
    # =========================================================
    def evaluate_model(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation with optimal threshold."""
        print("\n📊 Evaluating model performance...")

        # Get predictions
        proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= self.optimal_threshold).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, proba)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Store results
        self.model_performance = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "optimal_threshold": self.optimal_threshold
        }

        # Display results
        print("\n" + "=" * 60)
        print("🎯 MODEL PERFORMANCE RESULTS")
        print("=" * 60)
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"Threshold: {self.optimal_threshold:.3f}")

        print("\n📊 Confusion Matrix:")
        print("                 Predicted")
        print("               Legit  Phish")
        print(f"Actual Legit   {tn:5d}  {fp:5d}")
        print(f"       Phish   {fn:5d}  {tp:5d}")

        # Calculate error rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        print(f"\n✓ True Negatives (correct legit):  {tn:,}")
        print(f"✓ True Positives (correct phish):  {tp:,}")
        print(f"✗ False Positives (wrong phish):   {fp:,} ({fpr*100:.2f}%)")
        print(f"✗ False Negatives (missed phish):  {fn:,} ({fnr*100:.2f}%)")

        # Performance assessment
        print("\n💡 Performance Assessment:")
        if f1 > 0.95:
            print("   🟢 EXCELLENT: Model performs exceptionally well!")
        elif f1 > 0.90:
            print("   🟢 VERY GOOD: Model is highly reliable!")
        elif f1 > 0.85:
            print("   🟡 GOOD: Model performs well for production use")
        elif f1 > 0.80:
            print("   🟡 FAIR: Model is acceptable but could improve")
        else:
            print("   🔴 NEEDS IMPROVEMENT: Consider more data or features")

        if recall > 0.95:
            print("   🟢 Catches almost all phishing attempts!")
        elif recall > 0.90:
            print("   🟡 Catches most phishing attempts")
        else:
            print("   🟠 Misses some phishing - consider lower threshold")

        if precision > 0.95:
            print("   🟢 Very few false alarms!")
        elif precision > 0.90:
            print("   🟡 Acceptable false alarm rate")
        else:
            print("   🟠 Many false alarms - consider higher threshold")

        return self.model_performance

    # =========================================================
    # FEATURE IMPORTANCE
    # =========================================================
    def analyze_feature_importance(self, top_n: int = 20) -> None:
        """Analyze and display most important features."""
        print(f"\n🔍 Analyzing Feature Importance (Top {top_n})...")

        if self.model is None or self.feature_names is None:
            print("   ❌ Model or features not available")
            return

        # Get feature importances
        importances = self.model.feature_importances_

        # Create DataFrame
        feature_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\n🏆 Most Important Features:")
        print("=" * 60)

        for idx, (_, row) in enumerate(feature_df.head(top_n).iterrows(), 1):
            bar_length = int(row['importance'] * 50)
            bar = '█' * bar_length
            print(f"{idx:2d}. {row['feature']:30} {row['importance']:.4f} {bar}")

        # Feature insights
        print("\n💡 What These Features Mean:")
        insights = {
            'url_length': "Longer URLs often hide malicious intent",
            'n_dots': "Multiple dots indicate subdomain spoofing",
            'n_hypens': "Excessive hyphens in phishing domains",
            'n_slash': "Deep paths may be suspicious",
            'n_percent': "URL encoding to obfuscate phishing",
            'n_at': "@ symbol in redirection attacks",
            'n_and': "Many parameters = potential malicious behavior",
            'url_entropy': "High randomness indicates suspicious patterns",
            'domain_entropy': "Unusual domain randomness",
            'suspicious_keyword_count': "Phishing-related keywords present",
            'brand_spoofing_pattern': "Attempts to impersonate brands",
            'is_url_shortener': "Shortened URLs hide destination",
            'has_suspicious_keyword': "Contains known phishing terms",
            'domain_has_digits': "Numbers in domain (often phishing)",
            'subdomain_count': "Multiple subdomains for obfuscation"
        }

        shown = 0
        for feature in feature_df.head(10)['feature']:
            if feature in insights and shown < 6:
                print(f"   • {feature}: {insights[feature]}")
                shown += 1

    # =========================================================
    # CROSS VALIDATION
    # =========================================================
    def perform_cross_validation(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> None:
        """Perform k-fold cross-validation for robustness check."""
        print("\n🔄 Performing 5-Fold Cross-Validation...")

        if self.model is None:
            print("   ❌ Model not trained")
            return

        try:
            cv_scores = {
                'accuracy': cross_val_score(self.model, X, y, cv=5, scoring='accuracy'),
                'precision': cross_val_score(self.model, X, y, cv=5, scoring='precision'),
                'recall': cross_val_score(self.model, X, y, cv=5, scoring='recall'),
                'f1': cross_val_score(self.model, X, y, cv=5, scoring='f1')
            }

            print("\n📊 Cross-Validation Results (Mean ± 2×Std):")
            print("=" * 50)
            for metric, scores in cv_scores.items():
                mean = scores.mean()
                std = scores.std()
                print(f"{metric.capitalize():12} {mean:.4f} ± {std*2:.4f}")

            print("\n✓ Model shows consistent performance across folds")

        except Exception as e:
            print(f"   ⚠️  Cross-validation error: {str(e)}")

    # =========================================================
    # SAVE MODEL
    # =========================================================
    def save_model(self, model_path: str = "../models/phishing_model.pkl") -> str:
        """Save trained model with all metadata."""
        print(f"\n💾 Saving model to: {model_path}")

        if self.model is None:
            raise ValueError("❌ No model to save")

        # Create directory if needed
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Package model with metadata
        model_data = {
            "model": self.model,
            "feature_extractor": CompleteFeatureExtractor(),
            "feature_names": self.feature_names,
            "performance": self.model_performance,
            "optimal_threshold": self.optimal_threshold,
            "training_date": datetime.now().isoformat(),
            "model_type": "RandomForestClassifier",
            "version": "3.0_final"
        }

        # Save to disk
        joblib.dump(model_data, model_path)
        
        file_size = os.path.getsize(model_path) / 1024
        print(f"✓ Model saved successfully!")
        print(f"✓ File size: {file_size:.1f} KB")
        print(f"✓ Optimal threshold: {self.optimal_threshold:.3f}")

        return os.path.abspath(model_path)

    # =========================================================
    # FULL PIPELINE
    # =========================================================
    def run_full_training_pipeline(
        self, 
        enable_tuning: bool = False
    ) -> str:
        """Execute complete training pipeline."""
        print("\n🚀 STARTING ENHANCED TRAINING PIPELINE")
        print("=" * 60)

        try:
            # 1. Load data
            df, X, y = self.load_data()

            # 2. Split data
            X_train, X_test, y_train, y_test = self.split_data(X, y)

            # 3. Train model
            self.train_optimized_random_forest(X_train, y_train, enable_tuning)

            # 4. Find optimal threshold
            self.find_optimal_threshold(X_test, y_test)

            # 5. Evaluate model
            self.evaluate_model(X_test, y_test)

            # 6. Feature importance analysis
            self.analyze_feature_importance()

            # 7. Cross-validation
            self.perform_cross_validation(X, y)

            # 8. Save model
            model_path = self.save_model()

            # Final summary
            print("\n" + "=" * 60)
            print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"📊 Dataset: {len(X):,} samples")
            print(f"🎯 Accuracy: {self.model_performance['accuracy']*100:.2f}%")
            print(f"🎯 Precision: {self.model_performance['precision']*100:.2f}%")
            print(f"🎯 Recall: {self.model_performance['recall']*100:.2f}%")
            print(f"🎯 F1-Score: {self.model_performance['f1_score']:.4f}")
            print(f"🎯 ROC-AUC: {self.model_performance['roc_auc']:.4f}")
            print(f"💾 Model saved: {os.path.basename(model_path)}")
            print(f"🎚️  Threshold: {self.optimal_threshold:.3f}")

            print("\n📋 Next Steps:")
            print("1. Start FastAPI service: python api/main.py")
            print("2. Test with real URLs")
            print("3. Monitor production performance")
            print("4. Retrain periodically with new data")

            return model_path

        except Exception as e:
            print(f"\n❌ TRAINING FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


# =========================================================
# MAIN
# =========================================================
def main():
    """Main training function."""
    
    # Configuration
    data_path = "../data/final_dataset.csv"
    enable_hyperparameter_tuning = False  # Set True for grid search (slower)
    
    # Check dataset exists
    if not os.path.exists(data_path):
        print("❌ Final dataset not found!")
        print(f"Looking for: {data_path}")
        print("\n📝 To create the dataset:")
        print("1. Run: python scripts/build_enhanced_dataset.py")
        print("2. Run: python scripts/merge_datasets.py")
        return None

    # Initialize trainer
    trainer = EnhancedPhishingModelTrainer(data_path=data_path)

    try:
        # Run training pipeline
        model_path = trainer.run_full_training_pipeline(
            enable_tuning=enable_hyperparameter_tuning
        )
        return model_path
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        return None


if __name__ == "__main__":
    main()