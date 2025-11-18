"""
Machine Learning Model Training Script for Phishing Detection

This script trains a Random Forest classifier to detect phishing URLs based on
extracted features. Random Forest is chosen because it:
1. Handles mixed data types well (numerical features from URLs)
2. Provides feature importance rankings
3. Is resistant to overfitting
4. Performs well without extensive hyperparameter tuning
5. Provides probability estimates for confidence scores
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Tuple, Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))


class PhishingModelTrainer:
    """Train and evaluate machine learning models for phishing detection."""
    
    def __init__(self, data_path: str = "../data/phishing_dataset_processed.csv"):
        self.data_path = data_path
        self.model = None
        self.feature_names = None
        self.model_performance = {}
        
        print("🤖 Phishing Detection Model Training")
        print("==================================")
    
    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Load and prepare the training data.
        
        Returns:
            Tuple containing the full dataframe, features (X), and labels (y)
        """
        print(f"📊 Loading dataset from: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Please run the data collection script first."
            )
        
        # Load the dataset
        df = pd.read_csv(self.data_path)
        print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
        
        # Separate features and labels
        feature_columns = [col for col in df.columns 
                          if col not in ['label', 'source', 'original_url']]
        
        X = df[feature_columns].values
        y = df['label'].values
        
        self.feature_names = feature_columns
        
        print(f"📈 Features: {len(feature_columns)}")
        print(f"🎯 Labels: {len(np.unique(y))} classes (0=legitimate, 1=phishing)")
        print(f"   - Legitimate URLs: {sum(y == 0)}")
        print(f"   - Phishing URLs: {sum(y == 1)}")
        
        return df, X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Split data into training and testing sets.
        
        We use stratified splitting to ensure both training and test sets
        have the same proportion of phishing vs legitimate URLs.
        """
        print("\\n🔀 Splitting data into train/test sets...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2,      # 20% for testing
            random_state=42,    # For reproducible results
            stratify=y          # Maintain label distribution
        )
        
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        print(f"   Training phishing ratio: {sum(y_train)/len(y_train):.2%}")
        print(f"   Test phishing ratio: {sum(y_test)/len(y_test):.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """
        Train a Random Forest classifier.
        
        Random Forest works by:
        1. Creating many decision trees (forest)
        2. Each tree sees a random sample of data and features
        3. Final prediction is the majority vote of all trees
        4. This reduces overfitting and improves generalization
        """
        print("\\n🌲 Training Random Forest classifier...")
        
        # Initialize Random Forest with good default parameters
        rf_model = RandomForestClassifier(
            n_estimators=100,        # Number of trees in the forest
            max_depth=20,            # Maximum depth of each tree
            min_samples_split=5,     # Minimum samples required to split a node
            min_samples_leaf=2,      # Minimum samples required at a leaf node
            random_state=42,         # For reproducible results
            n_jobs=-1,              # Use all available CPU cores
            class_weight='balanced'  # Handle class imbalance automatically
        )
        
        # Train the model
        print("   🔄 Training in progress...")
        rf_model.fit(X_train, y_train)
        
        print("   ✅ Training completed!")
        print(f"   📊 Model contains {rf_model.n_estimators} decision trees")
        
        self.model = rf_model
        return rf_model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model's performance.
        
        We use multiple metrics because:
        - Accuracy: Overall correctness
        - Precision: Of predicted phishing, how many are actually phishing
        - Recall: Of actual phishing, how many we correctly identified
        - F1-score: Harmonic mean of precision and recall
        - ROC-AUC: Area under the curve (good for binary classification)
        """
        print("\\n📊 Evaluating model performance...")
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_random_forest first.")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]  # Probability of phishing
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store performance metrics
        self.model_performance = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Print results
        print("\\n🎯 Model Performance Results:")
        print("=" * 40)
        print(f"Accuracy:  {accuracy:.4f} ({accuracy:.1%})")
        print(f"Precision: {precision:.4f} ({precision:.1%})")
        print(f"Recall:    {recall:.4f} ({recall:.1%})")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        
        # Confusion Matrix
        print("\\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print("                 Predicted")
        print("               Legit  Phish")
        print(f"Actual Legit    {cm[0,0]:4d}   {cm[0,1]:4d}")
        print(f"       Phish    {cm[1,0]:4d}   {cm[1,1]:4d}")
        
        # Interpretation
        print("\\n💡 Performance Interpretation:")
        if accuracy > 0.95:
            print("   🟢 Excellent accuracy! Model performs very well.")
        elif accuracy > 0.90:
            print("   🟡 Good accuracy. Model performs well.")
        elif accuracy > 0.80:
            print("   🟠 Fair accuracy. Consider improving features or model.")
        else:
            print("   🔴 Low accuracy. Model needs significant improvement.")
        
        if precision > 0.90:
            print("   🟢 High precision: Few false phishing alerts.")
        elif precision > 0.80:
            print("   🟡 Good precision: Some false phishing alerts.")
        else:
            print("   🟠 Low precision: Many false phishing alerts.")
        
        if recall > 0.90:
            print("   🟢 High recall: Catches most phishing attempts.")
        elif recall > 0.80:
            print("   🟡 Good recall: Catches many phishing attempts.")
        else:
            print("   🟠 Low recall: Misses some phishing attempts.")
        
        return self.model_performance
    
    def analyze_feature_importance(self) -> None:
        """
        Analyze which features are most important for phishing detection.
        
        Random Forest provides feature importance scores that tell us which
        URL characteristics are most useful for distinguishing phishing from
        legitimate websites.
        """
        print("\\n🔍 Analyzing feature importance...")
        
        if self.model is None or self.feature_names is None:
            print("   ❌ Model or feature names not available.")
            return
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create a dataframe for easier sorting and display
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\\n🏆 Top 15 Most Important Features:")
        print("=" * 50)
        
        for idx, (_, row) in enumerate(feature_importance_df.head(15).iterrows(), 1):
            print(f"{idx:2d}. {row['feature']:25} {row['importance']:.4f}")
        
        # Explain what these features mean for phishing detection
        print("\\n💡 Feature Importance Insights:")
        
        top_features = feature_importance_df.head(5)['feature'].tolist()
        
        feature_explanations = {
            'url_length': "Longer URLs often indicate phishing attempts",
            'dots_count': "Multiple dots can indicate deceptive subdomains",
            'dashes_count': "Many dashes often used in phishing URLs",
            'is_https': "Lack of HTTPS can indicate suspicious sites",
            'brand_keywords_count': "Phishing sites often impersonate brands",
            'action_keywords_count': "Urgency words are common in phishing",
            'suspicious_tld': "Some TLDs are more commonly used for phishing",
            'is_ip_address': "Using IP instead of domain is suspicious",
            'subdomain_count': "Multiple subdomains can be deceptive"
        }
        
        for feature in top_features[:3]:
            if feature in feature_explanations:
                print(f"   • {feature}: {feature_explanations[feature]}")
    
    def perform_cross_validation(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Perform cross-validation to get a more robust estimate of model performance.
        
        Cross-validation works by:
        1. Splitting data into k folds (we use 5)
        2. Training on k-1 folds, testing on the remaining fold
        3. Repeating this k times
        4. Averaging the results
        
        This gives us a better estimate of how the model will perform on new data.
        """
        print("\\n🔄 Performing 5-fold cross-validation...")
        
        if self.model is None:
            print("   ❌ Model not trained yet.")
            return
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        cv_precision = cross_val_score(self.model, X, y, cv=5, scoring='precision')
        cv_recall = cross_val_score(self.model, X, y, cv=5, scoring='recall')
        cv_f1 = cross_val_score(self.model, X, y, cv=5, scoring='f1')
        
        print("\\n📊 Cross-Validation Results:")
        print("=" * 40)
        print(f"Accuracy:  {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        print(f"Precision: {cv_precision.mean():.4f} (±{cv_precision.std()*2:.4f})")
        print(f"Recall:    {cv_recall.mean():.4f} (±{cv_recall.std()*2:.4f})")
        print(f"F1-Score:  {cv_f1.mean():.4f} (±{cv_f1.std()*2:.4f})")
        
        print("\\n💡 Cross-validation shows how consistent the model is across different data splits.")
    
    def save_model(self, model_path: str = "../models/phishing_model.pkl") -> str:
        """Save the trained model to disk."""
        print(f"\\n💾 Saving model to: {model_path}")
        
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model with metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'performance': self.model_performance,
            'training_date': datetime.now().isoformat(),
            'model_type': 'RandomForestClassifier',
            'version': '1.0'
        }
        
        joblib.dump(model_data, model_path)
        print(f"✅ Model saved successfully!")
        print(f"📊 Model file size: {os.path.getsize(model_path) / 1024:.1f} KB")
        
        return os.path.abspath(model_path)
    
    def run_full_training_pipeline(self) -> str:
        """Run the complete model training pipeline."""
        print("🚀 Starting machine learning training pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Load data
            df, X, y = self.load_data()
            
            # Step 2: Split data
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            # Step 3: Train model
            self.train_random_forest(X_train, y_train)
            
            # Step 4: Evaluate model
            self.evaluate_model(X_test, y_test)
            
            # Step 5: Analyze feature importance
            self.analyze_feature_importance()
            
            # Step 6: Cross-validation
            self.perform_cross_validation(X, y)
            
            # Step 7: Save model
            model_path = self.save_model()
            
            print("\\n🎉 Model training completed successfully!")
            print("=" * 60)
            print("📋 Training Summary:")
            print(f"   📊 Dataset: {len(X)} samples")
            print(f"   🎯 Accuracy: {self.model_performance['accuracy']:.1%}")
            print(f"   🎯 F1-Score: {self.model_performance['f1_score']:.3f}")
            print(f"   💾 Model saved: {os.path.basename(model_path)}")
            
            print("\\n📋 Next Steps:")
            print("1. Test the model with the FastAPI service")
            print("2. Try predicting some test URLs")
            print("3. Deploy the ML API service")
            
            return model_path
            
        except Exception as e:
            print(f"❌ Error in training pipeline: {str(e)}")
            raise


def main():
    """Main function to run model training."""
    # Check if dataset exists
    data_path = "../data/phishing_dataset_processed.csv"
    if not os.path.exists(data_path):
        print("❌ Dataset not found!")
        print("Please run the data collection script first:")
        print("   python scripts/data_collection.py")
        return None
    
    # Initialize trainer
    trainer = PhishingModelTrainer(data_path)
    
    # Run training pipeline
    try:
        model_path = trainer.run_full_training_pipeline()
        return model_path
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return None


if __name__ == "__main__":
    main()