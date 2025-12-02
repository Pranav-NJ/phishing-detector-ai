"""
Test script for phishing detection model
Tests with known phishing and legitimate URLs
"""
import os
import sys

# Add project root + utils/ to path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "utils"))

import joblib
import numpy as np
from enhanced_feature_extraction import CompleteFeatureExtractor

def load_model(model_path="../models/phishing_model.pkl"):

    """Load the trained model."""
    print(f"Loading model from: {model_path}")
    model_data = joblib.load(model_path)
    return model_data

def check_rule_based_phishing(features):
    """Check for rule-based phishing indicators."""
    # Rule 1: Very long numeric-only subdomain
    if features.get('subdomain_is_numeric_only', 0) > 0.5:
        subdomain_length = features.get('subdomain_length', 0)
        if subdomain_length > 15:
            return True, "Long numeric-only subdomain", 0.98
    
    # Rule 2: Extremely long subdomain with high numeric ratio
    subdomain_length = features.get('subdomain_length', 0)
    numeric_ratio = features.get('subdomain_numeric_ratio', 0)
    if subdomain_length > 20 and numeric_ratio > 0.7:
        return True, "Very long numeric subdomain", 0.96
    
    # Rule 3: Very long numeric subdomain feature
    if features.get('very_long_numeric_subdomain', 0) > 0.5:
        return True, "Very long numeric subdomain pattern", 0.95
    
    return False, None, 0.0

def predict_url(url, model_data):
    """Predict if URL is phishing."""
    model = model_data['model']
    feature_extractor = model_data['feature_extractor']
    feature_names = model_data['feature_names']
    threshold = model_data['optimal_threshold']
    
    # Extract features
    features = feature_extractor.extract_all_features(url)
    
    # Check rule-based first
    is_rule_phishing, rule_name, rule_confidence = check_rule_based_phishing(features)
    
    if is_rule_phishing:
        prediction = "PHISHING"
        confidence = rule_confidence
        source = f"RULE: {rule_name}"
    else:
        # Use ML model
        feature_vector = [features.get(name, 0.0) for name in feature_names]
        proba = model.predict_proba([feature_vector])[0]
        
        if proba[1] >= threshold:
            prediction = "PHISHING"
            confidence = proba[1]
        else:
            prediction = "LEGITIMATE"
            confidence = proba[0]
        
        source = f"ML Model (threshold: {threshold:.3f})"
    
    return prediction, confidence, source, features

def main():
    """Test the model with various URLs."""
    print("="*80)
    print("PHISHING DETECTION MODEL TESTER")
    print("="*80)
    
    # Load model
    try:
        model_data = load_model()
        print(f"✓ Model loaded: v{model_data['version']}")
        print(f"✓ Threshold: {model_data['optimal_threshold']:.3f}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test URLs - YOUR PHISHING EXAMPLES
    test_cases = [
        ("http://00000000883838383992929292222.ratingandreviews.in", "PHISHING"),
        ("http://00000000000000000000000000000000000000000.xyz", "PHISHING"),
        ("http://012345678901234567890.suspicious.com", "PHISHING"),
        ("https://www.google.com", "LEGITIMATE"),
        ("https://www.amazon.com", "LEGITIMATE"),
        ("https://github.com", "LEGITIMATE"),
        ("http://paypal.verify-account.com", "PHISHING"),
        ("http://025640258185444.litoralsulpremium.com.br", "PHISHING"),
    ]
    
    print("\n" + "="*80)
    print("TESTING URLS")
    print("="*80)
    
    correct = 0
    total = len(test_cases)
    
    for url, expected in test_cases:
        print(f"\n🔍 Testing: {url}")
        print(f"   Expected: {expected}")
        
        prediction, confidence, source, features = predict_url(url, model_data)
        
        print(f"   Predicted: {prediction} ({confidence:.3f})")
        print(f"   Source: {source}")
        
        # Show key features
        print(f"   Key Features:")
        print(f"      - Subdomain length: {features['subdomain_length']:.0f}")
        print(f"      - Numeric ratio: {features['subdomain_numeric_ratio']:.2f}")
        print(f"      - All numeric: {bool(features['subdomain_is_numeric_only'])}")
        print(f"      - Long numeric: {bool(features['long_numeric_subdomain'])}")
        print(f"      - Very long numeric: {bool(features['very_long_numeric_subdomain'])}")
        
        # Check if correct
        if prediction == expected:
            print("   ✅ CORRECT!")
            correct += 1
        else:
            print("   ❌ INCORRECT!")
    
    # Summary
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Correct: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"Incorrect: {total-correct}/{total}")
    
    if correct == total:
        print("\n🎉 PERFECT! All tests passed!")
    elif correct >= total * 0.9:
        print("\n✅ GOOD! Most tests passed.")
    else:
        print("\n⚠️  NEEDS IMPROVEMENT - Model missing phishing URLs")
        print("\nSuggestions:")
        print("1. Retrain with more phishing examples")
        print("2. Lower the detection threshold")
        print("3. Add more rule-based overrides")

if __name__ == "__main__":
    main()