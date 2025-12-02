"""
Simple script to check any URL for phishing
Usage: python scripts/predict_model.py <url>
"""

import os
import sys
import joblib

# ================================
# FIX: ADD PROJECT ROOT + utils/
# ================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))      # ml-api/
UTILS_DIR = os.path.join(BASE_DIR, "utils")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, UTILS_DIR)

from enhanced_feature_extraction import CompleteFeatureExtractor


# ============================================================
# RULE-BASED PHISHING CHECKS
# ============================================================

def check_rule_based_phishing(features):
    """Strong rule-based detection for numeric subdomains."""

    # Rule 1: Subdomain all numbers & long
    if features.get("subdomain_is_numeric_only", 0) > 0.5:
        if features.get("subdomain_length", 0) > 15:
            return True, "Long numeric-only subdomain", 0.98

    # Rule 2: High numeric ratio
    if features.get("subdomain_length", 0) > 20:
        if features.get("subdomain_numeric_ratio", 0) > 0.70:
            return True, "Very long numeric subdomain", 0.96

    # Rule 3: Generic numeric explosion
    if features.get("very_long_numeric_subdomain", 0) > 0.5:
        return True, "Excessive numeric subdomain", 0.95

    return False, None, 0.0


# ============================================================
# MAIN PREDICT FUNCTION
# ============================================================

def predict_url(url):
    """Predict if the given URL is phishing or legitimate."""

    # ==============================================
    # FIX: LOAD MODEL FROM ../models/phishing_model.pkl
    # ==============================================
    MODEL_PATH = os.path.join(BASE_DIR, "models", "phishing_model.pkl")

    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Error: Model file NOT found at:\n   {MODEL_PATH}")
        print("   Train your model first:")
        print("   python scripts/train_model.py\n")
        sys.exit(1)

    # Load model
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    feature_extractor = model_data["feature_extractor"]
    feature_names = model_data["feature_names"]
    threshold = model_data["optimal_threshold"]

    print(f"\n🔍 Analyzing URL: {url}")

    # Extract features
    features = feature_extractor.extract_all_features(url)

    # ===================================================
    # RULE-BASED CHECK FIRST
    # ===================================================
    is_rule, rule_name, rule_conf = check_rule_based_phishing(features)

    if is_rule:
        prediction = "PHISHING"
        confidence = rule_conf
        method = f"Rule: {rule_name}"

    else:
        # ML model prediction
        feature_vector = [features.get(name, 0.0) for name in feature_names]
        proba = model.predict_proba([feature_vector])[0]

        prob_legit = float(proba[0])
        prob_phish = float(proba[1])

        if prob_phish >= threshold:
            prediction = "PHISHING"
            confidence = prob_phish
        else:
            prediction = "LEGITIMATE"
            confidence = prob_legit

        method = f"ML Model (threshold {threshold:.3f})"

    # ===================================================
    # OUTPUT
    # ===================================================

    print("\n" + "=" * 60)
    if prediction == "PHISHING":
        print("🚨 PHISHING DETECTED!")
        print("=" * 60)
        print(f"⚠️  This URL is likely a phishing attack")
        print(f"Confidence: {confidence*100:.1f}%")
    else:
        print("✅ SAFE / LEGITIMATE")
        print("=" * 60)
        print(f"This URL appears safe")
        print(f"Confidence: {confidence*100:.1f}%")

    print(f"\nMethod Used: {method}")

    # ===================================================
    # Display key indicators
    # ===================================================
    print("\n📊 Key Indicators:")
    print(f"   Subdomain length: {features.get('subdomain_length', 0)}")
    print(f"   Numeric ratio: {features.get('subdomain_numeric_ratio', 0):.2f}")
    print(f"   All numeric: {bool(features.get('subdomain_is_numeric_only', 0))}")
    print(f"   Long numeric: {bool(features.get('long_numeric_subdomain', 0))}")
    print(f"   Very long numeric: {bool(features.get('very_long_numeric_subdomain', 0))}")
    print(f"   Suspicious TLD: {bool(features.get('suspicious_tld', 0))}")
    print(f"   Brand spoofing: {bool(features.get('brand_spoofing_pattern', 0))}")
    print(f"   HTTPS: {bool(features.get('is_https', 0))}")
    print(f"   Suspicious keywords: {int(features.get('suspicious_keyword_count', 0))}")

    # Warning summary
    warnings = []
    if features.get("brand_spoofing_pattern", 0): warnings.append("Brand spoofing detected")
    if features.get("suspicious_tld", 0): warnings.append("Suspicious TLD")
    if features.get("is_https", 0) == 0: warnings.append("Not using HTTPS")
    if features.get("subdomain_is_numeric_only", 0): warnings.append("Numeric-only subdomain")

    if warnings:
        print("\n⚠️ Warnings:")
        for w in warnings:
            print(f"   - {w}")

    print("\n" + "=" * 60)

    return prediction, confidence


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("\nUsage: python scripts/predict_model.py <url>\n")
        print("Example:")
        print("   python scripts/predict_model.py http://000000008....ratingandreviews.in\n")
        return

    url = sys.argv[1]
    predict_url(url)


if __name__ == "__main__":
    main()
