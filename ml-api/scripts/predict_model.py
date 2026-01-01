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
    
    # Create FRESH feature extractor (model doesn't store it)
    feature_extractor = CompleteFeatureExtractor()
    
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
    # Analyze URL and extract detailed indicators
    # ===================================================
    import urllib.parse
    import tldextract
    
    parsed = urllib.parse.urlparse(url)
    extracted = tldextract.extract(url)
    
    # Find which suspicious keywords were detected
    suspicious_keywords_found = []
    url_lower = url.lower()
    for keyword in feature_extractor.suspicious_keywords:
        if keyword in url_lower:
            suspicious_keywords_found.append(keyword)
    
    # Find which brand keywords were detected
    brand_keywords_found = []
    for keyword in feature_extractor.brand_keywords:
        if keyword in url_lower:
            brand_keywords_found.append(keyword)
    
    # ===================================================
    # Display detailed analysis
    # ===================================================
    print("\n📊 DETAILED ANALYSIS:")
    print("=" * 60)
    
    # URL Structure
    print("\n🔗 URL Structure:")
    print(f"   Protocol: {parsed.scheme}")
    print(f"   Domain: {extracted.domain}.{extracted.suffix}")
    if extracted.subdomain:
        print(f"   Subdomain: {extracted.subdomain}")
    if parsed.path and parsed.path != '/':
        print(f"   Path: {parsed.path}")
    if parsed.query:
        print(f"   Query: {parsed.query[:50]}...")
    
    # Domain Analysis
    print(f"\n🌐 Domain Analysis:")
    print(f"   Domain length: {int(features.get('domain_length', 0))} characters")
    print(f"   Has digits in domain: {'Yes' if features.get('domain_has_digits', 0) else 'No'}")
    if extracted.subdomain:
        print(f"   Subdomain length: {int(features.get('subdomain_length', 0))} characters")
        print(f"   Subdomain numeric ratio: {features.get('subdomain_numeric_ratio', 0):.0%}")
    print(f"   TLD: .{extracted.suffix}")
    
    # Suspicious Indicators
    print(f"\n⚠️  PHISHING INDICATORS:")
    indicators = []
    
    # Check all suspicious patterns
    if features.get('suspicious_tld', 0):
        indicators.append(f"❌ Suspicious TLD (.{extracted.suffix}) - commonly used for phishing")
    
    if features.get('domain_has_digits', 0):
        indicators.append(f"❌ Domain contains digits - unusual for legitimate sites")
    
    if features.get('subdomain_is_numeric_only', 0):
        indicators.append(f"❌ Numeric-only subdomain - strong phishing indicator")
    
    if features.get('long_numeric_subdomain', 0):
        indicators.append(f"❌ Long numeric subdomain (>{features.get('subdomain_length', 0)} chars)")
    
    if features.get('very_long_numeric_subdomain', 0):
        indicators.append(f"❌ Very long numeric subdomain - likely phishing")
    
    if features.get('brand_spoofing_pattern', 0):
        indicators.append(f"❌ Brand spoofing: brand name in subdomain (not main domain)")
    
    if features.get('brand_in_path', 0):
        indicators.append(f"❌ Brand name in URL path - possible impersonation")
    
    if features.get('has_php_extension', 0):
        indicators.append(f"❌ Uses .php extension - common in phishing sites")
    
    if features.get('path_depth', 0) > 3:
        indicators.append(f"❌ Deep path structure ({int(features.get('path_depth', 0))} levels) - suspicious")
    
    if features.get('url_length', 0) > 75:
        indicators.append(f"❌ Long URL ({int(features.get('url_length', 0))} chars) - often used to hide malicious intent")
    
    if features.get('is_https', 0) == 0:
        indicators.append(f"⚠️  Not using HTTPS - security risk")
    
    if features.get('is_ip_address', 0):
        indicators.append(f"❌ Uses IP address instead of domain name")
    
    if features.get('long_subdomain', 0):
        indicators.append(f"❌ Unusually long subdomain")
    
    if features.get('is_url_shortener', 0):
        indicators.append(f"❌ URL shortener detected - hides real destination")
    
    # Show suspicious keywords found
    if suspicious_keywords_found:
        indicators.append(f"❌ Suspicious keywords detected ({len(suspicious_keywords_found)}): {', '.join(suspicious_keywords_found[:5])}")
    
    # Show brand keywords found
    if brand_keywords_found:
        location = []
        if features.get('brand_in_domain', 0):
            location.append("domain")
        if features.get('brand_in_subdomain', 0):
            location.append("subdomain")
        if features.get('brand_in_path', 0):
            location.append("path")
        indicators.append(f"❌ Brand names found in {', '.join(location)}: {', '.join(brand_keywords_found)}")
    
    if indicators:
        for indicator in indicators:
            print(f"   {indicator}")
    else:
        print("   ✅ No major phishing indicators detected")
    
    # Positive indicators
    print(f"\n✅ POSITIVE INDICATORS:")
    positive = []
    
    if features.get('is_https', 0):
        positive.append("✓ Uses HTTPS")
    
    if not features.get('suspicious_tld', 0):
        positive.append("✓ Standard TLD")
    
    if not features.get('domain_has_digits', 0):
        positive.append("✓ No digits in domain")
    
    if features.get('url_length', 0) < 50:
        positive.append("✓ Short, clean URL")
    
    if features.get('is_pure_brand_domain', 0):
        positive.append("✓ Pure brand domain (not in subdomain)")
    
    if positive:
        for p in positive:
            print(f"   {p}")
    else:
        print("   (None found)")
    
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
