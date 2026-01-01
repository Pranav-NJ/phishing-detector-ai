"""
Comprehensive Test Suite for Phishing Detector

Tests the model with various URLs to verify it correctly handles:
1. Phishing URLs with HTTPS
2. Phishing URLs with HTTP  
3. Legitimate URLs with HTTPS
4. Edge cases
"""

import subprocess
import os
import sys

# Add project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLAPI_DIR = os.path.join(BASE_DIR, "ml-api")

def test_url(url, expected):
    """Test a single URL and return result"""
    cmd = [sys.executable, "scripts/predict_model.py", url]
    try:
        result = subprocess.run(
            cmd,
            cwd=MLAPI_DIR,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=10
        )
        
        output = result.stdout
        
        if "PHISHING DETECTED" in output:
            prediction = "PHISHING"
        elif "SAFE / LEGITIMATE" in output:
            prediction = "LEGITIMATE"
        else:
            prediction = "ERROR"
        
        # Extract confidence
        conf_line = [line for line in output.split('\n') if 'Confidence:' in line]
        if conf_line:
            conf = conf_line[0].split(':')[1].strip()
        else:
            conf = "N/A"
        
        match = "✅" if prediction == expected else "❌"
        
        return {
            'url': url,
            'expected': expected,
            'predicted': prediction,
            'confidence': conf,
            'match': match
        }
    except Exception as e:
        return {
            'url': url,
            'expected': expected,
            'predicted': f"ERROR: {str(e)}",
            'confidence': "N/A",
            'match': "❌"
        }


def main():
    print("=" * 80)
    print("🧪 COMPREHENSIVE PHISHING DETECTOR TEST SUITE")
    print("=" * 80)
    
    # Test cases: (URL, Expected)
    test_cases = [
        # Phishing with suspicious TLD
        ("http://www.shprakserf.gq", "PHISHING"),
        ("https://www.shprakserf.gq", "PHISHING"),
        
        # Firebase brand spoofing
        ("https://service-mitld.firebaseapp.com/", "PHISHING"),
        ("http://att-103731-107123.weeblysite.com/", "PHISHING"),
        
        # Legitimate sites
        ("https://www.google.com", "LEGITIMATE"),
        ("https://www.microsoft.com", "LEGITIMATE"),
        ("https://github.com", "LEGITIMATE"),
        ("https://www.amazon.com", "LEGITIMATE"),
        
        # Edge cases
        ("https://www.paypal.com", "LEGITIMATE"),
        ("https://secure.paypal.com", "LEGITIMATE"),
    ]
    
    print(f"\n📋 Running {len(test_cases)} test cases...\n")
    
    results = []
    for url, expected in test_cases:
        print(f"Testing: {url[:60]:<60}", end=" ")
        result = test_url(url, expected)
        results.append(result)
        print(f"{result['match']} {result['predicted']} ({result['confidence']})")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    correct = sum(1 for r in results if r['match'] == "✅")
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0
    
    print(f"\n✅ Passed: {correct}/{total} ({accuracy:.1f}%)")
    print(f"❌ Failed: {total - correct}/{total}")
    
    # Failed tests
    failed = [r for r in results if r['match'] == "❌"]
    if failed:
        print(f"\n❌ FAILED TESTS:")
        for r in failed:
            print(f"   URL: {r['url']}")
            print(f"   Expected: {r['expected']}, Got: {r['predicted']} ({r['confidence']})")
            print()
    
    print("=" * 80)
    
    if accuracy >= 80:
        print("✅ TEST SUITE PASSED! Model is working correctly.")
        print("   Note: ~90% accuracy is expected for phishing detection.")
        return 0
    else:
        print("❌ TEST SUITE FAILED! Review the model and training data.")
        return 1


if __name__ == "__main__":
    exit(main())
