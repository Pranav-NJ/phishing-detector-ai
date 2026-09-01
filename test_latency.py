import time
import sys
import os

# Add path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'ml-api'))
sys.path.insert(0, os.path.join(BASE_DIR, 'ml-api', 'utils'))

from enhanced_feature_extraction import CompleteFeatureExtractor
import joblib

# Load model
print("Loading model...")
model_data = joblib.load('ml-api/models/phishing_model.pkl')
model = model_data['model']
feature_names = model_data.get('feature_names', [])
extractor = CompleteFeatureExtractor()

print("Model loaded successfully!\n")

# Test URLs
test_urls = [
    "https://www.google.com",
    "http://paypal-verify.suspicious.com",
    "https://github.com",
    "http://00000000883838383992929292222.ratingandreviews.in",
    "https://www.amazon.com"
]

print("=" * 80)
print("LATENCY BENCHMARK TEST")
print("=" * 80)

latencies = []

for url in test_urls:
    # Measure time
    start = time.perf_counter()
    
    # Feature extraction
    features = extractor.extract_all_features(url)
    feature_vector = [features.get(name, 0.0) for name in feature_names]
    
    # ML prediction
    prediction_proba = model.predict_proba([feature_vector])[0]
    
    end = time.perf_counter()
    
    # Calculate latency in milliseconds
    latency_ms = (end - start) * 1000
    latencies.append(latency_ms)
    
    print(f"\nURL: {url[:60]}")
    print(f"  Latency: {latency_ms:.2f}ms")
    print(f"  Prob Phishing: {prediction_proba[1]:.4f}")

print("\n" + "=" * 80)
print("LATENCY STATISTICS")
print("=" * 80)

avg_latency = sum(latencies) / len(latencies)
min_latency = min(latencies)
max_latency = max(latencies)

print(f"Average Latency: {avg_latency:.2f}ms")
print(f"Min Latency:     {min_latency:.2f}ms")
print(f"Max Latency:     {max_latency:.2f}ms")

print("\n" + "=" * 80)
print("RESUME CLAIM VERIFICATION")
print("=" * 80)

if avg_latency < 30:
    print(f"[+] VERIFIED: Sub-30ms latency claim is VALID")
    print(f"    Average: {avg_latency:.2f}ms < 30ms threshold")
elif avg_latency < 50:
    print(f"[!] CAUTION: Use 'sub-50ms' instead of 'sub-30ms'")
    print(f"    Average: {avg_latency:.2f}ms is between 30-50ms")
else:
    print(f"[X] WARNING: Latency claim needs adjustment")
    print(f"    Average: {avg_latency:.2f}ms exceeds 50ms")

print("\n" + "=" * 80)
