import joblib

# Load model
model_data = joblib.load('models/phishing_model.pkl')
perf = model_data.get('performance', {})

# Get confusion matrix values
tp = perf.get('true_positives', 0)
fp = perf.get('false_positives', 0)
tn = perf.get('true_negatives', 0)
fn = perf.get('false_negatives', 0)

print("=" * 60)
print("CONFUSION MATRIX FROM SAVED MODEL")
print("=" * 60)
print(f"True Positives (TP):  {tp:,}")
print(f"False Positives (FP): {fp:,}")
print(f"True Negatives (TN):  {tn:,}")
print(f"False Negatives (FN): {fn:,}")

# Calculate metrics
if (fp + tn) > 0:
    fpr = fp / (fp + tn)
    print(f"\n{'='*60}")
    print(f"FALSE POSITIVE RATE: {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"{'='*60}")
    
if (fn + tp) > 0:
    fnr = fn / (fn + tp)
    print(f"False Negative Rate: {fnr:.4f} ({fnr*100:.2f}%)")

# Additional metrics
print(f"\n{'='*60}")
print("PERFORMANCE METRICS")
print(f"{'='*60}")
print(f"Accuracy:  {perf.get('accuracy', 0):.4f} ({perf.get('accuracy', 0)*100:.2f}%)")
print(f"Precision: {perf.get('precision', 0):.4f} ({perf.get('precision', 0)*100:.2f}%)")
print(f"Recall:    {perf.get('recall', 0):.4f} ({perf.get('recall', 0)*100:.2f}%)")
print(f"F1-Score:  {perf.get('f1_score', 0):.4f}")
print(f"ROC-AUC:   {perf.get('roc_auc', 0):.4f}")
