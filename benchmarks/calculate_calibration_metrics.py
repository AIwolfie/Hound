import json
import math
from pathlib import Path

report_file = Path("data/benchmarks/phase_6/reports/phase_6_benchmark_report.json")
with open(report_file, "r", encoding="utf-8") as f:
    report = json.load(f)

cases = report["per_case_results"]

print(f"Total benchmark cases: {len(cases)}")
print("-" * 60)

# Binary target: y = 1 if localized within 25km, 0 otherwise
y_true = []
y_prob = []
details = []

for c in cases:
    is_neg = c["is_negative_control"]
    w25 = c["within_25km"]
    prob = c["calibrated_probability"]
    
    # Ground truth outcome
    # For a positive target, correct means localized within tolerance (<= 25km)
    # For a negative control, correct output should be non-localized (y=0 for coordinate prediction)
    y = 1 if (not is_neg and w25) else 0
    y_true.append(y)
    y_prob.append(prob)
    details.append((c["case_id"], c["name"], prob, y, c["calibrated_tier"], c["abstention_recommended"]))
    print(f"{c['case_id']} | Prob: {prob:.3f} | Ground Truth y={y} | Tier: {c['calibrated_tier']:<12} | Abstain: {c['abstention_recommended']}")

print("-" * 60)

# 1. Brier Score = (1/N) * sum((prob - y)^2)
n = len(y_true)
brier_score = sum((p - y) ** 2 for p, y in zip(y_prob, y_true)) / n
print(f"Empirical Brier Score: {brier_score:.4f}")

# 2. Expected Calibration Error (ECE)
# Partition into M bins (e.g., 5 bins: [0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0])
bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
ece = 0.0

print("\nReliability Diagram / Calibration Bins:")
print(f"{'Bin Range':<15} | {'Count':<6} | {'Mean Pred Prob':<16} | {'Empirical Accuracy':<20} | {'Abs Calibration Error'}")
print("-" * 75)

for b_low, b_high in bins:
    bin_items = [(p, y) for p, y in zip(y_prob, y_true) if b_low <= p < b_high or (b_high == 1.0 and p == 1.0)]
    count = len(bin_items)
    if count > 0:
        mean_p = sum(p for p, _ in bin_items) / count
        mean_y = sum(y for _, y in bin_items) / count
        diff = abs(mean_p - mean_y)
        ece += (count / n) * diff
        print(f"[{b_low:.1f}, {b_high:.1f})      | {count:<6} | {mean_p:<16.4f} | {mean_y:<20.4f} | {diff:.4f}")
    else:
        print(f"[{b_low:.1f}, {b_high:.1f})      | 0      | {'N/A':<16} | {'N/A':<20} | N/A")

print("-" * 75)
print(f"Expected Calibration Error (ECE): {ece:.4f}")
