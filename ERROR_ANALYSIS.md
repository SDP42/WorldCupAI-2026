# 🔍 WorldCupAI — Error Analysis Report (Phase 5)

> Generated: 2026-06-28 17:05:20

This report examines prediction errors, calibration drift, and hard-to-predict
match patterns across all Phase 5 optimized models.

---

## 1. Per-Model Summary

| Model | Test Errors | Error Rate | Confidence (Correct) | Confidence (Wrong) | Hard Matches |
|---|---|---|---|---|---|
| **Logistic Regression** | 1,478 / 3,645 | 40.55% | 0.638 | 0.5316 | 729 (20.0%) |
| **Random Forest** | 1,407 / 3,645 | 38.60% | 0.64 | 0.5244 | 729 (20.0%) |
| **Extra Trees** | 1,438 / 3,645 | 39.45% | 0.6276 | 0.5217 | 729 (20.0%) |
| **Gradient Boosting** | 1,415 / 3,645 | 38.82% | 0.6764 | 0.5541 | 729 (20.0%) |
| **XGBoost** | 1,390 / 3,645 | 38.13% | 0.6663 | 0.5466 | 729 (20.0%) |

---

## 2. False Positive / False Negative Breakdown

### Logistic Regression

| Class | TP | FP | FN | TN | Precision | Recall |
|---|---|---|---|---|---|---|
| Away Win | 632 | 468 | 448 | 2097 | 0.5745 | 0.5852 |
| Draw | 14 | 36 | 827 | 2768 | 0.2800 | 0.0166 |
| Home Win | 1521 | 974 | 203 | 947 | 0.6096 | 0.8823 |

### Random Forest

| Class | TP | FP | FN | TN | Precision | Recall |
|---|---|---|---|---|---|---|
| Away Win | 748 | 546 | 332 | 2019 | 0.5781 | 0.6926 |
| Draw | 20 | 41 | 821 | 2763 | 0.3279 | 0.0238 |
| Home Win | 1470 | 820 | 254 | 1101 | 0.6419 | 0.8527 |

### Extra Trees

| Class | TP | FP | FN | TN | Precision | Recall |
|---|---|---|---|---|---|---|
| Away Win | 673 | 475 | 407 | 2090 | 0.5862 | 0.6231 |
| Draw | 62 | 110 | 779 | 2694 | 0.3605 | 0.0737 |
| Home Win | 1472 | 853 | 252 | 1068 | 0.6331 | 0.8538 |

### Gradient Boosting

| Class | TP | FP | FN | TN | Precision | Recall |
|---|---|---|---|---|---|---|
| Away Win | 730 | 540 | 350 | 2025 | 0.5748 | 0.6759 |
| Draw | 52 | 94 | 789 | 2710 | 0.3562 | 0.0618 |
| Home Win | 1448 | 781 | 276 | 1140 | 0.6496 | 0.8399 |

### XGBoost

| Class | TP | FP | FN | TN | Precision | Recall |
|---|---|---|---|---|---|---|
| Away Win | 759 | 548 | 321 | 2017 | 0.5807 | 0.7028 |
| Draw | 24 | 38 | 817 | 2766 | 0.3871 | 0.0285 |
| Home Win | 1472 | 804 | 252 | 1117 | 0.6467 | 0.8538 |

---

## 3. Calibration Drift per Class (ECE)

| Model | Away Win ECE | Draw ECE | Home Win ECE |
|---|---|---|---|
| **Logistic Regression** | 0.01423 | 0.01536 | 0.02311 |
| **Random Forest** | 0.02837 | 0.00638 | 0.03108 |
| **Extra Trees** | 0.01836 | 0.0176 | 0.03147 |
| **Gradient Boosting** | 0.01774 | 0.02061 | 0.01534 |
| **XGBoost** | 0.01177 | 0.01044 | 0.01642 |

---

## 4. Key Findings

- **High confidence on misclassified samples** indicates overconfident models — calibration is critical.
- **Hard matches** (bottom 20% of true-class probability) are predominantly **Draw** outcomes,
  reflecting football's inherent unpredictability for draws.
- **Away Win FP rates** tend to be high as models default to Home Win bias.
- Models with ECE > 0.05 per class should have their calibration method revisited.

---

> See individual model directories for `error_confidence_distribution.png` plots.