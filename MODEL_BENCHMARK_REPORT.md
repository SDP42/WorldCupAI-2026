# 📊 WorldCupAI — Model Benchmark Report

Generated on: 2026-06-28 16:49:02

## Baseline Model Comparison Table
| Rank | Model Name | Accuracy | Log Loss | Brier Score | F1-Score (Macro) | Training Time (s) | Inference Time (s) |
|---|---|---|---|---|---|---|---|
| 1 | **Gradient Boosting** | 0.6534 | 0.8007 | 0.4688 | 0.5085 | 6.45 | 0.0194 |
| 2 | **Random Forest** | 0.6479 | 0.8196 | 0.4778 | 0.4841 | 0.28 | 0.0382 |
| 3 | **Decision Tree** | 0.6389 | 0.8350 | 0.4823 | 0.4758 | 0.06 | 0.0011 |
| 4 | **XGBoost** | 0.6353 | 0.8137 | 0.4806 | 0.5201 | 0.53 | 0.0067 |
| 5 | **AdaBoost** | 0.6283 | 1.0395 | 0.6262 | 0.4765 | 0.62 | 0.0163 |
| 6 | **Logistic Regression** | 0.6077 | 0.8614 | 0.5048 | 0.4605 | 0.14 | 0.0033 |
| 7 | **Extra Trees** | 0.6032 | 0.9020 | 0.5280 | 0.4288 | 0.12 | 0.0370 |
| 8 | **Naive Bayes** | 0.5750 | 1.6585 | 0.6334 | 0.4920 | 0.00 | 0.0046 |
| 9 | **K-Nearest Neighbors** | 0.5328 | 2.0040 | 0.5827 | 0.4542 | 0.00 | 0.3121 |
| 10 | **SVM (Linear)** | 0.3552 | 1.0629 | 0.6424 | 0.3233 | 4.26 | 0.4717 |


## Core Observations
- **Top Performer**: The best performing baseline model is **Gradient Boosting** with an accuracy of **0.6534** and Log Loss of **0.8007**.
- **Calibration recommendation**: Isotonic regression is recommended for calibration in subsequent phases.
