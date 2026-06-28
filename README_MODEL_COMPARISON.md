# 📊 WorldCupAI — Model Comparison & Benchmark Results

This document contains a comparison and technical analysis of the 12 baseline models trained in Phase 4.

## 1. Baseline Model Comparison

The models are ranked based on their multiclass validation accuracy on the 2019-2022 dataset:

| Rank | Model Name | Validation Accuracy | Log Loss | Brier Score | Training Time (s) | Inference Time (s) |
|---|---|---|---|---|---|---|
| 1 | **Gradient Boosting** | **0.6534** | 0.8007 | 0.4688 | 6.45 | 0.0194 |
| 2 | **Random Forest** | **0.6479** | 0.8196 | 0.4778 | 0.28 | 0.0382 |
| 3 | **Decision Tree** | **0.6389** | 0.8350 | 0.4823 | 0.06 | 0.0011 |
| 4 | **XGBoost** | **0.6353** | 0.8137 | 0.4806 | 0.53 | 0.0067 |
| 5 | **AdaBoost** | **0.6283** | 1.0395 | 0.6262 | 0.62 | 0.0163 |
| 6 | **Logistic Regression** | **0.6077** | 0.8614 | 0.5048 | 0.14 | 0.0033 |
| 7 | **Extra Trees** | **0.6032** | 0.9020 | 0.5280 | 0.12 | 0.0370 |
| 8 | **Naive Bayes** | **0.5750** | 1.6585 | 0.6334 | 0.00 | 0.0046 |
| 9 | **K-Nearest Neighbors** | **0.5328** | 2.0040 | 0.5827 | 0.00 | 0.3121 |
| 10 | **SVM (Linear)** | **0.3552** | 1.0629 | 0.6424 | 4.26 | 0.4717 |

---

## 2. Key Insights

1. **Tree-based Ensemble Superiority**: Ensembles like **Gradient Boosting** and **Random Forest** significantly outperform single classifiers and linear models. Gradient Boosting achieved the highest accuracy of **0.6534** and the lowest Log Loss of **0.8007**.
2. **Inference Latency**: Single decision trees and linear models are the fastest (less than 3ms), while distance-based/kernel models (KNN, SVM) are significantly slower due to calculation overhead.
3. **Probability Calibration**: Tree-based models tend to output uncalibrated probabilities due to ensemble averaging. Running Isotonic Regression is recommended for Phase 5 to get well-calibrated scoreline probabilities.
