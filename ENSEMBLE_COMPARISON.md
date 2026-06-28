# 🏆 WorldCupAI — Ensemble vs Baseline Comparison (Phase 7.1)

> Generated: 2026-06-28 19:52:08

## Full Comparison (Test Set: 2023+)

| model_name | accuracy | roc_auc_macro | pr_auc_macro | log_loss | f1_macro | brier_score | ece | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.6187 | 0.7549 | 0.5942 | 0.8408 | 0.4750 | 0.4961 | 0.0129 | 27.7563 |
| Weighted Soft Voting | 0.6151 | 0.7547 | 0.5932 | 0.8422 | 0.4794 | 0.4970 | 0.0162 | 0.1190 |
| Blending | 0.6170 | 0.7542 | 0.5916 | 0.8711 | 0.4787 | 0.5078 | 0.0451 | 0.0060 |
| Gradient Boosting | 0.6118 | 0.7528 | 0.5901 | 0.8463 | 0.4864 | 0.4992 | 0.0179 | 155.8221 |
| Stacking | 0.6184 | 0.7519 | 0.5920 | 0.8524 | 0.4812 | 0.5009 | 0.0298 | 0.0747 |
| Hard Voting | 0.6134 | 0.7513 | 0.5906 | 0.8638 | 0.4646 | 0.5049 | 0.0358 | 0.0000 |
| Soft Voting | 0.6145 | 0.7513 | 0.5906 | 0.8638 | 0.4649 | 0.5049 | 0.0358 | 0.0000 |
| Random Forest | 0.6140 | 0.7509 | 0.5867 | 0.8520 | 0.4690 | 0.5007 | 0.0219 | 30.5242 |
| ANN | 0.6000 | 0.7364 | 0.5773 | 0.8695 | 0.4767 | 0.5130 | 0.0234 | 33.1900 |
| Extra Trees | 0.6058 | 0.7350 | 0.5706 | 0.8719 | 0.4848 | 0.5132 | 0.0225 | 14.7791 |
| Logistic Regression | 0.5945 | 0.7279 | 0.5562 | 0.8901 | 0.4441 | 0.5240 | 0.0176 | 14.0989 |
| LSTM | 0.4642 | 0.5082 | 0.3419 | 1.0669 | 0.2475 | 0.6439 | 0.0411 | 9.9000 |

## Ensemble Strategies Only

| model_name | accuracy | roc_auc_macro | pr_auc_macro | log_loss | f1_macro | brier_score | ece | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hard Voting | 0.6134 | 0.7513 | 0.5906 | 0.8638 | 0.4646 | 0.5049 | 0.0358 | 0.0000 |
| Soft Voting | 0.6145 | 0.7513 | 0.5906 | 0.8638 | 0.4649 | 0.5049 | 0.0358 | 0.0000 |
| Weighted Soft Voting | 0.6151 | 0.7547 | 0.5932 | 0.8422 | 0.4794 | 0.4970 | 0.0162 | 0.1190 |
| Stacking | 0.6184 | 0.7519 | 0.5920 | 0.8524 | 0.4812 | 0.5009 | 0.0298 | 0.0747 |
| Blending | 0.6170 | 0.7542 | 0.5916 | 0.8711 | 0.4787 | 0.5078 | 0.0451 | 0.0060 |

## Key Findings

- Best Overall Ensemble : **Weighted Soft Voting**
- Most Calibrated       : **Weighted Soft Voting**
- Most Robust (ROC-AUC) : **Weighted Soft Voting**
- Fastest               : **Soft Voting**
