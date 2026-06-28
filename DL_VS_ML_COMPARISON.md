# 🆚 WorldCupAI — Deep Learning vs Machine Learning Comparison (Phase 6)

> Generated: 2026-06-28 17:37:57

## Full Model Comparison (Test Set: 2023+)

| model_name | accuracy | roc_auc_macro | pr_auc_macro | log_loss | f1_macro | brier_score | ece | training_time_sec | prediction_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.6187 | 0.7549 | 0.5942 | 0.8408 | 0.4750 | 0.4961 | 0.0129 | 27.7563 | 0.0524 |
| Gradient Boosting | 0.6118 | 0.7528 | 0.5901 | 0.8463 | 0.4864 | 0.4992 | 0.0179 | 155.8221 | 0.0453 |
| Random Forest | 0.6140 | 0.7509 | 0.5867 | 0.8520 | 0.4690 | 0.5007 | 0.0219 | 30.5242 | 0.0516 |
| ANN | 0.6000 | 0.7364 | 0.5773 | 0.8695 | 0.4767 | 0.5130 | 0.0234 | 33.1900 | 0.0629 |
| Extra Trees | 0.6058 | 0.7350 | 0.5706 | 0.8719 | 0.4848 | 0.5132 | 0.0225 | 14.7791 | 0.0772 |
| Logistic Regression | 0.5945 | 0.7279 | 0.5562 | 0.8901 | 0.4441 | 0.5240 | 0.0176 | 14.0989 | 0.0008 |
| LSTM | 0.4642 | 0.5082 | 0.3419 | 1.0669 | 0.2475 | 0.6439 | 0.0411 | 9.9000 | 0.0842 |

## Key Observations

- DL models are compared against ALL Phase 5 optimized + calibrated ML models.
- Metrics are computed on identical test sets (2023-01-01 onwards).
- All probability predictions have been exported to `predictions/` for ensemble use.

## Prediction Files (for Phase 7 Ensemble)

| Model | File |
|---|---|
| ANN | `predictions/ann_predictions.csv` |
| LSTM | `predictions/lstm_predictions.csv` |
| XGBoost | `predictions/xgboost_predictions.csv` |
| Gradient Boosting | `predictions/gradient_boosting_predictions.csv` |
| Random Forest | `predictions/random_forest_predictions.csv` |
| Extra Trees | `predictions/extra_trees_predictions.csv` |
| Logistic Regression | `predictions/logistic_regression_predictions.csv` |

> All prediction CSV files contain: match_id, true_label, predicted_label, prob_away_win, prob_draw, prob_home_win
