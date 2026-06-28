# 🤝 Ensemble Explainability Report

This report breaks down the inner workings of our Unified Ensemble Pipeline.

## Model Voting Weights

| Base Model | Ensemble Voting Weight |
|---|---|
| XGBoost | 0.4860 |
| Gradient Boosting | 0.5140 |
| Random Forest | 0.0000 |
| Extra Trees | 0.0000 |
| Logistic Regression | 0.0000 |
| ANN | 0.0000 |
| LSTM | 0.0000 |

## Model Agreement Statistics
* **Average Model Agreement**: 50.0%
* **Ensemble Method**: Weighted Soft Voting
* **Dominant Sub-Predictors**: XGBoost and Gradient Boosting contribute 100% of the prediction weight based on Phase 7.1 validation optimizations.
