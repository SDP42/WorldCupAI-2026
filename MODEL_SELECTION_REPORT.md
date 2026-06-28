# 🏆 WorldCupAI — Model Selection Report (Phase 5)

> Generated: 2026-06-28 17:05:20

This report provides **objective, metric-weighted rankings** for all Phase 5
optimized and calibrated models. Selections are fully automated — no manual
override.

---

## Scoring Weights

| Metric | Weight | Direction |
|---|---|---|
| Accuracy | 20% | ↑ Higher better |
| ROC-AUC (macro) | 20% | ↑ Higher better |
| Log Loss | 20% | ↓ Lower better |
| F1 (macro) | 15% | ↑ Higher better |
| Brier Score | 15% | ↓ Lower better |
| ECE | 10% | ↓ Lower better |

---

## Full Ranking

| rank | model_name | composite_score | accuracy | roc_auc_macro | log_loss | f1_macro | brier_score | ece | prediction_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | XGBoost | 0.9597 | 0.6187 | 0.7549 | 0.8408 | 0.4750 | 0.4961 | 0.0129 | 0.0524 |
| 2 | Gradient Boosting | 0.8365 | 0.6118 | 0.7528 | 0.8463 | 0.4864 | 0.4992 | 0.0179 | 0.0453 |
| 3 | Random Forest | 0.7059 | 0.6140 | 0.7509 | 0.8520 | 0.4690 | 0.5007 | 0.0219 | 0.0516 |
| 4 | Extra Trees | 0.4222 | 0.6058 | 0.7350 | 0.8719 | 0.4848 | 0.5132 | 0.0225 | 0.0772 |
| 5 | Logistic Regression | 0.0515 | 0.5945 | 0.7279 | 0.8901 | 0.4441 | 0.5240 | 0.0176 | 0.0008 |

---

## Objective Selections

| Role | Selected Model |
|---|---|
| 🥇 **Best Overall** | XGBoost |
| ⚡ **Fastest Inference** | Logistic Regression |
| 🛡️ **Most Robust** | XGBoost |
| 📐 **Best Calibrated** | XGBoost |

---

## Top 5 Ensemble Candidates

1. **XGBoost**
2. **Gradient Boosting**
3. **Random Forest**
4. **Extra Trees**
5. **Logistic Regression**

These models are recommended for inclusion in Phase 6 (Deep Learning Comparison)
and ultimately Phase 7 (Ensemble Construction).

---

## Recommendations for Phase 6

1. The **XGBoost** model should be used as the ML baseline
   for Deep Learning comparison.
2. The **XGBoost** model provides the most trustworthy
   probability estimates for match outcome confidence scoring.
3. All **Top 5 ensemble candidates** should be retained for stacking/voting
   ensemble construction in a later phase.
4. Models with Log Loss > 0.85 post-calibration should be excluded from the ensemble.

---

> All model artifacts are stored in `models/{model_name}_optimized/`.
