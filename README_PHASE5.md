# 🚀 WorldCupAI — Phase 5: Hyperparameter Optimization

> Generated: 2026-06-28 17:05:20

## Overview

Phase 5 optimizes the top ML models identified in Phase 4, applies probability
calibration, and objectively selects the best candidates for the final ensemble.

## Pipeline

```
feature_store.parquet
    → Time-Aware Split (2005-2018 train / 2019-2022 val / 2023+ test)
    → PreprocessingPipeline (Phase 4 — unchanged)
    → HyperparameterOptimizer (GridSearch / RandomizedSearch + TimeSeriesSplit)
    → Early Stopping (XGBoost / LightGBM / CatBoost)
    → ProbabilityCalibrator (Platt vs Isotonic → best selected by Brier Score)
    → ModelEvaluator (test set: Acc, ROC-AUC, Log Loss, F1, Brier, ECE, MCE)
    → ErrorAnalyzer (FP/FN, confidence distributions, hard matches)
    → ModelCandidateSelector (weighted composite score)
```

## Execution

```bash
python3 optimize_models.py
```

## Results Summary

| rank | model_name | accuracy | roc_auc_macro | log_loss | f1_macro | brier_score | ece |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | XGBoost | 0.6187 | 0.7549 | 0.8408 | 0.4750 | 0.4961 | 0.0129 |
| 2 | Gradient Boosting | 0.6118 | 0.7528 | 0.8463 | 0.4864 | 0.4992 | 0.0179 |
| 3 | Random Forest | 0.6140 | 0.7509 | 0.8520 | 0.4690 | 0.5007 | 0.0219 |
| 4 | Extra Trees | 0.6058 | 0.7350 | 0.8719 | 0.4848 | 0.5132 | 0.0225 |
| 5 | Logistic Regression | 0.5945 | 0.7279 | 0.8901 | 0.4441 | 0.5240 | 0.0176 |

## Key Selections

| Role | Model |
|---|---|
| 🥇 Best Overall | XGBoost |
| 📐 Best Calibrated | XGBoost |
| ⚡ Fastest | Logistic Regression |

## Artifacts

All optimized model artifacts are stored in `models/{name}_optimized/`:
- `model.pkl` — optimized model
- `calibrated_model.pkl` — calibrated model
- `preprocessing.pkl` — unchanged Phase 4 pipeline
- `best_params.json` — best hyperparameters found
- `metrics.json` — full test-set metric suite
- `calibration.json` — ECE, MCE, Brier before/after
- `feature_importance.csv` — ranked feature importances
- `training_log.json` — full CV search results
- `roc_curve.png`, `confusion_matrix.png`, `calibration_curve.png`
- `error_confidence_distribution.png`
