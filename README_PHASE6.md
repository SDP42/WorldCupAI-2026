# 🧠 WorldCupAI — Phase 6: Deep Learning Benchmark

> Generated: 2026-06-28 17:37:57

## Overview

Phase 6 trains and evaluates two deep learning architectures (ANN and LSTM)
and compares them against all Phase 5 optimized ML models on the same test set.

## Pipeline

```
feature_store.parquet
  → Time-Aware Split (2005-2018 / 2019-2022 / 2023+)
  → PreprocessingPipeline(scale=True)   [reused from Phase 4/5]
  → ANN: TabularDataset → feed-forward NN → evaluate on test set
  → LSTM: SequenceBuilder(seq_len=5) → SequenceDataset → LSTM → evaluate
  → Load Phase 5 calibrated models → evaluate same test set
  → Export predictions/{model}_predictions.csv for all models
  → DL_VS_ML_COMPARISON.md
  → PHASE_6_APPROVAL_REPORT.md
```

## Execution

```bash
python3 train_deep_learning.py
```

## Results Summary

| Model | Accuracy | ROC-AUC | Log Loss |
|---|---|---|---|
| ANN (DL) | 0.6000 | 0.7364 | 0.8695 |
| LSTM (DL) | 0.4642 | 0.5082 | 1.0669 |

## Artifacts

```
models/ann/          → model.pt, model_complete.pt, model_config.json, metrics.json ...
models/lstm/         → model.pt, model_complete.pt, model_config.json, metrics.json ...
predictions/         → ann_predictions.csv, lstm_predictions.csv, xgboost_predictions.csv ...
DL_VS_ML_COMPARISON.md
PHASE6_ERROR_ANALYSIS.md
PHASE_6_APPROVAL_REPORT.md
```
