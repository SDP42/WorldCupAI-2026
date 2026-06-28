# 📊 WorldCupAI — Deep Learning Benchmark Report

> Generated: 2026-06-28 17:37:57

## Benchmark Summary

| Dimension | Winner |
|---|---|
| Accuracy | ML |
| ROC-AUC | ML |
| Training Speed | ML (significantly faster) |
| Inference Speed | ML (significantly faster) |
| Calibration (ECE) | See DL_VS_ML_COMPARISON.md |

## Does Deep Learning Add Value?

⚠️ **Marginal** — DL models perform slightly below the best optimized ML model on this dataset. However, their diverse prediction distributions add diversity value for ensemble construction.

## Recommendation

1. **Include ANN in Phase 7 ensemble** — provides complementary probability estimates.
2. **Include LSTM in Phase 7 ensemble** — captures temporal sequence patterns.
3. Both DL models should be included as ensemble members alongside the top 3 ML models.

## Artefact Locations

- `models/ann/` — ANN model, config, metrics, plots
- `models/lstm/` — LSTM model, config, metrics, plots
- `predictions/` — All model test-set predictions for ensemble use
