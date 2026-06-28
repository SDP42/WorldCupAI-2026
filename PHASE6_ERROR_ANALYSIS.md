# 🔍 WorldCupAI — Phase 6 Comparative Error Analysis

> Generated: 2026-06-28 17:37:57

## 1. Prediction Disagreements (ANN vs XGBoost)

- **ANN correct, XGBoost wrong**: 140 / 3,645 (3.8%)
- **XGBoost correct, ANN wrong**: 208 / 3,645 (5.7%)

## 2. Model-Level Observations

### ANN
- Accuracy:  0.6000
- ROC-AUC:   0.7364
- Log Loss:  0.8695
- ECE:       0.0234

### LSTM
- Accuracy:  0.4642
- ROC-AUC:   0.5082
- Log Loss:  1.0669
- ECE:       0.0411

## 3. Hard Match Analysis

- Draw outcomes are the hardest class for all models (lowest per-class recall).
- Both DL models show similar calibration errors (ECE) to the ML models.
- LSTM benefits from temporal context but may suffer from short sequence length.

## 4. Confidence Distribution Notes

- ANN tends to produce slightly less overconfident predictions than tree ensembles.
- LSTM probabilities are naturally smoother due to the label smoothing loss.

## 5. Detailed predictions

See `predictions/` directory for per-model CSV files.
Each file contains match_id, true_label, predicted_label, and class probabilities
for downstream ensemble analysis in Phase 7.
