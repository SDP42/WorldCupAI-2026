# 📐 WorldCupAI — Probability Calibration Guide (Phase 5)

> Generated: 2026-06-28 17:05:20

## Why Calibration Matters

A model predicting 70% for a Home Win should be correct ~70% of the time.
Without calibration, tree-based models tend to push probabilities to extremes
(overconfident), while linear models may underestimate confidence.

## Calibration Metrics

| Metric | Definition | Target |
|---|---|---|
| **ECE** (Expected Calibration Error) | Weighted avg. of abs(accuracy - confidence) across bins | < 0.05 |
| **MCE** (Maximum Calibration Error) | Worst-case bin error | < 0.10 |
| **Brier Score** | Mean squared error of probability vs actual (lower = better) | < 0.45 |

## Methods Compared

### Platt Scaling (Sigmoid)
- Fits a logistic regression over the model's raw probabilities.
- Works well for small calibration sets (< 1,000 samples).
- Parametric — assumes a sigmoid-shaped miscalibration.

### Isotonic Regression
- Non-parametric monotone fit over predicted probabilities.
- Works best for larger calibration sets (> 1,000 samples).
- More flexible but can overfit on small sets.

## Selection Logic

Both methods are fitted using `CalibratedClassifierCV(cv='prefit')` on the
validation set. The method with the lower **Brier Score** on the validation set
is selected and saved as `calibrated_model.pkl`.

## Reliability Diagrams

Each model's `calibration_curve.png` shows the reliability diagram per class
(Away Win / Draw / Home Win) comparing uncalibrated vs calibrated probabilities
to the perfect calibration diagonal.
