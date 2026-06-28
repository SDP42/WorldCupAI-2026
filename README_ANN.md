# 🔵 WorldCupAI — ANN Architecture

> Generated: 2026-06-28 17:37:57

## Architecture

```
Input (37 features)
  → Linear(37, 256) + BatchNorm1d + ReLU + Dropout(0.3)
  → Linear(256, 128) + BatchNorm1d + ReLU + Dropout(0.3)
  → Linear(128, 64)  + BatchNorm1d + ReLU + Dropout(0.2)
  → Linear(64, 3)   [logits → softmax]
```

## Training Config

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay (L2) | 1e-4 |
| Loss | LabelSmoothingCrossEntropy (ε=0.05) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early Stopping | patience=15 |
| Batch Size | 256 |
| Max Epochs | 200 |
| Weight Init | Kaiming uniform (He) |
| Gradient Clipping | max_norm=1.0 |

## Test Set Results

| Metric | Value |
|---|---|
| Accuracy | 0.6000 |
| ROC-AUC | 0.7364 |
| Log Loss | 0.8695 |
| F1 (macro) | 0.4767 |
| Brier Score | 0.5130 |
| ECE | 0.0234 |
| Training Time | 33.2s |
