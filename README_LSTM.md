# 🟢 WorldCupAI — LSTM Architecture

> Generated: 2026-06-28 17:37:57

## Architecture

```
Input: (batch, SEQ_LEN=5, 37 features)
  → LSTM(37 → 128, num_layers=2, dropout=0.3, batch_first=True)
  → LayerNorm(128)
  → last hidden state (batch, 128)
  → Linear(128, 64) + ReLU + Dropout(0.2)
  → Linear(64, 3)  [logits → softmax]
```

## Sequence Construction (Leakage-Safe)

For each match at absolute index `i` in the chronologically sorted feature store:
- Input sequence = feature rows `[i-5 : i]` (5 prior matches, global)
- If `i < 5`, sequence is left-padded with zeros
- All 37 feature columns were computed with `.shift(1)` in Phase 3 (no future leakage)

## Training Config

| Hyperparameter | Value |
|---|---|
| Sequence Length | 5 |
| Hidden Dim | 128 |
| LSTM Layers | 2 |
| LSTM Dropout | 0.3 |
| FC Dropout | 0.2 |
| Weight Init | Kaiming (ih) + Orthogonal (hh) + forget_bias=1 |
| Optimizer | AdamW |
| Loss | LabelSmoothingCrossEntropy (ε=0.05) |
| Early Stopping | patience=15 |
| Batch Size | 256 |

## Test Set Results

| Metric | Value |
|---|---|
| Accuracy | 0.4642 |
| ROC-AUC | 0.5082 |
| Log Loss | 1.0669 |
| F1 (macro) | 0.2475 |
| Brier Score | 0.6439 |
| ECE | 0.0411 |
| Training Time | 9.9s |
