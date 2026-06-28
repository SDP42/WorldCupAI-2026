# 🧠 WorldCupAI — Deep Learning Framework Guide

> Generated: 2026-06-28 17:37:57

## Framework

**PyTorch 2.9** with **Apple MPS (Metal GPU) acceleration**.
Auto-detects MPS → CUDA → CPU.

## Package Structure

```
src/deep_learning/
  base_model.py          Abstract base (save/load/predict_proba interface)
  ann_model.py           Feed-forward ANN (BN + Dropout + L2 + He init)
  lstm_model.py          LSTM sequence classifier (LayerNorm + orthogonal init)
  dataset.py             TabularDataset + SequenceDataset + SequenceBuilder
  losses.py              LabelSmoothingCrossEntropyLoss
  metrics.py             Thin shim → reuses src/models/metrics (Phase 4/5)
  callbacks.py           EarlyStopping, ModelCheckpoint, TrainingLogger
  trainer.py             Unified training loop (AdamW + ReduceLROnPlateau)
  evaluation.py          DLEvaluator + export_ml_predictions
  model_registry.py      Extends Phase 5 ModelRegistry for DL
  prediction_interface.py Unified predict_proba for Phase 7 ensemble
```

## Saved Artifacts Per Model

| File | Description |
|---|---|
| `model.pt` | PyTorch state dict (recommended for loading) |
| `model_complete.pt` | Full serialised model object |
| `model_config.json` | Architecture + hyperparameters |
| `preprocessing.pkl` | sklearn pipeline (identical to Phase 5) |
| `training_history.json` | Per-epoch loss/accuracy log |
| `metrics.json` | Full test-set metric suite |
| `learning_curves.png` | Train vs val loss/accuracy |
| `confusion_matrix.png` | Confusion matrix |
| `roc_curve.png` | ROC curve per class |
| `classification_report.txt` | Precision/recall/F1 per class |
| `training_log.json` | Training summary |
