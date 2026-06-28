# 🤖 WorldCupAI — Phase 4: Machine Learning Benchmark Framework

> Establishes a modular benchmarking framework for baseline models.

---

## 1. Overview
Phase 4 builds a standardized, reproducible, and time-aware evaluation framework. We train 12 baseline classifiers on the feature store and evaluate their performance.

---

## 2. Pipeline Execution
To train all baseline models and generate metrics, run:

```bash
python3 train_baselines.py
```

All trained models, confusion matrices, and plots are stored under `models/`.
