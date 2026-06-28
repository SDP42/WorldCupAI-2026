# 📦 WorldCupAI — Ensemble Architecture Reference

> Generated: 2026-06-28 19:52:08

## Package Structure
```
src/ensemble/
  __init__.py
  diversity.py     ModelDiversityAnalyzer (Kappa, disagree, double-fault, KL)
  voting.py        Hard, Soft, Weighted Soft Classifiers
  stacking.py      Stacking, Blending Classifiers
  optimizer.py     EnsembleWeightOptimizer (SLSQP: LogLoss vs Multi-Objective, Constrained)
```

## Key Optimization Strategies
- **Log Loss minimization**: Minimizes multi-class cross-entropy on validation predictions.
- **Multi-Objective optimization**: Jointly minimizes Log Loss, ECE, Brier score, and rewards model diversity.
- **Top-K and Weight Constraints**: Restricts candidate members to the strongest models with configurable minimum/maximum weights.
