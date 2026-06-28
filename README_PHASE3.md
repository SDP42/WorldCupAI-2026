# 🧪 WorldCupAI — Phase 3: Advanced Feature Engineering

> Transforms the validated master dataset into a high-quality, leakage-safe feature store for machine learning.

---

## 1. Overview

Phase 3 implements the feature engineering pipeline. It computes multi-dimensional features representing team strength, recent form, head-to-head history, tournament experience, and match context.

All features are calculated using a chronological shift to guarantee 100% temporal safety and prevent leakage.

---

## 2. Feature Families

We engineered features across the following families:
1. **Team Strength**: Elo ratings, FIFA ranks, differences, and ratios.
2. **Recent Form**: Rolling win/draw/loss rates, goals, and clean sheets over 5- and 10-match windows.
3. **Head-to-Head**: Cumulative wins, draws, and goal differences between the two teams.
4. **Attack/Defence**: Relative attacking and defensive ratings compared to global baselines.
5. **Tournament Experience**: Historical World Cup titles and participations.
6. **Context**: Neutral venue flags, rest days, and rest differences.

---

## 3. Key Outputs & Reports

The pipeline automatically generates the following files:
- **Feature Store**:
  - `processed/feature_store.parquet` (Primary binary storage)
  - `processed/feature_store.csv` (Compatibility storage)
- **Documentation & Reports**:
  - `FEATURE_DICTIONARY.md` — Detailed registry of every feature.
  - `README_FEATURE_VALIDATION.md` — Variance, missing values, and outliers.
  - `FEATURE_SELECTION_REPORT.md` — Correlation, mutual information, and tree importance.
  - `FEATURE_LEAKAGE_REPORT.md` — Leakage prevention validation.
  - `PHASE_3_APPROVAL_REPORT.md` — Final Phase 3 sign-off.
