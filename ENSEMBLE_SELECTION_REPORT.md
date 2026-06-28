# 🔬 WorldCupAI — Ensemble Selection Report (Phase 7.1)

> Generated: 2026-06-28 19:52:08

## 1. Candidate Analysis

| model               |   roc_auc |   accuracy |   log_loss |    brier |       ece | selected   | note                   |
|:--------------------|----------:|-----------:|-----------:|---------:|----------:|:-----------|:-----------------------|
| XGBoost             |  0.754938 |   0.618656 |   0.840765 | 0.49615  | 0.0128759 | True       | INCLUDED               |
| Gradient Boosting   |  0.752772 |   0.611797 |   0.846285 | 0.499167 | 0.0178968 | True       | INCLUDED               |
| Random Forest       |  0.750929 |   0.613992 |   0.852037 | 0.500689 | 0.0219414 | True       | INCLUDED               |
| ANN                 |  0.736361 |   0.6      |   0.869486 | 0.512965 | 0.0234029 | True       | INCLUDED               |
| Extra Trees         |  0.735017 |   0.605761 |   0.871868 | 0.513201 | 0.0225321 | True       | INCLUDED               |
| Logistic Regression |  0.727878 |   0.594513 |   0.890092 | 0.523979 | 0.0175639 | True       | INCLUDED               |
| LSTM                |  0.508235 |   0.464198 |   1.06688  | 0.643866 | 0.0410645 | False      | EXCLUDED: weak learner |

## 2. Diversity Analysis

### Complementary Pair Recommendations
- **Random Forest** & **LSTM**: Highly complementary. Prob Correlation = 0.039, Error Overlap = 0.518.
- **XGBoost** & **LSTM**: Highly complementary. Prob Correlation = 0.039, Error Overlap = 0.512.
- **Logistic Regression** & **LSTM**: Highly complementary. Prob Correlation = 0.040, Error Overlap = 0.577.
- **Gradient Boosting** & **LSTM**: Highly complementary. Prob Correlation = 0.040, Error Overlap = 0.507.

### Redundancy Warnings
- ⚠️ **Random Forest** and **Extra Trees** show extremely high probability correlation (0.961). Consider down-weighting one to prevent feature duplication.
- ⚠️ **Gradient Boosting** and **Random Forest** show extremely high probability correlation (0.974). Consider down-weighting one to prevent feature duplication.
- ⚠️ **XGBoost** and **Random Forest** show extremely high probability correlation (0.984). Consider down-weighting one to prevent feature duplication.
- ⚠️ **XGBoost** and **Gradient Boosting** show extremely high probability correlation (0.992). Consider down-weighting one to prevent feature duplication.

## 3. Optimized Weights

| Model | Weight |
|---|---|
| Gradient Boosting | 0.5140 |
| XGBoost | 0.4860 |
| Logistic Regression | 0.0000 |
| Extra Trees | 0.0000 |
| ANN | 0.0000 |
| Random Forest | 0.0000 |
| LSTM | 0.0000 |

**Validation Loss (optimized)**: `0.79350`

## 4. Ensemble Selection Result

| Criterion | Winner |
|---|---|
| Best Overall (Composite Score) | Weighted Soft Voting |
| Most Calibrated (Min ECE) | Weighted Soft Voting |
| Most Robust (Max ROC-AUC) | Weighted Soft Voting |
| Fastest (Min Inference Time) | Soft Voting |

