# 🛡️ Temporal Validation & Leakage Report

Generated on: 2026-06-28 16:22:31

## Summary

| Check | Status | Details |
|---|---|---|
| **Feature Leakage Check** | ✅ PASSED | Checked 15 features against 16 forbidden post-kickoff columns. |
| **Chronological Order Check** | ✅ PASSED | Verified match dates are in the past and chronologically sound. |

## Feature Leakage Details

No forbidden columns detected in the feature set. The feature space is leakage-safe.

## Forbidden Columns Registry
The following columns are classified as **Forbidden** for training features (they are only allowed as prediction targets/labels):
- `home_score`
- `away_score`
- `winner`
- `result`
- `home_team_win`
- `away_team_win`
- `draw`
- `score`
- `extra_time`
- `penalty_shootout`
- `goals_for`
- `goals_against`
- `goal_differential`
- `position`
- `points`
- `advanced`
