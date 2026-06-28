# 🛡️ WorldCupAI — Time-Aware Validation Strategy

This document details the validation strategy designed to prevent temporal leakage and evaluate predictive capability.

## 1. Split Strategy

Because football matches are sequential events, standard K-Fold cross-validation leaks future information into past predictions (e.g. training on 2022 to predict 2014 matches). To prevent this, we implement a **Time-Aware Chronological Split**:

```
[================= Training (1970 - 2018) =================] [== Val (2019-2022) ==] [== Test (2023+) ==]
```

- **Training Set (<= 2018)**: 13,298 matches representing the historical baseline.
- **Validation Set (2019 - 2022)**: 3,581 matches used for model selection and calibration. Includes the 2022 FIFA World Cup.
- **Test Set (>= 2023)**: 3,645 matches reserved for final evaluation.

---

## 2. Chronological Integrity Check

- **No Future Leakage**: Verified. Training features only utilize match results that occurred strictly before the match date.
- **No Tournament Leakage**: Verified. Experience statistics (e.g. titles before) only aggregate tournaments completed prior to the match year.
- **ASOF Ranking Join**: FIFA ranks are joined backward chronologically, preventing lookup of future ranks.
