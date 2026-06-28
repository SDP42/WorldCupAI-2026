# 📊 WorldCupAI — Statistical Significance Analysis (Phase 7.1)

> Generated: 2026-06-28 19:52:07

This report evaluates whether the performance differences between the best single ML model (XGBoost) and the various ensemble methods are statistically significant.

---

## 1. McNemar Test (Pairwise Prediction Agreement)

McNemar's test assesses if the mismatch rate between two classifiers is significantly asymmetric. The null hypothesis ($H_0$) is that the two models have equal predictive power. A $p$-value $< 0.05$ rejects the null hypothesis.

| Comparison | $\chi^2$ Statistic | $p$-value | Statistically Significant? (at $lpha=0.05$) |
|---|---|---|---|
| Weighted Soft Voting vs XGBoost | 2.4407 | 0.1182 | No |
| Stacking vs XGBoost | 0.0000 | 1 | No |
| Blending vs XGBoost | 0.3049 | 0.5808 | No |

---

## 2. Bootstrap Confidence Intervals (95% CI)

Confidence intervals are calculated using $B=1000$ bootstrap resamples with replacement on the test set (2023+).

### Accuracy Confidence Intervals

| Model / Ensemble | Point Estimate | 95% Confidence Interval |
|---|---|---|
| XGBoost (Best Single ML) | 0.6187 | [0.6025, 0.6348] |
| Weighted Soft Voting | 0.6151 | [0.5986, 0.6305] |
| Stacking | 0.6184 | [0.6022, 0.6332] |
| Blending | 0.6170 | [0.6008, 0.6321] |

### ROC-AUC Confidence Intervals

| Model / Ensemble | Point Estimate | 95% Confidence Interval |
|---|---|---|
| XGBoost (Best Single ML) | 0.7549 | [0.7430, 0.7665] |
| Weighted Soft Voting | 0.7547 | [0.7429, 0.7661] |
| Stacking | 0.7519 | [0.7396, 0.7635] |
| Blending | 0.7542 | [0.7423, 0.7660] |

---

## 3. Conclusions and Findings

- **Statistical Significance**: Based on McNemar's test, the difference in predictions between the ensemble strategies and XGBoost is **not statistically significant** at the 95% confidence level.
- **Overlapping CIs**: The bootstrap confidence intervals for Accuracy and ROC-AUC exhibit substantial overlap, indicating that while ensembles can stabilize calibration and log loss, their raw accuracy is competitive but statistically comparable to the fine-tuned XGBoost baseline on this test set size.
