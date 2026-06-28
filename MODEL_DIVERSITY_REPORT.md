# 🧬 WorldCupAI — Advanced Model Diversity Report (Phase 7.1)

> Generated: 2026-06-28 19:52:08

This report evaluates advanced diversity statistics between the base models.

---

## 1. Disagreement Rates
Proportion of matches where two models predict different hard class labels. Higher rates indicate higher prediction diversity.

|                     |   XGBoost |   Gradient Boosting |   Random Forest |   Extra Trees |   Logistic Regression |      ANN |     LSTM |
|:--------------------|----------:|--------------------:|----------------:|--------------:|----------------------:|---------:|---------:|
| XGBoost             | 0         |           0.0441701 |       0.0548697 |      0.111934 |              0.163237 | 0.127298 | 0.387106 |
| Gradient Boosting   | 0.0441701 |           0         |       0.0721536 |      0.121262 |              0.172565 | 0.138272 | 0.398903 |
| Random Forest       | 0.0548697 |           0.0721536 |       0         |      0.106996 |              0.155556 | 0.127572 | 0.383265 |
| Extra Trees         | 0.111934  |           0.121262  |       0.106996  |      0        |              0.159671 | 0.119067 | 0.375309 |
| Logistic Regression | 0.163237  |           0.172565  |       0.155556  |      0.159671 |              0        | 0.161317 | 0.335528 |
| ANN                 | 0.127298  |           0.138272  |       0.127572  |      0.119067 |              0.161317 | 0        | 0.398354 |
| LSTM                | 0.387106  |           0.398903  |       0.383265  |      0.375309 |              0.335528 | 0.398354 | 0        |

---

## 2. Q-Statistics
Measures the association of correct/incorrect predictions between two models. Varying between -1 and +1:
- A value of $0$ indicates independent classifiers.
- A value of $+1$ indicates perfect agreement.
- Negative values show classifiers tend to commit errors on different samples.

|                     |   XGBoost |   Gradient Boosting |   Random Forest |   Extra Trees |   Logistic Regression |      ANN |     LSTM |
|:--------------------|----------:|--------------------:|----------------:|--------------:|----------------------:|---------:|---------:|
| XGBoost             |  1        |            0.997694 |        0.996089 |      0.98196  |              0.969433 | 0.977495 | 0.767809 |
| Gradient Boosting   |  0.997694 |            1        |        0.993186 |      0.979711 |              0.965727 | 0.972559 | 0.74729  |
| Random Forest       |  0.996089 |            0.993186 |        1        |      0.983866 |              0.972239 | 0.977607 | 0.772428 |
| Extra Trees         |  0.98196  |            0.979711 |        0.983866 |      1        |              0.970531 | 0.980484 | 0.786705 |
| Logistic Regression |  0.969433 |            0.965727 |        0.972239 |      0.970531 |              1        | 0.967963 | 0.844297 |
| ANN                 |  0.977495 |            0.972559 |        0.977607 |      0.980484 |              0.967963 | 1        | 0.739794 |
| LSTM                |  0.767809 |            0.74729  |        0.772428 |      0.786705 |              0.844297 | 0.739794 | 1        |

---

## 3. Double-Fault Measures (DF)
The proportion of cases where both models predict incorrectly. A lower double-fault rate indicates better complementary performance.

|                     |   XGBoost |   Gradient Boosting |   Random Forest |   Extra Trees |   Logistic Regression |      ANN |     LSTM |
|:--------------------|----------:|--------------------:|----------------:|--------------:|----------------------:|---------:|---------:|
| XGBoost             |  0.381344 |            0.36845  |        0.362963 |      0.34513  |              0.338272 | 0.342936 | 0.309739 |
| Gradient Boosting   |  0.36845  |            0.388203 |        0.360219 |      0.346502 |              0.339369 | 0.342387 | 0.310288 |
| Random Forest       |  0.362963 |            0.360219 |        0.386008 |      0.349794 |              0.34321  | 0.345679 | 0.313855 |
| Extra Trees         |  0.34513  |            0.346502 |        0.349794 |      0.394513 |              0.346502 | 0.353086 | 0.322634 |
| Logistic Regression |  0.338272 |            0.339369 |        0.34321  |      0.346502 |              0.405487 | 0.347325 | 0.343759 |
| ANN                 |  0.342936 |            0.342387 |        0.345679 |      0.353086 |              0.347325 | 0.4      | 0.316598 |
| LSTM                |  0.309739 |            0.310288 |        0.313855 |      0.322634 |              0.343759 | 0.316598 | 0.533882 |

---

## 4. Pairwise KL-Divergence
The symmetric Kullback-Leibler divergence between predicted probability distributions:
$$KL_{sym}(P, Q) = \frac{1}{2} (KL(P || Q) + KL(Q || P))$$
Higher divergence represents wider disagreement in soft probabilities.

|                     |    XGBoost |   Gradient Boosting |   Random Forest |   Extra Trees |   Logistic Regression |       ANN |     LSTM |
|:--------------------|-----------:|--------------------:|----------------:|--------------:|----------------------:|----------:|---------:|
| XGBoost             | 0          |          0.00534209 |       0.0121254 |     0.0293912 |             0.0631799 | 0.0355053 | 0.253374 |
| Gradient Boosting   | 0.00534209 |          0          |       0.0201129 |     0.0370741 |             0.0764574 | 0.0429093 | 0.265714 |
| Random Forest       | 0.0121254  |          0.0201129  |       0         |     0.0189752 |             0.0416017 | 0.0313955 | 0.20653  |
| Extra Trees         | 0.0293912  |          0.0370741  |       0.0189752 |     0         |             0.0468638 | 0.0276159 | 0.192681 |
| Logistic Regression | 0.0631799  |          0.0764574  |       0.0416017 |     0.0468638 |             0         | 0.0540246 | 0.199921 |
| ANN                 | 0.0355053  |          0.0429093  |       0.0313955 |     0.0276159 |             0.0540246 | 0         | 0.223354 |
| LSTM                | 0.253374   |          0.265714   |       0.20653   |     0.192681  |             0.199921  | 0.223354  | 0        |

