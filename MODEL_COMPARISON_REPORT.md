# 🤖 WorldCupAI — Per-Model Comparison Report

**Generated**: 2026-06-28 21:05:03

---

## Step 9: Individual Model Predictions

**Ensemble Weights**: XGBoost=0.486, Gradient Boosting=0.514 (all others ≈0)

### Brazil vs Japan

| Model | Home Win% | Draw% | Away Win% | Prediction |
|---|---|---|---|---|
| XGBoost | 69.5% | 24.7% | 5.7% | **Home Win** |
| Gradient Boosting | 72.8% | 23.2% | 3.9% | **Home Win** |
| Random Forest | 65.7% | 23.7% | 10.6% | **Home Win** |
| Extra Trees | 48.1% | 26.8% | 25.1% | **Home Win** |
| Logistic Regression | 64.2% | 24.7% | 11.1% | **Home Win** |
| **Ensemble (Weighted SVoting)** | **67.8%** | **26.4%** | **5.7%** | **Brazil** |

### France vs Sweden

| Model | Home Win% | Draw% | Away Win% | Prediction |
|---|---|---|---|---|
| XGBoost | 79.1% | 18.1% | 2.9% | **Home Win** |
| Gradient Boosting | 82.0% | 15.6% | 2.4% | **Home Win** |
| Random Forest | 75.6% | 19.7% | 4.6% | **Home Win** |
| Extra Trees | 62.2% | 28.5% | 9.3% | **Home Win** |
| Logistic Regression | 65.9% | 22.2% | 11.9% | **Home Win** |
| **Ensemble (Weighted SVoting)** | **80.1%** | **17.2%** | **2.7%** | **France** |

### Germany vs Paraguay

| Model | Home Win% | Draw% | Away Win% | Prediction |
|---|---|---|---|---|
| XGBoost | 77.5% | 19.8% | 2.7% | **Home Win** |
| Gradient Boosting | 76.9% | 20.5% | 2.6% | **Home Win** |
| Random Forest | 74.1% | 20.5% | 5.4% | **Home Win** |
| Extra Trees | 67.5% | 25.8% | 6.7% | **Home Win** |
| Logistic Regression | 66.6% | 21.3% | 12.1% | **Home Win** |
| **Ensemble (Weighted SVoting)** | **75.1%** | **22.1%** | **2.7%** | **Germany** |

### Argentina vs Cape Verde

| Model | Home Win% | Draw% | Away Win% | Prediction |
|---|---|---|---|---|
| XGBoost | 90.7% | 7.2% | 2.1% | **Home Win** |
| Gradient Boosting | 78.3% | 18.4% | 3.3% | **Home Win** |
| Random Forest | 84.0% | 9.7% | 6.3% | **Home Win** |
| Extra Trees | 80.5% | 15.7% | 3.8% | **Home Win** |
| Logistic Regression | 83.7% | 11.6% | 4.7% | **Home Win** |
| **Ensemble (Weighted SVoting)** | **85.9%** | **11.3%** | **2.8%** | **Argentina** |

### Belgium vs Senegal

| Model | Home Win% | Draw% | Away Win% | Prediction |
|---|---|---|---|---|
| XGBoost | 41.1% | 23.9% | 35.1% | **Home Win** |
| Gradient Boosting | 44.6% | 23.0% | 32.4% | **Home Win** |
| Random Forest | 41.8% | 25.5% | 32.7% | **Home Win** |
| Extra Trees | 40.1% | 19.7% | 40.2% | **Away Win** |
| Logistic Regression | 44.0% | 23.6% | 32.3% | **Home Win** |
| **Ensemble (Weighted SVoting)** | **44.2%** | **22.2%** | **33.5%** | **Belgium** |

### Spain vs Austria

| Model | Home Win% | Draw% | Away Win% | Prediction |
|---|---|---|---|---|
| XGBoost | 80.9% | 15.9% | 3.2% | **Home Win** |
| Gradient Boosting | 80.2% | 16.8% | 3.0% | **Home Win** |
| Random Forest | 70.5% | 23.4% | 6.1% | **Home Win** |
| Extra Trees | 54.8% | 35.4% | 9.8% | **Home Win** |
| Logistic Regression | 70.5% | 17.7% | 11.8% | **Home Win** |
| **Ensemble (Weighted SVoting)** | **78.9%** | **17.1%** | **4.1%** | **Spain** |

### Netherlands vs Morocco

| Model | Home Win% | Draw% | Away Win% | Prediction |
|---|---|---|---|---|
| XGBoost | 70.1% | 22.4% | 7.5% | **Home Win** |
| Gradient Boosting | 72.5% | 22.4% | 5.2% | **Home Win** |
| Random Forest | 63.0% | 26.0% | 11.0% | **Home Win** |
| Extra Trees | 44.3% | 26.1% | 29.5% | **Home Win** |
| Logistic Regression | 42.2% | 26.1% | 31.8% | **Home Win** |
| **Ensemble (Weighted SVoting)** | **68.3%** | **24.9%** | **6.8%** | **Netherlands** |

### Portugal vs Croatia

| Model | Home Win% | Draw% | Away Win% | Prediction |
|---|---|---|---|---|
| XGBoost | 62.8% | 28.4% | 8.7% | **Home Win** |
| Gradient Boosting | 63.1% | 30.9% | 6.0% | **Home Win** |
| Random Forest | 63.3% | 23.9% | 12.8% | **Home Win** |
| Extra Trees | 49.2% | 25.2% | 25.6% | **Home Win** |
| Logistic Regression | 45.2% | 26.2% | 28.7% | **Home Win** |
| **Ensemble (Weighted SVoting)** | **61.8%** | **30.5%** | **7.8%** | **Portugal** |

## Step 8: Feature Importances (XGBoost)

### XGBoost — Top Features

| Feature | Importance |
|---|---|
| `num__elo_ratio` | 0.1451 |
| `num__is_neutral` | 0.0817 |
| `num__elo_diff` | 0.0717 |
| `num__rank_diff` | 0.0551 |
| `num__rank_ratio` | 0.0486 |
| `num__h2h_gd` | 0.0302 |
| `num__is_friendly` | 0.0256 |
| `num__h2h_away_wins` | 0.0252 |
| `num__h2h_draws` | 0.0243 |
| `num__away_world_cup_titles_before` | 0.0222 |

### Gradient Boosting — Top Features

| Feature | Importance |
|---|---|
| `num__elo_ratio` | 0.2789 |
| `num__rank_ratio` | 0.1098 |
| `num__rank_diff` | 0.1041 |
| `num__elo_diff` | 0.0977 |
| `num__is_neutral` | 0.0573 |
| `num__h2h_gd` | 0.0485 |
| `num__h2h_meetings` | 0.0185 |
| `num__rest_difference` | 0.0183 |
| `num__h2h_away_wins` | 0.0166 |
| `num__away_form_avg_goals_scored_5` | 0.0156 |

