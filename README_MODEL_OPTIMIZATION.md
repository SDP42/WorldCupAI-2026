# ⚙️ WorldCupAI — Model Optimization Guide (Phase 5)

> Generated: 2026-06-28 17:05:20

## Cross-Validation Strategy

All searches use `TimeSeriesSplit(n_splits=5)` to maintain temporal integrity.
No future data leaks into training folds.

## Search Strategies

- **GridSearchCV**: Used for models with small parameter grids (Logistic Regression).
- **RandomizedSearchCV**: Used for large grids — samples `n_iter` random combinations.
- **Early Stopping**: XGBoost, LightGBM, and CatBoost are re-fit post-search using
  the validation set as an eval set with `early_stopping_rounds=20`.

## Per-Model Search Spaces

### Logistic Regression
- **Strategy**: GRID
- **Iterations**: Exhaustive
- **Scoring**: neg_log_loss (via TimeSeriesSplit CV, 5 folds)
```json
{
      "C": [
            0.01,
            0.1,
            1.0,
            10.0
      ],
      "solver": [
            "lbfgs",
            "saga"
      ],
      "max_iter": [
            2000
      ]
}
```
### Random Forest
- **Strategy**: RANDOM
- **Iterations**: 20
- **Scoring**: neg_log_loss (via TimeSeriesSplit CV, 5 folds)
```json
{
      "n_estimators": [
            100,
            200,
            300
      ],
      "max_depth": [
            6,
            8,
            10,
            12,
            null
      ],
      "min_samples_split": [
            2,
            5,
            10
      ],
      "min_samples_leaf": [
            1,
            2,
            4
      ],
      "max_features": [
            "sqrt",
            "log2"
      ]
}
```
### Extra Trees
- **Strategy**: RANDOM
- **Iterations**: 20
- **Scoring**: neg_log_loss (via TimeSeriesSplit CV, 5 folds)
```json
{
      "n_estimators": [
            100,
            200,
            300
      ],
      "max_depth": [
            6,
            8,
            10,
            12,
            null
      ],
      "min_samples_split": [
            2,
            5
      ],
      "max_features": [
            "sqrt",
            "log2"
      ]
}
```
### Gradient Boosting
- **Strategy**: RANDOM
- **Iterations**: 20
- **Scoring**: neg_log_loss (via TimeSeriesSplit CV, 5 folds)
```json
{
      "n_estimators": [
            50,
            100,
            200
      ],
      "max_depth": [
            3,
            4,
            5,
            6
      ],
      "learning_rate": [
            0.05,
            0.1,
            0.2
      ],
      "subsample": [
            0.7,
            0.8,
            1.0
      ],
      "min_samples_split": [
            2,
            5
      ]
}
```
### XGBoost
- **Strategy**: RANDOM (+ Early Stopping on val set)
- **Iterations**: 20
- **Scoring**: neg_log_loss (via TimeSeriesSplit CV, 5 folds)
```json
{
      "n_estimators": [
            100,
            200,
            300,
            500
      ],
      "max_depth": [
            3,
            4,
            5,
            6
      ],
      "learning_rate": [
            0.01,
            0.05,
            0.1,
            0.2
      ],
      "subsample": [
            0.7,
            0.8,
            1.0
      ],
      "colsample_bytree": [
            0.6,
            0.8,
            1.0
      ],
      "gamma": [
            0,
            0.1,
            0.3
      ],
      "min_child_weight": [
            1,
            3,
            5
      ]
}
```


## Reproducibility

All searches use `random_state=42`. Results are deterministic.
