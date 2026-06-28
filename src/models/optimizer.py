"""WorldCupAI — Phase 5: Hyperparameter Optimizer

Wraps sklearn GridSearchCV and RandomizedSearchCV with time-aware
cross-validation (TimeSeriesSplit). Reuses Phase 4 preprocessing pipeline
unchanged. Logs every search and persists best_params.json and
training_log.json alongside the optimized model artifacts.
"""
import os
import json
import time
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.utils.logger import setup_logger

logger = setup_logger("optimizer")

# ─────────────────────────────────────────────────────────────────────────────
# Search space registry
# ─────────────────────────────────────────────────────────────────────────────
SEARCH_SPACES: Dict[str, Dict] = {
    "Logistic Regression": {
        "strategy": "grid",
        "params": {
            "C":        [0.01, 0.1, 1.0, 10.0],
            "solver":   ["lbfgs", "saga"],
            "max_iter": [2000],
        },
        "n_iter": None,   # grid — exhaustive
    },
    "Random Forest": {
        "strategy": "random",
        "params": {
            "n_estimators":     [100, 200, 300],
            "max_depth":        [6, 8, 10, 12, None],
            "min_samples_split":[2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features":     ["sqrt", "log2"],
        },
        "n_iter": 20,
    },
    "Extra Trees": {
        "strategy": "random",
        "params": {
            "n_estimators":     [100, 200, 300],
            "max_depth":        [6, 8, 10, 12, None],
            "min_samples_split":[2, 5],
            "max_features":     ["sqrt", "log2"],
        },
        "n_iter": 20,
    },
    "Gradient Boosting": {
        "strategy": "random",
        "params": {
            "n_estimators":      [50, 100, 200],
            "max_depth":         [3, 4, 5, 6],
            "learning_rate":     [0.05, 0.1, 0.2],
            "subsample":         [0.7, 0.8, 1.0],
            "min_samples_split": [2, 5],
        },
        "n_iter": 20,
    },
    "XGBoost": {
        "strategy": "random",
        "params": {
            "n_estimators":     [100, 200, 300, 500],
            "max_depth":        [3, 4, 5, 6],
            "learning_rate":    [0.01, 0.05, 0.1, 0.2],
            "subsample":        [0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma":            [0, 0.1, 0.3],
            "min_child_weight": [1, 3, 5],
        },
        "n_iter": 20,
    },
    "LightGBM": {
        "strategy": "random",
        "params": {
            "n_estimators":  [100, 200, 300, 500],
            "max_depth":     [3, 5, 7, -1],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves":    [31, 63, 127],
            "subsample":     [0.7, 0.8, 1.0],
            "reg_alpha":     [0.0, 0.1, 0.5],
            "reg_lambda":    [0.0, 0.1, 1.0],
        },
        "n_iter": 20,
    },
    "CatBoost": {
        "strategy": "random",
        "params": {
            "iterations":    [100, 200, 300],
            "depth":         [4, 5, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "l2_leaf_reg":   [1, 3, 5, 7],
        },
        "n_iter": 20,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer class
# ─────────────────────────────────────────────────────────────────────────────
class HyperparameterOptimizer:
    """Performs time-aware hyperparameter search for a given model.

    Args:
        model_name:     Canonical model name (must match SEARCH_SPACES key).
        model:          Instantiated (un-fitted) sklearn-compatible estimator.
        output_dir:     Directory to persist optimized model artifacts.
        cv_splits:      Number of TimeSeriesSplit folds (default 5).
        random_state:   Random seed for reproducibility.
        n_jobs:         Parallel jobs for CV (-1 = all cores).
    """

    def __init__(
        self,
        model_name: str,
        model: Any,
        output_dir: str,
        cv_splits: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.model_name = model_name
        self.model = model
        self.output_dir = output_dir
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.n_jobs = n_jobs
        os.makedirs(output_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Any, Dict]:
        """Runs hyperparameter search and returns (best_model, best_params).

        Uses validation set for early-stopping-aware models
        (XGBoost, LightGBM, CatBoost). For all others, uses
        TimeSeriesSplit CV over the training set.
        """
        if self.model_name not in SEARCH_SPACES:
            logger.warning(f"No search space for '{self.model_name}'. Returning base model.")
            return self.model, {}

        space = SEARCH_SPACES[self.model_name]
        strategy = space["strategy"]
        param_grid = space["params"]
        n_iter = space["n_iter"]

        logger.info(f"[{self.model_name}] Starting {strategy.upper()} search "
                    f"({n_iter or 'exhaustive'} iterations, {self.cv_splits}-fold TSS CV)")

        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        start = time.time()

        if strategy == "grid":
            searcher = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=tscv,
                scoring="neg_log_loss",
                n_jobs=self.n_jobs,
                refit=True,
                verbose=0,
                error_score="raise",
            )
        else:
            searcher = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=tscv,
                scoring="neg_log_loss",
                n_jobs=self.n_jobs,
                refit=True,
                verbose=0,
                random_state=self.random_state,
                error_score="raise",
            )

        # For gradient boosting libs that support early stopping, combine
        # train+val and rely on CV; early stopping is handled post-search.
        try:
            searcher.fit(X_train, y_train)
        except Exception as exc:
            logger.error(f"[{self.model_name}] Search failed: {exc}. Fitting base model as fallback.")
            try:
                self.model.fit(X_train, y_train)
            except Exception as fit_exc:
                logger.error(f"[{self.model_name}] Fallback fit also failed: {fit_exc}")
            return self.model, {}

        elapsed = time.time() - start
        best_model = searcher.best_estimator_
        best_params = searcher.best_params_
        best_score = -searcher.best_score_   # convert neg_log_loss → log_loss

        logger.info(f"[{self.model_name}] Search complete in {elapsed:.1f}s | "
                    f"Best CV log-loss: {best_score:.4f} | Params: {best_params}")

        # ── Apply early stopping on val set for boosting models ───────────────
        best_model, best_params = self._apply_early_stopping(
            best_model, best_params, X_train, y_train, X_val, y_val
        )

        # ── Save artifacts ────────────────────────────────────────────────────
        self._save_model(best_model)
        self._save_params(best_params, best_score, elapsed, searcher)

        return best_model, best_params

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _apply_early_stopping(
        self,
        model: Any,
        params: Dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Any, Dict]:
        """Re-fits boosting models with early stopping using the validation set."""
        name = self.model_name
        if name not in ("XGBoost", "LightGBM", "CatBoost"):
            return model, params

        logger.info(f"[{name}] Applying early stopping on validation set …")

        try:
            if name == "XGBoost":
                from xgboost import XGBClassifier
                m = XGBClassifier(
                    **{k: v for k, v in params.items()},
                    random_state=self.random_state,
                    eval_metric="mlogloss",
                    early_stopping_rounds=20,
                    n_jobs=self.n_jobs,
                )
                m.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                best_iter = m.best_iteration
                params["best_iteration"] = int(best_iter)
                logger.info(f"[XGBoost] Best iteration: {best_iter}")
                return m, params

            elif name == "LightGBM":
                from lightgbm import LGBMClassifier, early_stopping, log_evaluation
                m = LGBMClassifier(
                    **{k: v for k, v in params.items()},
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbose=-1,
                )
                m.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[early_stopping(20, verbose=False), log_evaluation(period=-1)],
                )
                best_iter = m.best_iteration_
                params["best_iteration"] = int(best_iter)
                logger.info(f"[LightGBM] Best iteration: {best_iter}")
                return m, params

            elif name == "CatBoost":
                from catboost import CatBoostClassifier, Pool
                m = CatBoostClassifier(
                    **{k: v for k, v in params.items()},
                    random_state=self.random_state,
                    verbose=0,
                    early_stopping_rounds=20,
                )
                m.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                )
                best_iter = m.best_iteration_
                params["best_iteration"] = int(best_iter)
                logger.info(f"[CatBoost] Best iteration: {best_iter}")
                return m, params

        except Exception as exc:
            logger.warning(f"[{name}] Early stopping failed ({exc}). Keeping CV best model.")

        return model, params

    def _save_model(self, model: Any):
        path = os.path.join(self.output_dir, "model.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"[{self.model_name}] Optimized model saved → {path}")

    def _save_params(
        self,
        params: Dict,
        best_cv_log_loss: float,
        elapsed: float,
        searcher: Any,
    ):
        # best_params.json
        params_path = os.path.join(self.output_dir, "best_params.json")
        with open(params_path, "w") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "best_params": {k: (v if not isinstance(v, np.generic) else v.item())
                                    for k, v in params.items()},
                    "best_cv_neg_log_loss": float(best_cv_log_loss),
                    "search_time_sec": float(elapsed),
                    "cv_folds": self.cv_splits,
                    "strategy": SEARCH_SPACES[self.model_name]["strategy"],
                },
                f,
                indent=4,
            )

        # training_log.json — stores all CV results
        try:
            cv_results = pd.DataFrame(searcher.cv_results_)
            log = cv_results[["params", "mean_test_score", "std_test_score", "rank_test_score"]].copy()
            log["mean_test_score"] = -log["mean_test_score"]  # neg_log_loss → log_loss
            log_path = os.path.join(self.output_dir, "training_log.json")
            log.to_json(log_path, orient="records", indent=2)
            logger.info(f"[{self.model_name}] Training log saved → {log_path}")
        except Exception as exc:
            logger.warning(f"[{self.model_name}] Could not save training log: {exc}")
