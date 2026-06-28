"""WorldCupAI — Phase 7.1: Constrained & Multi-Objective Ensemble Weight Optimization.

Uses scipy.optimize.minimize to find the optimal set of soft voting
weights that minimize target objectives on validation predictions.
Supports Log Loss optimization, Multi-objective optimization, and constraints
on candidates (Top-K, min/max weights).
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score
from typing import Dict, List, Tuple, Optional, Any

from src.utils.logger import setup_logger
from src.models.calibrator import compute_ece

logger = setup_logger("ensemble_optimizer")


class EnsembleWeightOptimizer:
    """Optimizes soft voting weights on validation set predictions.

    Args:
        model_names: List of model names to include in soft voting.
    """

    def __init__(self, model_names: List[str]):
        self.model_names = model_names

    def _loss_objective(
        self,
        weights:          np.ndarray,
        val_predictions:  Dict[str, Dict[str, np.ndarray]],
        y_val:            np.ndarray,
    ) -> float:
        """Computes multiclass Log Loss of the weighted average probabilities."""
        w_sum = np.sum(weights)
        if w_sum > 0:
            w_norm = weights / w_sum
        else:
            w_norm = weights

        n_samples = len(y_val)
        weighted_probs = np.zeros((n_samples, 3))

        for idx, mname in enumerate(self.model_names):
            weighted_probs += w_norm[idx] * val_predictions[mname]["y_prob"]

        weighted_probs = np.clip(weighted_probs, 1e-15, 1.0 - 1e-15)
        return log_loss(y_val, weighted_probs, labels=[0, 1, 2])

    def _multi_objective_objective(
        self,
        weights:          np.ndarray,
        val_predictions:  Dict[str, Dict[str, np.ndarray]],
        y_val:            np.ndarray,
        one_hot_y:        np.ndarray,
        corr_matrix:      np.ndarray,
        w_ll:             float = 0.4,
        w_auc:            float = 0.2,
        w_brier:          float = 0.2,
        w_ece:            float = 0.1,
        w_div:            float = 0.1,
    ) -> float:
        """Joint objective combining Log Loss, ROC-AUC, Brier Score, ECE, and Diversity."""
        w_sum = np.sum(weights)
        if w_sum > 0:
            w_norm = weights / w_sum
        else:
            w_norm = weights

        n_samples = len(y_val)
        weighted_probs = np.zeros((n_samples, 3))

        for idx, mname in enumerate(self.model_names):
            weighted_probs += w_norm[idx] * val_predictions[mname]["y_prob"]

        weighted_probs = np.clip(weighted_probs, 1e-15, 1.0 - 1e-15)

        # 1. Log Loss
        ll = log_loss(y_val, weighted_probs, labels=[0, 1, 2])

        # 2. ROC-AUC (One-vs-Rest macro)
        try:
            auc_val = roc_auc_score(y_val, weighted_probs, multi_class="ovr", average="macro", labels=[0, 1, 2])
        except Exception:
            auc_val = 0.5
        auc_loss = 1.0 - auc_val

        # 3. Brier Score (multiclass mean squared error)
        brier = np.mean(np.sum((weighted_probs - one_hot_y) ** 2, axis=1))

        # 4. ECE
        ece_val = compute_ece(y_val, weighted_probs)

        # 5. Diversity Penalty (sum of weighted pairwise correlations)
        # Minimize the weighted correlation: w^T * Corr * w
        div_loss = float(np.dot(w_norm, np.dot(corr_matrix, w_norm)))

        # Combine
        total_loss = (
            w_ll * ll +
            w_auc * auc_loss +
            w_brier * brier +
            w_ece * ece_val +
            w_div * div_loss
        )
        return float(total_loss)

    def _get_bounds_and_constraints(
        self,
        val_predictions:  Dict[str, Dict[str, np.ndarray]],
        y_val:            np.ndarray,
        top_k:            Optional[int] = None,
        min_weight:       Optional[float] = None,
        max_weight:       Optional[float] = None,
    ) -> Tuple[List[Tuple[float, float]], List[Dict[str, Any]], List[str]]:
        """Computes optimal bounds for each model name, zeroing out non-top-K models."""
        n_models = len(self.model_names)
        
        # Determine top-K active models based on individual validation log loss
        active_models = set(self.model_names)
        if top_k is not None and top_k < n_models:
            losses = []
            for name in self.model_names:
                probs = val_predictions[name]["y_prob"]
                probs_clipped = np.clip(probs, 1e-15, 1.0 - 1e-15)
                loss_val = log_loss(y_val, probs_clipped, labels=[0, 1, 2])
                losses.append((name, loss_val))
            # Sort by ascending loss
            losses.sort(key=lambda x: x[1])
            active_models = {name for name, _ in losses[:top_k]}
            logger.info(f"Top-{top_k} active models selected for optimization: {active_models}")

        # Compute valid min and max weights
        k_active = len(active_models)
        
        actual_min = 0.0
        if min_weight is not None:
            # If min_weight * k_active > 1.0, it is impossible. Cap it.
            if min_weight * k_active > 1.0:
                actual_min = 1.0 / k_active
                logger.warning(f"min_weight {min_weight} is infeasible for {k_active} active models. Capped to {actual_min:.4f}")
            else:
                actual_min = min_weight

        actual_max = 1.0
        if max_weight is not None:
            if max_weight * k_active < 1.0:
                actual_max = 1.0 / k_active
                logger.warning(f"max_weight {max_weight} is infeasible for {k_active} active models. Raised to {actual_max:.4f}")
            else:
                actual_max = max_weight

        # Build bounds
        bounds = []
        for name in self.model_names:
            if name in active_models:
                bounds.append((actual_min, actual_max))
            else:
                # Inactive models are forced to 0.0
                bounds.append((0.0, 0.0))

        # Constraint: sum of weights = 1.0
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        return bounds, constraints, list(active_models)

    def optimize(
        self,
        val_predictions: Dict[str, Dict[str, np.ndarray]],
        y_val:           np.ndarray,
        top_k:            Optional[int] = None,
        min_weight:       Optional[float] = None,
        max_weight:       Optional[float] = None,
    ) -> Tuple[Dict[str, float], float]:
        """Finds optimal non-negative weights summing to 1.0 using Log Loss objective.

        Returns:
            best_weights: Dict of model_name -> float weight.
            best_loss:    Optimal Log Loss value.
        """
        n_models = len(self.model_names)
        bounds, constraints, active_models = self._get_bounds_and_constraints(
            val_predictions, y_val, top_k, min_weight, max_weight
        )

        # Initialize weights: split uniform among active models
        init_weights = np.zeros(n_models)
        for idx, name in enumerate(self.model_names):
            if name in active_models:
                init_weights[idx] = 1.0 / len(active_models)

        logger.info(f"Optimizing soft voting weights for {n_models} models on validation split (Log Loss objective)...")
        
        res = minimize(
            fun=self._loss_objective,
            x0=init_weights,
            args=(val_predictions, y_val),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-6},
        )

        if not res.success:
            logger.warning(f"Optimization warning: minimize terminated with message: {res.message}")

        # Normalize and clean output
        best_w = np.clip(res.x, 0.0, 1.0)
        w_sum = np.sum(best_w)
        if w_sum > 0:
            best_w = best_w / w_sum
        else:
            best_w = init_weights

        best_weights = {self.model_names[i]: float(best_w[i]) for i in range(n_models)}
        best_loss    = float(res.fun)

        logger.info("Ensemble weight optimization complete (Log Loss).")
        for name, w in best_weights.items():
            logger.info(f"  - {name}: {w:.4f}")
        logger.info(f"Best Validation Log Loss: {best_loss:.5f}")

        return best_weights, best_loss

    def optimize_multi_objective(
        self,
        val_predictions: Dict[str, Dict[str, np.ndarray]],
        y_val:           np.ndarray,
        top_k:            Optional[int] = None,
        min_weight:       Optional[float] = None,
        max_weight:       Optional[float] = None,
        w_ll:             float = 0.4,
        w_auc:            float = 0.2,
        w_brier:          float = 0.2,
        w_ece:            float = 0.1,
        w_div:            float = 0.1,
    ) -> Tuple[Dict[str, float], float]:
        """Optimizes weights using joint Multi-objective loss.

        Returns:
            best_weights: Dict of model_name -> float weight.
            best_loss:    Optimal Multi-objective loss value.
        """
        n_models = len(self.model_names)
        bounds, constraints, active_models = self._get_bounds_and_constraints(
            val_predictions, y_val, top_k, min_weight, max_weight
        )

        # Initialize weights
        init_weights = np.zeros(n_models)
        for idx, name in enumerate(self.model_names):
            if name in active_models:
                init_weights[idx] = 1.0 / len(active_models)

        # 1. Precompute One-Hot representation of y_val
        one_hot_y = np.eye(3)[y_val]

        # 2. Compute Pearson Correlation Matrix on the validation Home Win probabilities
        # Focus on Home Win probability (index 2)
        probs_list = []
        for name in self.model_names:
            probs_list.append(val_predictions[name]["y_prob"][:, 2])
        probs_arr = np.array(probs_list)  # shape (n_models, n_samples)
        corr_matrix = np.corrcoef(probs_arr)  # shape (n_models, n_models)
        # Handle potential NaNs in correlation matrix
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        logger.info("Optimizing soft voting weights on validation split (Multi-Objective)...")

        res = minimize(
            fun=self._multi_objective_objective,
            x0=init_weights,
            args=(val_predictions, y_val, one_hot_y, corr_matrix, w_ll, w_auc, w_brier, w_ece, w_div),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-6},
        )

        if not res.success:
            logger.warning(f"Multi-objective optimization warning: {res.message}")

        best_w = np.clip(res.x, 0.0, 1.0)
        w_sum = np.sum(best_w)
        if w_sum > 0:
            best_w = best_w / w_sum
        else:
            best_w = init_weights

        best_weights = {self.model_names[i]: float(best_w[i]) for i in range(n_models)}
        best_loss    = float(res.fun)

        logger.info("Multi-objective weight optimization complete.")
        for name, w in best_weights.items():
            logger.info(f"  - {name}: {w:.4f}")
        logger.info(f"Best Multi-Objective Loss: {best_loss:.5f}")

        return best_weights, best_loss
