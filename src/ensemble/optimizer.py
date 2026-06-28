"""WorldCupAI — Phase 7: Constrained Ensemble Weight Optimization.

Uses scipy.optimize.minimize to find the optimal set of soft voting
weights that minimize multi-class Log Loss on validation predictions.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from typing import Dict, List, Tuple

from src.utils.logger import setup_logger

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
        # Normalize weights to sum to 1.0 (scipy optimizer can make tiny excursions)
        w_sum = np.sum(weights)
        if w_sum > 0:
            w_norm = weights / w_sum
        else:
            w_norm = weights

        n_samples = len(y_val)
        weighted_probs = np.zeros((n_samples, 3))

        for idx, mname in enumerate(self.model_names):
            weighted_probs += w_norm[idx] * val_predictions[mname]["y_prob"]

        # Handle edge probabilities to prevent log_loss computation errors
        weighted_probs = np.clip(weighted_probs, 1e-15, 1.0 - 1e-15)

        # Multiclass log loss
        return log_loss(y_val, weighted_probs, labels=[0, 1, 2])

    def optimize(
        self,
        val_predictions: Dict[str, Dict[str, np.ndarray]],
        y_val:           np.ndarray,
    ) -> Tuple[Dict[str, float], float]:
        """Finds optimal non-negative weights summing to 1.0.

        Returns:
            best_weights: Dict of model_name -> float weight.
            best_loss:    Optimal Log Loss value.
        """
        n_models = len(self.model_names)
        init_weights = np.ones(n_models) / n_models

        # Bounds: weights in range [0, 1]
        bounds = [(0.0, 1.0) for _ in range(n_models)]

        # Constraint: sum of weights = 1.0
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        logger.info(f"Optimizing soft voting weights for {n_models} models on validation split...")
        
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
        best_w = best_w / np.sum(best_w)

        # Convert to dictionary
        best_weights = {self.model_names[i]: float(best_w[i]) for i in range(n_models)}
        best_loss    = float(res.fun)

        logger.info("Ensemble weight optimization complete.")
        for name, w in best_weights.items():
            logger.info(f"  - {name}: {w:.4f}")
        logger.info(f"Best Validation Log Loss: {best_loss:.5f}")

        return best_weights, best_loss
