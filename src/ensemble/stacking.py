"""WorldCupAI — Phase 7: Stacking & Blending Classifiers.

Implements meta-learning ensembles (Stacking and Blending)
using scikit-learn estimators as meta-learners.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from typing import Dict, List, Optional, Any

from src.utils.logger import setup_logger

logger = setup_logger("ensemble_stacking")


class StackingClassifier:
    """Stacking classifier that trains a meta-model on base predictions.

    Args:
        model_names:  List of model names to use as base classifiers.
        meta_learner: Instantiated sklearn classifier (defaults to LogisticRegression).
    """

    def __init__(self, model_names: List[str], meta_learner: Optional[Any] = None):
        self.model_names  = model_names
        self.meta_learner = meta_learner or LogisticRegression(C=0.1, random_state=42, max_iter=1000)
        self.is_fitted    = False

    def _build_features(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Concats probability predictions of all base models.

        Args:
            model_predictions: Dict of model predictions.

        Returns:
            X_meta: Array of shape (N, len(model_names) * 3)
        """
        feats = []
        for name in self.model_names:
            probs = model_predictions[name]["y_prob"]  # (N, 3)
            feats.append(probs)

        # Concatenate horizontally
        return np.hstack(feats)

    def fit(self, val_predictions: Dict[str, Dict[str, np.ndarray]], y_val: np.ndarray):
        """Fits the meta-learner on validation set predictions.

        Args:
            val_predictions: Predictions dictionary on the validation split.
            y_val:           Validation set true targets.
        """
        X_val_meta = self._build_features(val_predictions)
        logger.info(f"Stacking: fitting meta-learner on feature shape {X_val_meta.shape}")

        self.meta_learner.fit(X_val_meta, y_val)
        self.is_fitted = True

    def predict_proba(self, test_predictions: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Predicts probabilities using the fitted meta-learner."""
        if not self.is_fitted:
            raise ValueError("StackingClassifier has not been fitted yet!")

        X_test_meta = self._build_features(test_predictions)

        if hasattr(self.meta_learner, "predict_proba"):
            return self.meta_learner.predict_proba(X_test_meta)
        else:
            # Fallback for models without predict_proba (like RidgeClassifier)
            # Use decision function and apply softmax
            decision = self.meta_learner.decision_function(X_test_meta)
            if decision.ndim == 1:
                # Binary classification fallback
                p = 1 / (1 + np.exp(-decision))
                return np.vstack([1 - p, p]).T
            else:
                exp_dec = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                return exp_dec / np.sum(exp_dec, axis=1, keepdims=True)

    def predict(self, test_predictions: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Predicts class labels using the fitted meta-learner."""
        if not self.is_fitted:
            raise ValueError("StackingClassifier has not been fitted yet!")

        X_test_meta = self._build_features(test_predictions)
        return self.meta_learner.predict(X_test_meta)


class BlendingClassifier:
    """Blending classifier using Ridge Regression on validation probability outputs.

    Forces predictions to form valid probability distributions via Softmax scaling.
    """

    def __init__(self, model_names: List[str], alpha: float = 1.0):
        self.model_names  = model_names
        self.alpha        = alpha
        self.meta_learner = RidgeClassifier(alpha=alpha, random_state=42)
        self.is_fitted    = False

    def _build_features(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        feats = []
        for name in self.model_names:
            feats.append(model_predictions[name]["y_prob"])
        return np.hstack(feats)

    def fit(self, val_predictions: Dict[str, Dict[str, np.ndarray]], y_val: np.ndarray):
        X_val_meta = self._build_features(val_predictions)
        logger.info(f"Blending: fitting RidgeClassifier on validation predictions (alpha={self.alpha})")
        self.meta_learner.fit(X_val_meta, y_val)
        self.is_fitted = True

    def predict_proba(self, test_predictions: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("BlendingClassifier is not fitted.")

        X_test_meta = self._build_features(test_predictions)
        decision    = self.meta_learner.decision_function(X_test_meta)

        # Softmax scaling of decision outputs
        exp_dec = np.exp(decision - np.max(decision, axis=1, keepdims=True))
        return exp_dec / np.sum(exp_dec, axis=1, keepdims=True)

    def predict(self, test_predictions: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("BlendingClassifier is not fitted.")

        X_test_meta = self._build_features(test_predictions)
        return self.meta_learner.predict(X_test_meta)
