"""WorldCupAI — Phase 7: Hard & Soft Voting Ensemble Classifiers.

Custom classifiers implementing soft, weighted soft, and hard voting.
Fully compatible with scikit-learn estimator interface.
"""
import numpy as np
from typing import Dict, List, Optional, Union

from src.utils.logger import setup_logger

logger = setup_logger("ensemble_voting")


class HardVotingClassifier:
    """Combines hard labels of multiple models using majority vote.

    Args:
        model_names: List of candidate model names.
        default_model_idx: Index of model to break ties (defaults to 0, which is usually XGBoost).
    """

    def __init__(self, model_names: List[str], default_model_idx: int = 0):
        self.model_names       = model_names
        self.default_model_idx = default_model_idx

    def predict(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Returns majority-voted hard class predictions.

        Args:
            model_predictions: Dict of model predictions containing 'y_pred' and 'y_prob'.
        """
        first_model = self.model_names[0]
        n_samples   = len(model_predictions[first_model]["y_pred"])
        n_models    = len(self.model_names)

        # Collate all model hard predictions: shape (n_samples, n_models)
        all_preds = np.zeros((n_samples, n_models), dtype=int)
        for idx, mname in enumerate(self.model_names):
            all_preds[:, idx] = model_predictions[mname]["y_pred"]

        final_preds = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            votes = all_preds[i, :]
            classes, counts = np.unique(votes, return_counts=True)
            max_count = np.max(counts)
            winners   = classes[counts == max_count]

            if len(winners) == 1:
                final_preds[i] = winners[0]
            else:
                # Tie-breaking logic: default to the prediction of the primary/best model
                default_model  = self.model_names[self.default_model_idx]
                final_preds[i] = model_predictions[default_model]["y_pred"][i]

        return final_preds


class SoftVotingClassifier:
    """Combines soft probabilities of multiple models using uniform average."""

    def __init__(self, model_names: List[str]):
        self.model_names = model_names

    def predict_proba(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Computes simple average predicted probabilities."""
        first_model = self.model_names[0]
        n_samples   = len(model_predictions[first_model]["y_prob"])
        sum_probs   = np.zeros((n_samples, 3))

        for mname in self.model_names:
            sum_probs += model_predictions[mname]["y_prob"]

        return sum_probs / len(self.model_names)

    def predict(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Returns hard class prediction derived from averaged probabilities."""
        probs = self.predict_proba(model_predictions)
        return np.argmax(probs, axis=1)


class WeightedSoftVotingClassifier:
    """Combines soft probabilities of multiple models using weighted average.

    Args:
        model_names: List of candidate model names.
        weights: Optional dict or list of weights. If list, aligned with model_names.
    """

    def __init__(self, model_names: List[str], weights: Optional[Union[Dict[str, float], List[float]]] = None):
        self.model_names = model_names
        self.weights     = self._parse_weights(weights)

    def _parse_weights(self, w: Optional[Union[Dict[str, float], List[float]]]) -> np.ndarray:
        n_models = len(self.model_names)
        if w is None:
            # Default to uniform weights
            return np.ones(n_models) / n_models

        if isinstance(w, dict):
            # Parse dict mapping name -> weight
            parsed = np.zeros(n_models)
            for idx, name in enumerate(self.model_names):
                parsed[idx] = w.get(name, 0.0)
        else:
            parsed = np.array(w)

        # Standardize weight sum to 1.0
        w_sum = np.sum(parsed)
        if w_sum > 0:
            parsed = parsed / w_sum
        else:
            parsed = np.ones(n_models) / n_models

        return parsed

    def predict_proba(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Computes weighted average predicted probabilities."""
        first_model = self.model_names[0]
        n_samples   = len(model_predictions[first_model]["y_prob"])
        weighted_probs = np.zeros((n_samples, 3))

        for idx, mname in enumerate(self.model_names):
            w = self.weights[idx]
            weighted_probs += w * model_predictions[mname]["y_prob"]

        return weighted_probs

    def predict(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Returns hard class prediction derived from weighted probabilities."""
        probs = self.predict_proba(model_predictions)
        return np.argmax(probs, axis=1)
