"""WorldCupAI — Phase 7.1: Model Diversity & Agreement Analysis.

Measures pairwise agreement, Cohen's Kappa, soft probability correlations,
error overlap, disagreement rate, Q-statistic, double-fault measure, and KL-divergence
to ensure selected ensemble members are highly complementary.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from typing import Dict, List, Tuple

from src.utils.logger import setup_logger

logger = setup_logger("ensemble_diversity")


class ModelDiversityAnalyzer:
    """Analyzes prediction agreement, correlations, and error overlap between models.

    Args:
        model_predictions: Dict mapping model name to:
                           {
                             "y_prob": np.ndarray of shape (N, 3),
                             "y_pred": np.ndarray of shape (N,),
                             "y_true": np.ndarray of shape (N,)
                           }
    """

    def __init__(self, model_predictions: Dict[str, Dict[str, np.ndarray]]):
        self.model_preds = model_predictions
        self.models      = list(model_predictions.keys())
        self.n_models    = len(self.models)
        self.y_true      = next(iter(model_predictions.values()))["y_true"]

    def compute_probability_correlations(self) -> pd.DataFrame:
        """Computes pairwise Pearson correlation of predicted probabilities for the 'Home Win' class (class 2)."""
        probs = {}
        for name in self.models:
            # Focus on Home Win probabilities (standard reference class for soft alignment)
            probs[name] = self.model_preds[name]["y_prob"][:, 2]

        df = pd.DataFrame(probs)
        return df.corr()

    def compute_prediction_agreements(self) -> pd.DataFrame:
        """Computes pairwise hard prediction agreement (percentage of identical predictions)."""
        matrix = np.zeros((self.n_models, self.n_models))

        for i, m1 in enumerate(self.models):
            for j, m2 in enumerate(self.models):
                pred1 = self.model_preds[m1]["y_pred"]
                pred2 = self.model_preds[m2]["y_pred"]
                matrix[i, j] = np.mean(pred1 == pred2)

        return pd.DataFrame(matrix, index=self.models, columns=self.models)

    def compute_cohens_kappa(self) -> pd.DataFrame:
        """Computes pairwise Cohen's Kappa score between hard prediction labels."""
        matrix = np.zeros((self.n_models, self.n_models))

        for i, m1 in enumerate(self.models):
            for j, m2 in enumerate(self.models):
                pred1 = self.model_preds[m1]["y_pred"]
                pred2 = self.model_preds[m2]["y_pred"]
                matrix[i, j] = cohen_kappa_score(pred1, pred2)

        return pd.DataFrame(matrix, index=self.models, columns=self.models)

    def compute_error_overlaps(self) -> pd.DataFrame:
        """Computes pairwise Jaccard similarity of model prediction errors.

        Jaccard = Intersection(errors_m1, errors_m2) / Union(errors_m1, errors_m2)
        Low value means models fail on different matches, indicating high complement.
        """
        matrix = np.zeros((self.n_models, self.n_models))

        for i, m1 in enumerate(self.models):
            err1 = (self.model_preds[m1]["y_pred"] != self.y_true)
            for j, m2 in enumerate(self.models):
                err2 = (self.model_preds[m2]["y_pred"] != self.y_true)

                intersection = np.sum(err1 & err2)
                union        = np.sum(err1 | err2)

                if union > 0:
                    matrix[i, j] = intersection / union
                else:
                    matrix[i, j] = 1.0

        return pd.DataFrame(matrix, index=self.models, columns=self.models)

    def compute_disagreement_rates(self) -> pd.DataFrame:
        """Computes pairwise disagreement rates (proportion of differing predictions)."""
        matrix = np.zeros((self.n_models, self.n_models))

        for i, m1 in enumerate(self.models):
            for j, m2 in enumerate(self.models):
                pred1 = self.model_preds[m1]["y_pred"]
                pred2 = self.model_preds[m2]["y_pred"]
                matrix[i, j] = np.mean(pred1 != pred2)

        return pd.DataFrame(matrix, index=self.models, columns=self.models)

    def compute_q_statistics(self) -> pd.DataFrame:
        """Computes pairwise Q-statistics between classifiers correct/incorrect patterns.

        Q varies between -1 and 1. Values close to 0 suggest independent classifiers.
        """
        matrix = np.zeros((self.n_models, self.n_models))

        for i, m1 in enumerate(self.models):
            for j, m2 in enumerate(self.models):
                if i == j:
                    matrix[i, j] = 1.0
                    continue
                correct1 = (self.model_preds[m1]["y_pred"] == self.y_true)
                correct2 = (self.model_preds[m2]["y_pred"] == self.y_true)

                n11 = np.sum(correct1 & correct2)
                n00 = np.sum((~correct1) & (~correct2))
                n10 = np.sum(correct1 & (~correct2))
                n01 = np.sum((~correct1) & correct2)

                num = n11 * n00 - n10 * n01
                den = n11 * n00 + n10 * n01
                if den == 0:
                    matrix[i, j] = 0.0
                else:
                    matrix[i, j] = num / den

        return pd.DataFrame(matrix, index=self.models, columns=self.models)

    def compute_double_faults(self) -> pd.DataFrame:
        """Computes pairwise double-fault measure (proportion of samples where both models fail)."""
        matrix = np.zeros((self.n_models, self.n_models))
        n_samples = len(self.y_true)

        for i, m1 in enumerate(self.models):
            for j, m2 in enumerate(self.models):
                incorrect1 = (self.model_preds[m1]["y_pred"] != self.y_true)
                incorrect2 = (self.model_preds[m2]["y_pred"] != self.y_true)
                n00 = np.sum(incorrect1 & incorrect2)
                matrix[i, j] = n00 / n_samples

        return pd.DataFrame(matrix, index=self.models, columns=self.models)

    def compute_kl_divergences(self) -> pd.DataFrame:
        """Computes pairwise symmetric KL divergence of predicted probabilities."""
        matrix = np.zeros((self.n_models, self.n_models))

        for i, m1 in enumerate(self.models):
            for j, m2 in enumerate(self.models):
                if i == j:
                    matrix[i, j] = 0.0
                    continue
                p = np.clip(self.model_preds[m1]["y_prob"], 1e-15, 1.0 - 1e-15)
                q = np.clip(self.model_preds[m2]["y_prob"], 1e-15, 1.0 - 1e-15)

                kl_pq = np.sum(p * np.log(p / q), axis=1)
                kl_qp = np.sum(q * np.log(q / p), axis=1)
                matrix[i, j] = np.mean(0.5 * (kl_pq + kl_qp))

        return pd.DataFrame(matrix, index=self.models, columns=self.models)

    def generate_recommendations(self) -> List[str]:
        """Automatically checks metrics and recommends candidate pruning/complements."""
        recs = []
        corr = self.compute_probability_correlations()
        err_jaccard = self.compute_error_overlaps()

        recs.append("### Complementary Pair Recommendations")
        pairs = []
        for i, m1 in enumerate(self.models):
            for j, m2 in enumerate(self.models):
                if i >= j:
                    continue
                c_val = corr.iloc[i, j]
                j_val = err_jaccard.iloc[i, j]
                pairs.append((m1, m2, c_val, j_val))

        pairs = sorted(pairs, key=lambda x: x[2])
        for m1, m2, c_val, j_val in pairs[:4]:
            recs.append(
                f"- **{m1}** & **{m2}**: Highly complementary. "
                f"Prob Correlation = {c_val:.3f}, Error Overlap = {j_val:.3f}."
            )

        recs.append("\n### Redundancy Warnings")
        redundant = False
        for m1, m2, c_val, j_val in pairs:
            if c_val > 0.96:
                recs.append(
                    f"- ⚠️ **{m1}** and **{m2}** show extremely high probability correlation ({c_val:.3f}). "
                    f"Consider down-weighting one to prevent feature duplication."
                )
                redundant = True

        if not redundant:
            recs.append("- No critical redundancy detected. All candidate models show healthy divergence.")

        return recs
