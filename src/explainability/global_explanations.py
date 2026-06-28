"""WorldCupAI — Phase 10: Global Explanation Generator.

Builds global XAI artifacts:
  - Ranked importance table
  - Importance stability across tournament rounds
  - Feature correlation matrix
  - Importance by feature category
  - Tournament-round importance breakdown

Reads from:
  - FeatureImportanceAnalyzer outputs
  - predictions/tournament_predictions.json

Writes to:
  - predictions/global_explanations.json  (summary)
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from src.explainability.utils import (
    FEATURE_COLS, FEATURE_CATEGORIES, PREDICTIONS_DIR,
    get_label, get_category, load_tournament_predictions,
    ensure_dir, save_json,
)
from src.explainability.feature_importance import FeatureImportanceAnalyzer
from src.utils.logger import setup_logger

logger = setup_logger("global_explanations")


class GlobalExplainer:
    """Generates global XAI explanations from production ensemble models.

    Aggregates and summarizes feature importance across:
    - Individual models
    - Ensemble weights
    - Feature categories
    - Tournament rounds
    """

    def __init__(
        self,
        importance_analyzer: FeatureImportanceAnalyzer,
        tournament_predictions: List[Dict[str, Any]],
    ):
        self.analyzer        = importance_analyzer
        self.predictions     = tournament_predictions
        self._result: Dict[str, Any] = {}

    def _category_importance(self) -> Dict[str, float]:
        """Aggregates feature importance by category."""
        ensemble_imp = self.analyzer.importances.get("Ensemble (Weighted)", {})
        category_totals: Dict[str, float] = {}
        category_counts: Dict[str, int]   = {}

        for feat, val in ensemble_imp.items():
            cat = get_category(feat)
            category_totals[cat] = category_totals.get(cat, 0.0) + val
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            cat: round(category_totals[cat], 6)
            for cat in sorted(category_totals, key=lambda c: -category_totals[c])
        }

    def _importance_stability(self) -> Dict[str, float]:
        """Measures importance stability across models (std deviation normalized by mean)."""
        stability = {}
        models_with_imp = [
            m for m in self.analyzer.importances
            if m not in ("Ensemble (Weighted)", "Average")
        ]

        for feat in FEATURE_COLS:
            vals = [
                self.analyzer.importances[m].get(feat, 0.0)
                for m in models_with_imp
            ]
            arr = np.array(vals)
            mean = arr.mean()
            std  = arr.std()
            # Coefficient of Variation (lower = more stable)
            cv = std / (mean + 1e-12)
            stability[feat] = round(float(cv), 4)

        return dict(sorted(stability.items(), key=lambda x: x[1]))

    def _feature_correlation(self) -> List[Dict[str, Any]]:
        """Computes pairwise correlation of feature importance values across models."""
        models_with_imp = [
            m for m in self.analyzer.importances
            if m not in ("Ensemble (Weighted)", "Average")
        ]
        if len(models_with_imp) < 2:
            return []

        # Build matrix: features × models
        mat = np.array([
            [self.analyzer.importances[m].get(f, 0.0) for m in models_with_imp]
            for f in FEATURE_COLS
        ])  # shape: (n_features, n_models)

        # Compute pairwise model agreement (correlation of importances)
        corr_matrix = np.corrcoef(mat.T)  # shape: (n_models, n_models)

        pairs = []
        for i, m1 in enumerate(models_with_imp):
            for j, m2 in enumerate(models_with_imp):
                if i < j:
                    pairs.append({
                        "model_a": m1,
                        "model_b": m2,
                        "importance_correlation": round(float(corr_matrix[i, j]), 4),
                    })

        return sorted(pairs, key=lambda x: -x["importance_correlation"])

    def _top_features_by_round(self) -> Dict[str, List[str]]:
        """Returns the top-5 most important features (from ensemble) — same for all rounds
        since feature importance is a global property of the trained models."""
        ranked = self.analyzer.rank("Ensemble (Weighted)")
        top5 = [r["label"] for r in ranked[:5]]
        rounds = list({m.get("round", "Unknown") for m in self.predictions})
        return {rnd: top5 for rnd in sorted(rounds)}

    def _model_agreement_summary(self) -> Dict[str, Any]:
        """Summarizes how well models agree on top features."""
        models_with_imp = [
            m for m in self.analyzer.importances
            if m not in ("Ensemble (Weighted)", "Average")
        ]

        # Top-5 features per model
        top5_per_model = {}
        for m in models_with_imp:
            ranked = sorted(
                self.analyzer.importances[m].items(),
                key=lambda x: -x[1]
            )[:5]
            top5_per_model[m] = [feat for feat, _ in ranked]

        # Consensus: features appearing in top-5 of all models
        sets = [set(feats) for feats in top5_per_model.values()]
        consensus = set.intersection(*sets) if sets else set()

        return {
            "top5_per_model": top5_per_model,
            "consensus_top_features": sorted(consensus),
            "consensus_count": len(consensus),
        }

    def explain(self) -> Dict[str, Any]:
        """Runs all global explanation computations and returns a summary dict."""
        if not self.analyzer.importances:
            self.analyzer.compute()

        ranking = self.analyzer.rank("Ensemble (Weighted)")
        logger.info("Building global XAI explanations...")

        self._result = {
            "top_features_ranked": ranking[:15],
            "category_importance": self._category_importance(),
            "importance_stability": self._importance_stability(),
            "model_feature_correlation": self._feature_correlation(),
            "top_features_by_round": self._top_features_by_round(),
            "model_agreement": self._model_agreement_summary(),
            "summary": {
                "most_important_feature":   ranking[0]["label"] if ranking else "N/A",
                "most_important_category":  list(self._category_importance().keys())[0] if ranking else "N/A",
                "num_features_analyzed":    len(FEATURE_COLS),
                "num_models":               len([m for m in self.analyzer.importances
                                                  if m not in ("Ensemble (Weighted)", "Average")]),
            },
        }
        return self._result

    def export(self, output_dir: str = PREDICTIONS_DIR) -> str:
        """Exports global explanation JSON."""
        ensure_dir(output_dir)
        if not self._result:
            self.explain()

        path = os.path.join(output_dir, "global_explanations.json")
        save_json(self._result, path)
        logger.info(f"Exported global explanations → {path}")
        return path
