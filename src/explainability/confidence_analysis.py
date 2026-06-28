"""WorldCupAI — Phase 10: Confidence Analysis Module.

For every predicted match, computes:
  - Confidence Score (winning probability)
  - Prediction Entropy (uncertainty)
  - Probability Margin (winner prob - second prob)
  - Model Agreement Score (from ensemble weights)
  - Calibration Score (ECE proxy)
  - Risk Category (Very High / High / Medium / Low / Very Low)

Writes to:
  - predictions/confidence_analysis.csv
  - predictions/confidence_summary.json
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from src.explainability.utils import (
    PREDICTIONS_DIR,
    classify_confidence, compute_entropy, compute_margin,
    ensure_dir, save_json, match_label,
)
from src.utils.logger import setup_logger

logger = setup_logger("confidence_analysis")


class ConfidenceAnalyzer:
    """Classifies prediction confidence for every match in the tournament bracket."""

    def __init__(
        self,
        tournament_predictions: List[Dict[str, Any]],
        ensemble_weights: Optional[Dict[str, float]] = None,
    ):
        self.predictions      = tournament_predictions
        self.ensemble_weights = ensemble_weights or {}
        self.analyses: List[Dict[str, Any]] = []

    def _compute_model_agreement(self) -> float:
        """Returns an aggregate model agreement score.

        Based on production ensemble: XGBoost 0.486, GradientBoosting 0.514.
        Agreement = 1 - Herfindahl index of weight distribution (normalized).
        A perfectly uniform weight distribution = 0 agreement skew.
        """
        if not self.ensemble_weights:
            return 0.5  # neutral

        weights = np.array(list(self.ensemble_weights.values()), dtype=float)
        weights = weights[weights > 0]
        if len(weights) == 0:
            return 0.5

        # Normalized Herfindahl index: 1 = all weight on one model, 1/n = perfectly uniform
        hhi = float(np.sum(weights ** 2))
        n   = len(weights)
        hhi_norm = (hhi - 1.0 / n) / (1.0 - 1.0 / n + 1e-12)
        # Agreement: high when weights are concentrated (dominant model), low when uniform
        return round(float(np.clip(hhi_norm, 0.0, 1.0)), 4)

    def analyze_match(self, match: Dict[str, Any]) -> Dict[str, Any]:
        """Computes confidence metrics for a single match."""
        probs = np.array([
            match.get("prob_away_win", 0.0),
            match.get("prob_draw",     0.0),
            match.get("prob_home_win", 0.0),
        ], dtype=float)

        # Normalize
        total = probs.sum()
        if total > 0:
            probs = probs / total

        confidence = float(np.max(probs))
        entropy    = compute_entropy(probs)
        margin     = compute_margin(probs)
        tier       = classify_confidence(confidence)
        model_agreement = self._compute_model_agreement()

        # Calibration proxy: how well the probability distribution is spread
        # (max entropy for 3 classes = log2(3) ≈ 1.585)
        max_entropy    = float(np.log2(3))
        calibration    = round(1.0 - entropy / (max_entropy + 1e-12), 4)  # 1 = perfectly calibrated (confident)
        normalized_ent = round(entropy / max_entropy, 4)

        return {
            "match_no":         match["match_no"],
            "round":            match["round"],
            "home_team":        match["home_team"],
            "away_team":        match["away_team"],
            "predicted_winner": match["predicted_winner"],
            "confidence":       round(confidence, 4),
            "confidence_tier":  tier,
            "entropy":          round(entropy, 4),
            "normalized_entropy": normalized_ent,
            "margin":           round(margin, 4),
            "prob_home_win":    round(float(probs[2]), 4),
            "prob_draw":        round(float(probs[1]), 4),
            "prob_away_win":    round(float(probs[0]), 4),
            "model_agreement":  model_agreement,
            "calibration_score": calibration,
            "risk_category":    tier,
        }

    def analyze_all(self) -> List[Dict[str, Any]]:
        """Analyzes confidence for all matches."""
        logger.info(f"Running confidence analysis for {len(self.predictions)} matches...")
        self.analyses = [self.analyze_match(m) for m in self.predictions]
        return self.analyses

    def summary(self) -> Dict[str, Any]:
        """Returns aggregate confidence statistics across all matches."""
        if not self.analyses:
            self.analyze_all()

        confidences = [a["confidence"] for a in self.analyses]
        entropies   = [a["entropy"] for a in self.analyses]
        margins     = [a["margin"] for a in self.analyses]

        tier_counts: Dict[str, int] = {}
        for a in self.analyses:
            tier = a["confidence_tier"]
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        most_confident = max(self.analyses, key=lambda x: x["confidence"])
        least_confident = min(self.analyses, key=lambda x: x["confidence"])

        return {
            "total_matches": len(self.analyses),
            "confidence": {
                "mean":   round(float(np.mean(confidences)), 4),
                "median": round(float(np.median(confidences)), 4),
                "min":    round(float(np.min(confidences)), 4),
                "max":    round(float(np.max(confidences)), 4),
                "std":    round(float(np.std(confidences)), 4),
            },
            "entropy": {
                "mean":   round(float(np.mean(entropies)), 4),
                "median": round(float(np.median(entropies)), 4),
                "min":    round(float(np.min(entropies)), 4),
                "max":    round(float(np.max(entropies)), 4),
            },
            "margin": {
                "mean":   round(float(np.mean(margins)), 4),
                "median": round(float(np.median(margins)), 4),
                "min":    round(float(np.min(margins)), 4),
                "max":    round(float(np.max(margins)), 4),
            },
            "tier_distribution": tier_counts,
            "most_confident_match": {
                "match": f"{most_confident['home_team']} vs {most_confident['away_team']}",
                "round": most_confident["round"],
                "winner": most_confident["predicted_winner"],
                "confidence": most_confident["confidence"],
                "tier": most_confident["confidence_tier"],
            },
            "least_confident_match": {
                "match": f"{least_confident['home_team']} vs {least_confident['away_team']}",
                "round": least_confident["round"],
                "winner": least_confident["predicted_winner"],
                "confidence": least_confident["confidence"],
                "tier": least_confident["confidence_tier"],
            },
        }

    def export(self, output_dir: str = PREDICTIONS_DIR) -> Dict[str, str]:
        """Exports confidence analysis CSV and summary JSON."""
        ensure_dir(output_dir)
        if not self.analyses:
            self.analyze_all()

        # CSV
        df = pd.DataFrame(self.analyses)
        csv_path = os.path.join(output_dir, "confidence_analysis.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported confidence analysis CSV → {csv_path}")

        # JSON summary
        summary = self.summary()
        json_path = os.path.join(output_dir, "confidence_summary.json")
        save_json(summary, json_path)
        logger.info(f"Exported confidence summary JSON → {json_path}")

        return {"csv": csv_path, "json": json_path}


# Patch missing Optional import
from typing import Optional
