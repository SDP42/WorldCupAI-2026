"""WorldCupAI — Phase 10: Ensemble Explainability Module.

For every predicted match, explains the ensemble prediction by showing:
  - Per-model probabilities
  - Ensemble combined probabilities
  - Voting weights applied
  - Model contribution to final decision
  - Agreement level between models
  - Confidence level

For production ensemble (Weighted Soft Voting):
  - XGBoost: weight 0.486
  - Gradient Boosting: weight 0.514
  - All others: weight 0.0

Reads from:
  - predictions/tournament_predictions.json
  - models/*/calibrated_model.pkl (for per-model predictions on match features)

Writes to:
  - predictions/ensemble_explanations.csv
  - predictions/ensemble_summary.json
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

from src.explainability.utils import (
    FEATURE_COLS, MODEL_DIRS, PREDICTIONS_DIR,
    ensure_dir, save_json,
)
from src.utils.logger import setup_logger

logger = setup_logger("ensemble_explanations")

OUTCOME_LABELS = ["Away Win", "Draw", "Home Win"]


def _predict_single_model(
    features_dict: Dict[str, Any],
    m_dir: str,
) -> Optional[np.ndarray]:
    """Runs a single ML model prediction on feature dict. Returns (3,) proba array."""
    prep_path = os.path.join(m_dir, "preprocessing.pkl")
    cal_path  = os.path.join(m_dir, "calibrated_model.pkl")
    if not os.path.exists(prep_path) or not os.path.exists(cal_path):
        return None
    try:
        with open(prep_path, "rb") as f:
            prep = pickle.load(f)
        with open(cal_path, "rb") as f:
            model = pickle.load(f)

        row = {feat: float(features_dict.get(feat, 0.0) or 0.0) for feat in FEATURE_COLS}
        df = pd.DataFrame([row])
        X = prep.transform(df)
        prob = model.predict_proba(X)[0]
        return np.array(prob, dtype=float)
    except Exception as e:
        logger.warning(f"Model prediction failed in {m_dir}: {e}")
        return None


class EnsembleExplainer:
    """Explains ensemble predictions by breaking down individual model contributions."""

    def __init__(
        self,
        tournament_predictions: List[Dict[str, Any]],
        model_dirs: Optional[Dict[str, str]] = None,
        ensemble_weights: Optional[Dict[str, float]] = None,
    ):
        self.predictions      = tournament_predictions
        self.model_dirs       = model_dirs or MODEL_DIRS
        self.ensemble_weights = ensemble_weights or {}
        self.explanations: List[Dict[str, Any]] = []

    def _agreement_score(self, model_preds: Dict[str, np.ndarray]) -> float:
        """Computes fraction of models that agree on the predicted outcome."""
        outcomes = [np.argmax(p) for p in model_preds.values()]
        if not outcomes:
            return 0.0
        most_common = max(set(outcomes), key=outcomes.count)
        return round(outcomes.count(most_common) / len(outcomes), 4)

    def explain_match(self, match: Dict[str, Any]) -> Dict[str, Any]:
        """Generates ensemble explanation for one match using stored feature vectors."""
        features = match.get("features", {})
        home_team = match["home_team"]
        away_team = match["away_team"]

        # Get per-model predictions
        model_probs: Dict[str, np.ndarray] = {}
        for m_name, m_dir in self.model_dirs.items():
            w = self.ensemble_weights.get(m_name, 0.0)
            if w <= 0:
                continue  # skip zero-weight models for speed
            prob = _predict_single_model(features, m_dir)
            if prob is not None:
                model_probs[m_name] = prob

        # Ensemble weighted average
        ensemble_probs = np.array([
            match.get("prob_away_win", 0.0),
            match.get("prob_draw",     0.0),
            match.get("prob_home_win", 0.0),
        ], dtype=float)
        ensemble_probs /= (ensemble_probs.sum() + 1e-12)

        # Model contributions (weight × max_prob)
        contributions = {}
        for m_name, probs in model_probs.items():
            w = self.ensemble_weights.get(m_name, 0.0)
            contributions[m_name] = {
                "weight":         round(w, 4),
                "prob_home_win":  round(float(probs[2]), 4),
                "prob_draw":      round(float(probs[1]), 4),
                "prob_away_win":  round(float(probs[0]), 4),
                "predicted":      OUTCOME_LABELS[int(np.argmax(probs))],
                "contribution_to_home": round(w * float(probs[2]), 4),
                "contribution_to_draw": round(w * float(probs[1]), 4),
                "contribution_to_away": round(w * float(probs[0]), 4),
            }

        agreement = self._agreement_score(model_probs)

        # Dominant model = highest weight with non-zero contribution
        dominant = max(
            self.ensemble_weights.items(),
            key=lambda x: x[1],
            default=("N/A", 0.0)
        )[0]

        return {
            "match_no":              match["match_no"],
            "round":                 match["round"],
            "home_team":             home_team,
            "away_team":             away_team,
            "ensemble_prob_home":    round(float(ensemble_probs[2]), 4),
            "ensemble_prob_draw":    round(float(ensemble_probs[1]), 4),
            "ensemble_prob_away":    round(float(ensemble_probs[0]), 4),
            "predicted_winner":      match["predicted_winner"],
            "ensemble_confidence":   round(float(np.max(ensemble_probs)), 4),
            "model_contributions":   contributions,
            "model_agreement":       agreement,
            "dominant_model":        dominant,
            "num_active_models":     len(model_probs),
        }

    def explain_all(self) -> List[Dict[str, Any]]:
        """Generates ensemble explanations for all matches."""
        logger.info(f"Generating ensemble explanations for {len(self.predictions)} matches...")
        self.explanations = [self.explain_match(m) for m in self.predictions]
        logger.info("Ensemble explanations complete.")
        return self.explanations

    def summary(self) -> Dict[str, Any]:
        """Returns ensemble-level summary statistics."""
        if not self.explanations:
            self.explain_all()

        agreements  = [e["model_agreement"] for e in self.explanations]
        confidences = [e["ensemble_confidence"] for e in self.explanations]

        # Per-model average contribution to home-win probability
        model_contrib: Dict[str, List[float]] = {}
        for exp in self.explanations:
            for m_name, contrib in exp.get("model_contributions", {}).items():
                model_contrib.setdefault(m_name, []).append(contrib["contribution_to_home"])

        avg_model_contrib = {
            m: round(float(np.mean(vals)), 4)
            for m, vals in model_contrib.items()
        }

        return {
            "ensemble_method":    "Weighted Soft Voting",
            "ensemble_weights":   self.ensemble_weights,
            "model_agreement": {
                "mean":   round(float(np.mean(agreements)), 4),
                "min":    round(float(np.min(agreements)), 4),
                "max":    round(float(np.max(agreements)), 4),
            },
            "ensemble_confidence": {
                "mean":   round(float(np.mean(confidences)), 4),
                "min":    round(float(np.min(confidences)), 4),
                "max":    round(float(np.max(confidences)), 4),
            },
            "avg_model_contributions": avg_model_contrib,
            "total_matches": len(self.explanations),
        }

    def export(self, output_dir: str = PREDICTIONS_DIR) -> Dict[str, str]:
        """Exports ensemble explanations CSV and summary JSON."""
        ensure_dir(output_dir)
        if not self.explanations:
            self.explain_all()

        # CSV (flat)
        rows = []
        for exp in self.explanations:
            base = {
                "match_no":           exp["match_no"],
                "round":              exp["round"],
                "home_team":          exp["home_team"],
                "away_team":          exp["away_team"],
                "predicted_winner":   exp["predicted_winner"],
                "ensemble_prob_home": exp["ensemble_prob_home"],
                "ensemble_prob_draw": exp["ensemble_prob_draw"],
                "ensemble_prob_away": exp["ensemble_prob_away"],
                "ensemble_confidence":exp["ensemble_confidence"],
                "model_agreement":    exp["model_agreement"],
                "dominant_model":     exp["dominant_model"],
            }
            for m_name, contrib in exp.get("model_contributions", {}).items():
                base[f"{m_name}_weight"]     = contrib["weight"]
                base[f"{m_name}_prob_home"]  = contrib["prob_home_win"]
                base[f"{m_name}_prob_draw"]  = contrib["prob_draw"]
                base[f"{m_name}_prob_away"]  = contrib["prob_away_win"]
                base[f"{m_name}_predicted"]  = contrib["predicted"]
            rows.append(base)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "ensemble_explanations.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported ensemble explanations CSV → {csv_path}")

        # Summary JSON
        summary = self.summary()
        json_path = os.path.join(output_dir, "ensemble_summary.json")
        save_json(summary, json_path)
        logger.info(f"Exported ensemble summary JSON → {json_path}")

        return {"csv": csv_path, "json": json_path}
