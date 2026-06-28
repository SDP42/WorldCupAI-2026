"""WorldCupAI — Phase 10: Counterfactual Analysis Engine.

Supports "What-if" analysis: perturbs input features for key matches
and re-runs predictions to quantify probability changes and decision flips.

Perturbation scenarios:
  - Elo +50 / -50
  - Recent Form +20% / -20%
  - Attack Rating +0.1 / -0.1
  - Defence Rating +0.1 / -0.1
  - FIFA Ranking ±10

Scope: Final + Semi-finals + Quarter-finals (7 key matches)

Reads from:
  - predictions/tournament_predictions.json (features stored per match)

Writes to:
  - predictions/counterfactual_examples.json
  - predictions/counterfactual_report.csv
"""
import os
import copy
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

from src.explainability.utils import (
    FEATURE_COLS, PREDICTIONS_DIR,
    get_label, classify_confidence, ensure_dir, save_json,
)
from src.utils.logger import setup_logger

logger = setup_logger("counterfactuals")

# Rounds included in counterfactual analysis (most impactful matches)
COUNTERFACTUAL_ROUNDS = {"Quarter-final", "Semi-final", "Third Place Play-off", "Final"}

# Perturbation scenarios: (scenario_name, feature, delta, description)
PERTURBATION_SCENARIOS: List[Tuple[str, str, float, str]] = [
    ("Elo +50",           "elo_diff",              +50.0,  "Home team Elo rating increased by 50 points"),
    ("Elo -50",           "elo_diff",              -50.0,  "Home team Elo rating decreased by 50 points"),
    ("Form +20%",         "home_form_win_rate_5",  +0.20,  "Home team recent form win rate improved by 20%"),
    ("Form -20%",         "home_form_win_rate_5",  -0.20,  "Home team recent form win rate reduced by 20%"),
    ("Attack +0.1",       "home_attack_rating",    +0.10,  "Home team attack rating boosted by 0.1"),
    ("Attack -0.1",       "home_attack_rating",    -0.10,  "Home team attack rating reduced by 0.1"),
    ("Defence +0.1",      "home_defence_rating",   +0.10,  "Home team defence rating improved by 0.1"),
    ("Defence -0.1",      "home_defence_rating",   -0.10,  "Home team defence rating weakened by 0.1"),
    ("Ranking Better",    "rank_diff",             -10.0,  "Home team FIFA ranking improved by 10 places"),
    ("Ranking Worse",     "rank_diff",             +10.0,  "Home team FIFA ranking worsened by 10 places"),
]


def _perturb_features(
    features: Dict[str, Any],
    feature: str,
    delta: float,
) -> Dict[str, Any]:
    """Returns a copy of features with the target feature perturbed."""
    perturbed = dict(features)
    if feature in perturbed and perturbed[feature] is not None:
        try:
            perturbed[feature] = float(perturbed[feature]) + delta
        except (ValueError, TypeError):
            pass  # Non-numeric — skip
    return perturbed


def _run_prediction_from_features(
    features: Dict[str, Any],
    model_dirs: Dict[str, str],
    ensemble_weights: Dict[str, float],
) -> Optional[np.ndarray]:
    """Runs a subprocess-safe prediction using feature dict directly.

    Uses only the two weighted ensemble models (XGBoost + GB) for speed.
    Returns 3-class probability array [away, draw, home] or None on failure.
    """
    import pickle
    import tempfile
    import subprocess
    import sys
    import tempfile

    # Build a single-row DataFrame
    row = {}
    for feat in FEATURE_COLS:
        val = features.get(feat, 0.0)
        row[feat] = float(val) if val is not None else 0.0

    df = pd.DataFrame([row])

    probs_list = []
    weights_used = []

    for m_name, m_dir in model_dirs.items():
        w = ensemble_weights.get(m_name, 0.0)
        if w <= 0:
            continue

        prep_path = os.path.join(m_dir, "preprocessing.pkl")
        cal_path  = os.path.join(m_dir, "calibrated_model.pkl")
        if not os.path.exists(prep_path) or not os.path.exists(cal_path):
            continue

        try:
            with open(prep_path, "rb") as f:
                prep = pickle.load(f)
            with open(cal_path, "rb") as f:
                model = pickle.load(f)

            X = prep.transform(df)
            prob = model.predict_proba(X)[0]  # shape (3,)
            probs_list.append(prob)
            weights_used.append(w)
        except Exception as e:
            logger.warning(f"Counterfactual prediction failed for {m_name}: {e}")

    if not probs_list:
        return None

    # Weighted average
    weights_used = np.array(weights_used)
    weights_used /= weights_used.sum()
    combined = sum(w * p for w, p in zip(weights_used, probs_list))
    combined /= combined.sum()
    return combined


class CounterfactualEngine:
    """Generates What-if counterfactual analyses for key tournament matches."""

    def __init__(
        self,
        tournament_predictions: List[Dict[str, Any]],
        model_dirs: Optional[Dict[str, str]] = None,
        ensemble_weights: Optional[Dict[str, float]] = None,
    ):
        self.predictions      = tournament_predictions
        self.model_dirs       = model_dirs or {
            "XGBoost": "models/xgboost_optimized",
            "Gradient Boosting": "models/gradient_boosting_optimized",
        }
        self.ensemble_weights = ensemble_weights or {
            "XGBoost": 0.486, "Gradient Boosting": 0.514
        }
        self.examples: List[Dict[str, Any]] = []

    def _analyze_match(self, match: Dict[str, Any]) -> Dict[str, Any]:
        """Runs all perturbation scenarios for a single match."""
        home_team = match["home_team"]
        away_team = match["away_team"]
        original_features = match.get("features", {})

        original_probs = np.array([
            match.get("prob_away_win", 0.0),
            match.get("prob_draw",     0.0),
            match.get("prob_home_win", 0.0),
        ], dtype=float)
        original_probs /= (original_probs.sum() + 1e-12)

        original_winner  = match["predicted_winner"]
        original_confidence = float(np.max(original_probs))

        scenarios_results = []

        for scenario_name, feat, delta, description in PERTURBATION_SCENARIOS:
            perturbed = _perturb_features(original_features, feat, delta)
            new_probs  = _run_prediction_from_features(
                perturbed, self.model_dirs, self.ensemble_weights
            )

            if new_probs is None:
                continue

            new_winner    = home_team if np.argmax(new_probs) == 2 else (
                            away_team if np.argmax(new_probs) == 0 else "Draw")
            new_confidence = float(np.max(new_probs))
            decision_flip  = (new_winner != original_winner and new_winner not in ("Draw",))

            prob_delta = float(np.max(new_probs) - np.max(original_probs))

            scenarios_results.append({
                "scenario":           scenario_name,
                "feature_perturbed":  get_label(feat),
                "delta_applied":      delta,
                "description":        description,
                "original_winner":    original_winner,
                "new_winner":         new_winner,
                "original_confidence": round(original_confidence, 4),
                "new_confidence":     round(new_confidence, 4),
                "confidence_delta":   round(prob_delta, 4),
                "decision_flip":      decision_flip,
                "new_prob_home":      round(float(new_probs[2]), 4),
                "new_prob_draw":      round(float(new_probs[1]), 4),
                "new_prob_away":      round(float(new_probs[0]), 4),
            })

        n_flips = sum(1 for s in scenarios_results if s["decision_flip"])

        return {
            "match_no":             match["match_no"],
            "round":                match["round"],
            "home_team":            home_team,
            "away_team":            away_team,
            "original_winner":      original_winner,
            "original_confidence":  round(original_confidence, 4),
            "original_probs": {
                "home": round(float(original_probs[2]), 4),
                "draw": round(float(original_probs[1]), 4),
                "away": round(float(original_probs[0]), 4),
            },
            "scenarios":            scenarios_results,
            "decision_flips":       n_flips,
            "robustness_score":     round(1.0 - n_flips / max(len(scenarios_results), 1), 4),
        }

    def run(self) -> List[Dict[str, Any]]:
        """Runs counterfactual analysis for all key matches."""
        key_matches = [
            m for m in self.predictions
            if m.get("round") in COUNTERFACTUAL_ROUNDS
        ]
        logger.info(f"Running counterfactual analysis on {len(key_matches)} key matches...")

        for match in key_matches:
            logger.info(f"  Counterfactuals for: {match['home_team']} vs {match['away_team']}")
            result = self._analyze_match(match)
            self.examples.append(result)

        logger.info(f"Counterfactual analysis complete. {len(self.examples)} matches analyzed.")
        return self.examples

    def export(self, output_dir: str = PREDICTIONS_DIR) -> Dict[str, str]:
        """Exports counterfactual examples JSON and flat CSV."""
        ensure_dir(output_dir)
        if not self.examples:
            self.run()

        # JSON
        json_path = os.path.join(output_dir, "counterfactual_examples.json")
        save_json(self.examples, json_path)
        logger.info(f"Exported counterfactual examples JSON → {json_path}")

        # CSV (flat — one row per scenario per match)
        rows = []
        for ex in self.examples:
            for sc in ex.get("scenarios", []):
                rows.append({
                    "match_no":             ex["match_no"],
                    "round":                ex["round"],
                    "home_team":            ex["home_team"],
                    "away_team":            ex["away_team"],
                    "original_winner":      ex["original_winner"],
                    "original_confidence":  ex["original_confidence"],
                    "scenario":             sc["scenario"],
                    "feature_perturbed":    sc["feature_perturbed"],
                    "delta_applied":        sc["delta_applied"],
                    "new_winner":           sc["new_winner"],
                    "new_confidence":       sc["new_confidence"],
                    "confidence_delta":     sc["confidence_delta"],
                    "decision_flip":        sc["decision_flip"],
                    "robustness_score":     ex["robustness_score"],
                })

        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "counterfactual_report.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported counterfactual report CSV → {csv_path}")

        return {"json": json_path, "csv": csv_path}
