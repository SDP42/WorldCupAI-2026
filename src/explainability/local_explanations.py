"""WorldCupAI — Phase 10: Local Match Explanation Generator.

For every predicted tournament match, generates a structured local explanation:
  - Winning probability
  - Confidence classification
  - Top positive & negative features (relative to a draw baseline)
  - Feature contributions (signed delta from baseline)
  - Natural language decision narrative
  - Prediction margin

Reads from:
  - predictions/tournament_predictions.json (Phase 8 output)
  - predictions/feature_importance.csv (global importances as proxy weights)

Writes to:
  - predictions/match_explanations.json
  - predictions/match_explanations.csv
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from src.explainability.utils import (
    FEATURE_COLS, FEATURE_LABELS, PREDICTIONS_DIR,
    get_label, get_category, classify_confidence,
    compute_entropy, compute_margin, ensure_dir, save_json,
)
from src.utils.logger import setup_logger

logger = setup_logger("local_explanations")

# Features whose positive value favors home team
HOME_POSITIVE_FEATURES = {
    "elo_diff", "rank_diff",  # home - away (positive = home better)
    "home_form_win_rate_5", "home_form_win_rate_10",
    "home_form_avg_goals_scored_5", "home_form_avg_goals_scored_10",
    "home_form_clean_sheet_rate_5", "home_form_clean_sheet_rate_10",
    "h2h_home_wins", "h2h_gd",
    "home_attack_rating", "home_defence_rating",
    "home_world_cup_titles_before",
    "home_rest_days",
    "rest_difference",
}

# Features whose positive value favors away team
AWAY_POSITIVE_FEATURES = {
    "away_form_win_rate_5", "away_form_win_rate_10",
    "away_form_avg_goals_scored_5", "away_form_avg_goals_scored_10",
    "away_form_clean_sheet_rate_5", "away_form_clean_sheet_rate_10",
    "h2h_away_wins",
    "away_attack_rating", "away_defence_rating",
    "away_world_cup_titles_before",
    "away_rest_days",
}

# Narrative templates
_CONFIDENCE_NARRATIVES = {
    "Very High": "dominates convincingly",
    "High":      "is the clear favourite",
    "Medium":    "holds a moderate advantage",
    "Low":       "holds a slight edge in a competitive matchup",
    "Very Low":  "is narrowly favoured in an extremely competitive match",
}


def _build_feature_contributions(
    match_features: Dict[str, Any],
    global_importances: Dict[str, float],
    winner: str,
    home_team: str,
) -> List[Dict[str, Any]]:
    """Computes signed feature contributions relative to the predicted winner."""
    contributions = []

    for feat in FEATURE_COLS:
        raw_val = match_features.get(feat)
        if raw_val is None:
            continue

        importance = global_importances.get(feat, 0.0)
        val = float(raw_val)

        # Determine sign: positive = favours predicted winner
        if winner == home_team:
            if feat in HOME_POSITIVE_FEATURES:
                sign = 1.0 if val > 0 else (-1.0 if val < 0 else 0.0)
            elif feat in AWAY_POSITIVE_FEATURES:
                sign = -1.0 if val > 0 else (1.0 if val < 0 else 0.0)
            else:
                sign = 0.0
        else:
            # Away team won
            if feat in AWAY_POSITIVE_FEATURES:
                sign = 1.0 if val > 0 else (-1.0 if val < 0 else 0.0)
            elif feat in HOME_POSITIVE_FEATURES:
                sign = -1.0 if val > 0 else (1.0 if val < 0 else 0.0)
            else:
                sign = 0.0

        contribution = sign * importance

        contributions.append({
            "feature":      feat,
            "label":        get_label(feat),
            "category":     get_category(feat),
            "raw_value":    round(val, 4),
            "importance":   round(importance, 6),
            "contribution": round(contribution, 6),
            "direction":    "positive" if contribution > 0 else ("negative" if contribution < 0 else "neutral"),
        })

    return sorted(contributions, key=lambda x: -abs(x["contribution"]))


def _build_narrative(
    home_team: str,
    away_team: str,
    winner: str,
    confidence: float,
    top_positive: List[Dict],
    top_negative: List[Dict],
    prob_home: float,
    prob_away: float,
) -> str:
    """Builds a natural language prediction narrative."""
    confidence_tier = classify_confidence(confidence)
    verb = _CONFIDENCE_NARRATIVES.get(confidence_tier, "is predicted to win")

    pos_labels = ", ".join(c["label"] for c in top_positive[:3])
    neg_labels = ", ".join(c["label"] for c in top_negative[:2]) if top_negative else "none"

    loser = away_team if winner == home_team else home_team

    narrative = (
        f"{winner} {verb} against {loser}. "
        f"Key positive drivers: {pos_labels if pos_labels else 'superior overall metrics'}. "
    )
    if neg_labels != "none":
        narrative += f"Counteracting factors: {neg_labels}. "

    narrative += (
        f"Win probability: {confidence * 100:.1f}% "
        f"(Home: {prob_home*100:.1f}%, Away: {prob_away*100:.1f}%)."
    )
    return narrative


class LocalExplainer:
    """Generates per-match local explanations for all predicted tournament matches."""

    def __init__(
        self,
        tournament_predictions: List[Dict[str, Any]],
        global_importances: Optional[Dict[str, float]] = None,
    ):
        self.predictions      = tournament_predictions
        self.global_importances = global_importances or {}
        self.explanations: List[Dict[str, Any]] = []

    def _load_global_importances(self) -> Dict[str, float]:
        """Loads ensemble feature importances from CSV if not provided."""
        csv_path = os.path.join(PREDICTIONS_DIR, "feature_importance.csv")
        if not os.path.exists(csv_path):
            logger.warning("feature_importance.csv not found — using uniform importances.")
            return {f: 1.0 / len(FEATURE_COLS) for f in FEATURE_COLS}

        df = pd.read_csv(csv_path)
        ensemble_df = df[df["model"] == "Ensemble (Weighted)"]
        if ensemble_df.empty:
            ensemble_df = df[df["model"] == "Average"]

        imp_dict = {}
        for _, row in ensemble_df.iterrows():
            imp_dict[row["feature"]] = float(row["importance"])
        return imp_dict

    def explain_match(self, match: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a full local explanation for a single match."""
        features = match.get("features", {})
        home_team = match["home_team"]
        away_team = match["away_team"]
        winner    = match["predicted_winner"]
        confidence = float(match["confidence"])
        entropy    = float(match.get("entropy", 0.0))

        probs = np.array([
            match.get("prob_away_win", 0.0),
            match.get("prob_draw", 0.0),
            match.get("prob_home_win", 0.0),
        ])
        margin = compute_margin(probs)

        # Feature contributions
        contributions = _build_feature_contributions(
            features, self.global_importances, winner, home_team
        )
        top_positive = [c for c in contributions if c["direction"] == "positive"][:5]
        top_negative = [c for c in contributions if c["direction"] == "negative"][:3]

        narrative = _build_narrative(
            home_team, away_team, winner, confidence,
            top_positive, top_negative,
            match.get("prob_home_win", 0.0),
            match.get("prob_away_win", 0.0),
        )

        return {
            "match_no":         match["match_no"],
            "round":            match["round"],
            "home_team":        home_team,
            "away_team":        away_team,
            "date":             match.get("date", ""),
            "predicted_winner": winner,
            "confidence":       round(confidence, 4),
            "confidence_tier":  classify_confidence(confidence),
            "entropy":          round(entropy, 4),
            "margin":           round(margin, 4),
            "prob_home_win":    round(match.get("prob_home_win", 0.0), 4),
            "prob_draw":        round(match.get("prob_draw", 0.0), 4),
            "prob_away_win":    round(match.get("prob_away_win", 0.0), 4),
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
            "all_feature_contributions": contributions,
            "narrative":        narrative,
            "shootout_played":  match.get("shootout_played", False),
        }

    def explain_all(self) -> List[Dict[str, Any]]:
        """Generates explanations for all tournament matches."""
        if not self.global_importances:
            self.global_importances = self._load_global_importances()

        logger.info(f"Generating local explanations for {len(self.predictions)} matches...")
        self.explanations = [self.explain_match(m) for m in self.predictions]
        logger.info("Local explanations complete.")
        return self.explanations

    def export(self, output_dir: str = PREDICTIONS_DIR) -> Dict[str, str]:
        """Exports match explanations JSON and CSV."""
        ensure_dir(output_dir)
        if not self.explanations:
            self.explain_all()

        # ── JSON export ───────────────────────────────────────────────────────
        json_path = os.path.join(output_dir, "match_explanations.json")
        save_json(self.explanations, json_path)
        logger.info(f"Exported match explanations JSON → {json_path}")

        # ── CSV export (flat summary) ─────────────────────────────────────────
        rows = []
        for exp in self.explanations:
            rows.append({
                "match_no":         exp["match_no"],
                "round":            exp["round"],
                "home_team":        exp["home_team"],
                "away_team":        exp["away_team"],
                "date":             exp["date"],
                "predicted_winner": exp["predicted_winner"],
                "confidence":       exp["confidence"],
                "confidence_tier":  exp["confidence_tier"],
                "entropy":          exp["entropy"],
                "margin":           exp["margin"],
                "prob_home_win":    exp["prob_home_win"],
                "prob_draw":        exp["prob_draw"],
                "prob_away_win":    exp["prob_away_win"],
                "top_positive_1":   exp["top_positive_features"][0]["label"] if exp["top_positive_features"] else "",
                "top_positive_2":   exp["top_positive_features"][1]["label"] if len(exp["top_positive_features"]) > 1 else "",
                "top_positive_3":   exp["top_positive_features"][2]["label"] if len(exp["top_positive_features"]) > 2 else "",
                "top_negative_1":   exp["top_negative_features"][0]["label"] if exp["top_negative_features"] else "",
                "top_negative_2":   exp["top_negative_features"][1]["label"] if len(exp["top_negative_features"]) > 1 else "",
                "narrative":        exp["narrative"],
                "shootout_played":  exp["shootout_played"],
            })

        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "match_explanations.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported match explanations CSV → {csv_path}")

        return {"json": json_path, "csv": csv_path}
