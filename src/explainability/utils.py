"""WorldCupAI — Phase 10: XAI Shared Utilities.

Provides feature metadata, confidence thresholds, label constants,
and reusable helper functions for all explainability modules.
"""
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any

# ── Feature column list (must match knockout_engine.py / shap_subprocess.py) ──
FEATURE_COLS: List[str] = [
    "elo_diff", "elo_ratio", "rank_diff", "rank_ratio",
    "home_form_win_rate_5", "away_form_win_rate_5",
    "home_form_avg_goals_scored_5", "away_form_avg_goals_scored_5",
    "home_form_avg_goals_conceded_5", "away_form_avg_goals_conceded_5",
    "home_form_clean_sheet_rate_5", "away_form_clean_sheet_rate_5",
    "home_form_win_rate_10", "away_form_win_rate_10",
    "home_form_avg_goals_scored_10", "away_form_avg_goals_scored_10",
    "home_form_avg_goals_conceded_10", "away_form_avg_goals_conceded_10",
    "home_form_clean_sheet_rate_10", "away_form_clean_sheet_rate_10",
    "h2h_meetings", "h2h_home_wins", "h2h_away_wins", "h2h_draws", "h2h_gd",
    "home_attack_rating", "away_attack_rating",
    "home_defence_rating", "away_defence_rating",
    "home_world_cup_titles_before", "away_world_cup_titles_before",
    "is_neutral", "is_world_cup", "is_friendly",
    "home_rest_days", "away_rest_days", "rest_difference",
]

# ── Human-readable feature labels ─────────────────────────────────────────────
FEATURE_LABELS: Dict[str, str] = {
    "elo_diff":                       "Elo Rating Difference",
    "elo_ratio":                      "Elo Rating Ratio",
    "rank_diff":                      "FIFA Ranking Difference",
    "rank_ratio":                     "FIFA Ranking Ratio",
    "home_form_win_rate_5":           "Home Win Rate (Last 5)",
    "away_form_win_rate_5":           "Away Win Rate (Last 5)",
    "home_form_avg_goals_scored_5":   "Home Avg Goals Scored (Last 5)",
    "away_form_avg_goals_scored_5":   "Away Avg Goals Scored (Last 5)",
    "home_form_avg_goals_conceded_5": "Home Avg Goals Conceded (Last 5)",
    "away_form_avg_goals_conceded_5": "Away Avg Goals Conceded (Last 5)",
    "home_form_clean_sheet_rate_5":   "Home Clean Sheet Rate (Last 5)",
    "away_form_clean_sheet_rate_5":   "Away Clean Sheet Rate (Last 5)",
    "home_form_win_rate_10":          "Home Win Rate (Last 10)",
    "away_form_win_rate_10":          "Away Win Rate (Last 10)",
    "home_form_avg_goals_scored_10":  "Home Avg Goals Scored (Last 10)",
    "away_form_avg_goals_scored_10":  "Away Avg Goals Scored (Last 10)",
    "home_form_avg_goals_conceded_10":"Home Avg Goals Conceded (Last 10)",
    "away_form_avg_goals_conceded_10":"Away Avg Goals Conceded (Last 10)",
    "home_form_clean_sheet_rate_10":  "Home Clean Sheet Rate (Last 10)",
    "away_form_clean_sheet_rate_10":  "Away Clean Sheet Rate (Last 10)",
    "h2h_meetings":                   "Head-to-Head Meetings",
    "h2h_home_wins":                  "H2H Home Wins",
    "h2h_away_wins":                  "H2H Away Wins",
    "h2h_draws":                      "H2H Draws",
    "h2h_gd":                         "H2H Goal Difference",
    "home_attack_rating":             "Home Attack Rating",
    "away_attack_rating":             "Away Attack Rating",
    "home_defence_rating":            "Home Defence Rating",
    "away_defence_rating":            "Away Defence Rating",
    "home_world_cup_titles_before":   "Home World Cup Titles",
    "away_world_cup_titles_before":   "Away World Cup Titles",
    "is_neutral":                     "Neutral Venue",
    "is_world_cup":                   "World Cup Match",
    "is_friendly":                    "Friendly Match",
    "home_rest_days":                 "Home Rest Days",
    "away_rest_days":                 "Away Rest Days",
    "rest_difference":                "Rest Day Difference",
}

# ── Feature categories ─────────────────────────────────────────────────────────
FEATURE_CATEGORIES: Dict[str, str] = {
    "elo_diff": "Elo & Ranking",
    "elo_ratio": "Elo & Ranking",
    "rank_diff": "Elo & Ranking",
    "rank_ratio": "Elo & Ranking",
    "home_form_win_rate_5": "Recent Form",
    "away_form_win_rate_5": "Recent Form",
    "home_form_avg_goals_scored_5": "Recent Form",
    "away_form_avg_goals_scored_5": "Recent Form",
    "home_form_avg_goals_conceded_5": "Recent Form",
    "away_form_avg_goals_conceded_5": "Recent Form",
    "home_form_clean_sheet_rate_5": "Recent Form",
    "away_form_clean_sheet_rate_5": "Recent Form",
    "home_form_win_rate_10": "Recent Form",
    "away_form_win_rate_10": "Recent Form",
    "home_form_avg_goals_scored_10": "Recent Form",
    "away_form_avg_goals_scored_10": "Recent Form",
    "home_form_avg_goals_conceded_10": "Recent Form",
    "away_form_avg_goals_conceded_10": "Recent Form",
    "home_form_clean_sheet_rate_10": "Recent Form",
    "away_form_clean_sheet_rate_10": "Recent Form",
    "h2h_meetings": "Head-to-Head",
    "h2h_home_wins": "Head-to-Head",
    "h2h_away_wins": "Head-to-Head",
    "h2h_draws": "Head-to-Head",
    "h2h_gd": "Head-to-Head",
    "home_attack_rating": "Team Strength",
    "away_attack_rating": "Team Strength",
    "home_defence_rating": "Team Strength",
    "away_defence_rating": "Team Strength",
    "home_world_cup_titles_before": "Tournament History",
    "away_world_cup_titles_before": "Tournament History",
    "is_neutral": "Match Context",
    "is_world_cup": "Match Context",
    "is_friendly": "Match Context",
    "home_rest_days": "Fitness",
    "away_rest_days": "Fitness",
    "rest_difference": "Fitness",
}

# ── Confidence thresholds ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLDS = {
    "Very High": 0.75,
    "High":      0.60,
    "Medium":    0.45,
    "Low":       0.35,
    # Below 0.35 → Very Low
}

# ── Outcome labels ─────────────────────────────────────────────────────────────
OUTCOME_LABELS = {0: "Away Win", 1: "Draw", 2: "Home Win"}

# ── Model directories ──────────────────────────────────────────────────────────
MODEL_DIRS: Dict[str, str] = {
    "XGBoost":            "models/xgboost_optimized",
    "Gradient Boosting":  "models/gradient_boosting_optimized",
    "Random Forest":      "models/random_forest_optimized",
    "Extra Trees":        "models/extra_trees_optimized",
    "Logistic Regression":"models/logistic_regression_optimized",
}

# ── Output paths ───────────────────────────────────────────────────────────────
PREDICTIONS_DIR   = "predictions"
PLOTS_DIR         = os.path.join("outputs", "plots")
REPORTS_DIR       = "."  # Reports live at project root


def get_label(feature: str) -> str:
    """Returns human-readable label for a feature column."""
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def get_category(feature: str) -> str:
    """Returns the category for a feature column."""
    return FEATURE_CATEGORIES.get(feature, "Other")


def classify_confidence(prob: float) -> str:
    """Classifies a winning probability into a confidence tier."""
    if prob >= CONFIDENCE_THRESHOLDS["Very High"]:
        return "Very High"
    elif prob >= CONFIDENCE_THRESHOLDS["High"]:
        return "High"
    elif prob >= CONFIDENCE_THRESHOLDS["Medium"]:
        return "Medium"
    elif prob >= CONFIDENCE_THRESHOLDS["Low"]:
        return "Low"
    else:
        return "Very Low"


def compute_entropy(probs: np.ndarray) -> float:
    """Computes Shannon entropy of a probability distribution."""
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log2(probs)))


def compute_margin(probs: np.ndarray) -> float:
    """Computes the margin between the top-2 probabilities."""
    sorted_p = np.sort(probs)[::-1]
    return float(sorted_p[0] - sorted_p[1])


def strip_preprocessor_prefix(name: str) -> str:
    """Strips sklearn pipeline prefixes like 'num__' from feature names."""
    for prefix in ("num__", "cat__", "remainder__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def ensure_dir(path: str) -> None:
    """Creates directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_tournament_predictions(path: str = "predictions/tournament_predictions.json") -> List[Dict[str, Any]]:
    """Loads Phase 8 tournament prediction results."""
    with open(path) as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    """Saves data as indented JSON."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def match_label(match: Dict[str, Any]) -> str:
    """Returns a formatted match label string."""
    return f"Match {match['match_no']}: {match['home_team']} vs {match['away_team']} ({match['round']})"
