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
# NOTE: For FIFA World Cup (neutral venue), 'home' = Team 1 (listed first in bracket),
#       'away' = Team 2 (listed second). No actual home-ground advantage exists.
FEATURE_LABELS: Dict[str, str] = {
    "elo_diff":                       "Elo Rating Difference (T1 − T2)",
    "elo_ratio":                      "Elo Rating Ratio (T1 / T2)",
    "rank_diff":                      "FIFA Ranking Difference (T1 − T2)",
    "rank_ratio":                     "FIFA Ranking Ratio (T1 / T2)",
    "home_form_win_rate_5":           "Team 1 Win Rate (Last 5)",
    "away_form_win_rate_5":           "Team 2 Win Rate (Last 5)",
    "home_form_avg_goals_scored_5":   "Team 1 Avg Goals Scored (Last 5)",
    "away_form_avg_goals_scored_5":   "Team 2 Avg Goals Scored (Last 5)",
    "home_form_avg_goals_conceded_5": "Team 1 Avg Goals Conceded (Last 5)",
    "away_form_avg_goals_conceded_5": "Team 2 Avg Goals Conceded (Last 5)",
    "home_form_clean_sheet_rate_5":   "Team 1 Clean Sheet Rate (Last 5)",
    "away_form_clean_sheet_rate_5":   "Team 2 Clean Sheet Rate (Last 5)",
    "home_form_win_rate_10":          "Team 1 Win Rate (Last 10)",
    "away_form_win_rate_10":          "Team 2 Win Rate (Last 10)",
    "home_form_avg_goals_scored_10":  "Team 1 Avg Goals Scored (Last 10)",
    "away_form_avg_goals_scored_10":  "Team 2 Avg Goals Scored (Last 10)",
    "home_form_avg_goals_conceded_10":"Team 1 Avg Goals Conceded (Last 10)",
    "away_form_avg_goals_conceded_10":"Team 2 Avg Goals Conceded (Last 10)",
    "home_form_clean_sheet_rate_10":  "Team 1 Clean Sheet Rate (Last 10)",
    "away_form_clean_sheet_rate_10":  "Team 2 Clean Sheet Rate (Last 10)",
    "h2h_meetings":                   "Head-to-Head Meetings",
    # h2h_home_wins = historical wins by Team 1 (positional, NOT venue-based)
    "h2h_home_wins":                  "H2H Team 1 Wins (Neutral Venue)",
    # h2h_away_wins = historical wins by Team 2 (positional, NOT venue-based)
    "h2h_away_wins":                  "H2H Team 2 Wins (Neutral Venue)",
    "h2h_draws":                      "H2H Draws",
    "h2h_gd":                         "H2H Goal Difference (T1 perspective)",
    "home_attack_rating":             "Team 1 Attack Rating",
    "away_attack_rating":             "Team 2 Attack Rating",
    "home_defence_rating":            "Team 1 Defence Rating",
    "away_defence_rating":            "Team 2 Defence Rating",
    "home_world_cup_titles_before":   "Team 1 World Cup Titles",
    "away_world_cup_titles_before":   "Team 2 World Cup Titles",
    "is_neutral":                     "Neutral Venue (always 1 for World Cup)",
    "is_world_cup":                   "World Cup Match",
    "is_friendly":                    "Friendly Match",
    "home_rest_days":                 "Team 1 Rest Days",
    "away_rest_days":                 "Team 2 Rest Days",
    "rest_difference":                "Rest Day Difference (T1 − T2)",
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
# NOTE: For World Cup (neutral venue), predicted_outcome in match reports uses
# team names (e.g. "Argentina Win") instead of "Home Win" / "Away Win".
# These integer-indexed labels are retained for internal model class mapping only.
OUTCOME_LABELS = {0: "Team 2 Win", 1: "Draw", 2: "Team 1 Win"}

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
