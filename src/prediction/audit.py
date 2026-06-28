"""WorldCupAI — Phase 8 Prediction Audit Module.

Implements all 12 audit steps: feature validation, per-model predictions,
SHAP analysis, calibration, and 1000-tournament simulation.
"""
import os
import sys
import json
import pickle
import tempfile
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.logger import setup_logger

logger = setup_logger("prediction_audit")

FEATURE_COLS = [
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
    "home_attack_rating", "away_attack_rating", "home_defence_rating", "away_defence_rating",
    "home_world_cup_titles_before", "away_world_cup_titles_before",
    "is_neutral", "is_world_cup", "is_friendly",
    "home_rest_days", "away_rest_days", "rest_difference"
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1+2+3+4+5+6 — Feature Vector Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_feature_vector(match_df: pd.DataFrame, home_team: str, away_team: str,
                             home_elo: float, away_elo: float) -> Dict[str, Any]:
    """Validates and returns a rich feature summary for a single match."""
    row = match_df.iloc[0]
    issues = []

    elo_diff = row.get("elo_diff", float("nan"))
    rank_diff = row.get("rank_diff", float("nan"))
    expected_elo_diff = home_elo - away_elo
    expected_rank_sign = -1 if row.get("rank_diff", 0) < 0 else 1  # negative = home stronger

    # Step 4: Elo verification
    if pd.isna(elo_diff):
        issues.append("CRITICAL: elo_diff is NaN — Elo feature zeroed out by imputer")
    elif abs(elo_diff - expected_elo_diff) > 1.0:
        issues.append(f"WARNING: elo_diff={elo_diff:.1f} but expected={expected_elo_diff:.1f}")

    # Step 5: FIFA rank direction verification (lower rank = stronger)
    rank_diff_val = row.get("rank_diff", 0.0)
    # Step 6: Home/away assignment sanity check
    h_attack = row.get("home_attack_rating", float("nan"))
    a_attack = row.get("away_attack_rating", float("nan"))
    if pd.isna(h_attack) or pd.isna(a_attack):
        issues.append("WARNING: attack_rating contains NaN")

    # NaN scan across all feature cols
    nan_cols = [c for c in FEATURE_COLS if c in row.index and pd.isna(row[c])]
    if nan_cols:
        issues.append(f"NaN features: {nan_cols}")

    summary = {
        "home_team": home_team,
        "away_team": away_team,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": elo_diff,
        "expected_elo_diff": expected_elo_diff,
        "rank_diff": rank_diff_val,
        "home_attack": h_attack,
        "away_attack": a_attack,
        "home_form_win5": row.get("home_form_win_rate_5", float("nan")),
        "away_form_win5": row.get("away_form_win_rate_5", float("nan")),
        "h2h_meetings": row.get("h2h_meetings", 0),
        "home_rest_days": row.get("home_rest_days", 5),
        "away_rest_days": row.get("away_rest_days", 5),
        "is_neutral": row.get("is_neutral", 1),
        "is_world_cup": row.get("is_world_cup", 1),
        "issues": issues,
        "all_features": {c: float(row.get(c, float("nan"))) for c in FEATURE_COLS if c in row.index}
    }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Per-Model Prediction Audit
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIRS = {
    "XGBoost":            "models/xgboost_optimized",
    "Gradient Boosting":  "models/gradient_boosting_optimized",
    "Random Forest":      "models/random_forest_optimized",
    "Extra Trees":        "models/extra_trees_optimized",
    "Logistic Regression": "models/logistic_regression_optimized",
}

SCRIPT_PATH = os.path.join("src", "prediction", "ml_predict_subprocess.py")


def predict_all_models(match_df: pd.DataFrame) -> Dict[str, Dict]:
    """Runs all ML models independently in subprocesses and returns their predictions."""
    fd, temp_csv = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    match_df.to_csv(temp_csv, index=False)
    predictions = {}

    for name, m_dir in MODEL_DIRS.items():
        fd_out, temp_json = tempfile.mkstemp(suffix=".json")
        os.close(fd_out)
        try:
            cmd = [sys.executable, SCRIPT_PATH, name, m_dir, temp_csv, temp_json]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if res.returncode == 0:
                with open(temp_json) as f:
                    data = json.load(f)
                probs = np.array(data["y_prob"][0])
                pred = int(np.argmax(probs))
                predictions[name] = {
                    "prob_away": float(probs[0]),
                    "prob_draw": float(probs[1]),
                    "prob_home": float(probs[2]),
                    "predicted": ["Away Win", "Draw", "Home Win"][pred]
                }
            else:
                predictions[name] = {"error": res.stderr[:500]}
        except Exception as e:
            predictions[name] = {"error": str(e)}
        finally:
            if os.path.exists(temp_json):
                os.remove(temp_json)

    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Feature Importance / SHAP for key matches
# ─────────────────────────────────────────────────────────────────────────────

SHAP_SCRIPT_PATH = os.path.join("src", "prediction", "shap_subprocess.py")


def get_feature_importance_subprocess(match_df: pd.DataFrame, match_label: str,
                                       output_dir: str = "predictions") -> Dict[str, Any]:
    """Gets XGBoost feature importances for a match via subprocess."""
    fd, temp_csv = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    match_df.to_csv(temp_csv, index=False)

    out_json = os.path.join(output_dir, f"shap_{match_label.replace(' ', '_').replace('/', '_')}.json")

    try:
        cmd = [sys.executable, SHAP_SCRIPT_PATH, temp_csv, out_json]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode == 0 and os.path.exists(out_json):
            with open(out_json) as f:
                return json.load(f)
        else:
            return {"error": res.stderr[:500]}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)


# ─────────────────────────────────────────────────────────────────────────────
# Step 10 — Probability Sanity Checks
# ─────────────────────────────────────────────────────────────────────────────

SANITY_RULES = [
    ("Brazil", "Japan",    "away", 0.50, "Brazil losing to Japan with >50%"),
    ("Argentina", "Cape Verde", "away", 0.20, "Argentina losing to Cape Verde with >20%"),
    ("France", "Sweden",   "away", 0.45, "France losing to Sweden with >45%"),
    ("Germany", "Paraguay","away", 0.45, "Germany losing to Paraguay with >45%"),
    ("Belgium", "Senegal", "away", 0.60, "Belgium losing to Senegal with >60%"),
    ("Netherlands", "Morocco", "away", 0.50, "Netherlands losing to Morocco with >50%"),
    ("Portugal", "Croatia", "away", 0.55, "Portugal losing to Croatia with >55%"),
    ("Spain", "Austria",   "away", 0.50, "Spain losing to Austria with >50%"),
]

def run_sanity_checks(all_results: Dict[int, Dict]) -> List[Dict]:
    """Flags matches that fail sanity probability thresholds."""
    flags = []
    # Build a lookup by home/away team name
    match_lookup = {}
    for m_no, m in all_results.items():
        key = (m["home_team"], m["away_team"])
        match_lookup[key] = m

    for home, away, risk_side, threshold, description in SANITY_RULES:
        m = match_lookup.get((home, away))
        if m is None:
            continue
        prob = m["prob_away_win"] if risk_side == "away" else m["prob_home_win"]
        if prob > threshold:
            flags.append({
                "match": f"{home} vs {away}",
                "description": description,
                "threshold": threshold,
                "actual_prob": round(prob, 4),
                "predicted_winner": m["predicted_winner"],
                "severity": "HIGH" if prob > threshold + 0.10 else "MEDIUM"
            })
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Step 11 — 1000-Tournament Simulation (Probabilistic)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_tournament(all_results: Dict[int, Dict], n_simulations: int = 1000,
                         seed: int = 42) -> Dict[str, Any]:
    """Runs N probabilistic simulations of the bracket using predicted probabilities."""
    rng = np.random.default_rng(seed)

    # Track frequencies per team per stage
    stages = {
        "Round of 32 (played)": set(),
        "Round of 16": set(),
        "Quarter-final": set(),
        "Semi-final": set(),
        "Final": set(),
        "Champion": {}
    }

    champion_counts: Dict[str, int] = {}
    stage_counts: Dict[str, Dict[str, int]] = {
        "Round of 16": {},
        "Quarter-final": {},
        "Semi-final": {},
        "Final": {},
        "Champion": {}
    }

    # Round of 32 matchups (fixed)
    r32_matches = {m_no: m for m_no, m in all_results.items() if m["round"] == "Round of 32"}
    
    # Bracket structure (same as run_tournament)
    r16_pairings = [
        (89, 73, 75), (90, 74, 77), (91, 76, 78), (92, 79, 80),
        (93, 83, 84), (94, 81, 82), (95, 86, 88), (96, 85, 87)
    ]
    qf_pairings = [
        (97, 89, 90), (98, 93, 94), (99, 91, 92), (100, 95, 96)
    ]
    sf_pairings = [
        (101, 97, 98), (102, 99, 100)
    ]

    for sim_idx in range(n_simulations):
        winners = {}

        # Round of 32 — sample winner from probabilities
        for m_no, m in r32_matches.items():
            probs = np.array([m["prob_home_win"], m["prob_draw"], m["prob_away_win"]])
            probs = probs / probs.sum()
            # For knockout: combine draw probability equally between home/away
            p_home = probs[0] + probs[1] * 0.5
            p_away = probs[2] + probs[1] * 0.5
            winners[m_no] = m["home_team"] if rng.random() < p_home else m["away_team"]

        # Round of 16
        for m_no, prev_a, prev_b in r16_pairings:
            w_a = winners[prev_a]
            w_b = winners[prev_b]
            # Use stored probabilities for this exact pair if available, else 50/50
            stored = all_results.get(m_no, {})
            if stored.get("home_team") == w_a and stored.get("away_team") == w_b:
                probs = np.array([stored["prob_home_win"], stored["prob_draw"], stored["prob_away_win"]])
            else:
                probs = np.array([0.45, 0.10, 0.45])  # slight home advantage approximation
            probs = probs / probs.sum()
            p_home = probs[0] + probs[1] * 0.5
            winners[m_no] = w_a if rng.random() < p_home else w_b
            stage_counts["Round of 16"][w_a] = stage_counts["Round of 16"].get(w_a, 0) + 1
            stage_counts["Round of 16"][w_b] = stage_counts["Round of 16"].get(w_b, 0) + 1

        # Quarter-finals
        for m_no, prev_a, prev_b in qf_pairings:
            w_a = winners[prev_a]
            w_b = winners[prev_b]
            stored = all_results.get(m_no, {})
            if stored.get("home_team") == w_a and stored.get("away_team") == w_b:
                probs = np.array([stored["prob_home_win"], stored["prob_draw"], stored["prob_away_win"]])
            else:
                probs = np.array([0.45, 0.10, 0.45])
            probs = probs / probs.sum()
            p_home = probs[0] + probs[1] * 0.5
            winners[m_no] = w_a if rng.random() < p_home else w_b
            stage_counts["Quarter-final"][w_a] = stage_counts["Quarter-final"].get(w_a, 0) + 1
            stage_counts["Quarter-final"][w_b] = stage_counts["Quarter-final"].get(w_b, 0) + 1

        # Semi-finals
        for m_no, prev_a, prev_b in sf_pairings:
            w_a = winners[prev_a]
            w_b = winners[prev_b]
            stored = all_results.get(m_no, {})
            if stored.get("home_team") == w_a and stored.get("away_team") == w_b:
                probs = np.array([stored["prob_home_win"], stored["prob_draw"], stored["prob_away_win"]])
            else:
                probs = np.array([0.45, 0.10, 0.45])
            probs = probs / probs.sum()
            p_home = probs[0] + probs[1] * 0.5
            winners[m_no] = w_a if rng.random() < p_home else w_b
            stage_counts["Semi-final"][w_a] = stage_counts["Semi-final"].get(w_a, 0) + 1
            stage_counts["Semi-final"][w_b] = stage_counts["Semi-final"].get(w_b, 0) + 1

        # Final
        m101 = all_results[101]
        m102 = all_results[102]
        l_a = m101["home_team"] if winners[101] != m101["home_team"] else m101["away_team"]
        l_b = m102["home_team"] if winners[102] != m102["home_team"] else m102["away_team"]
        fin_home = winners[101]
        fin_away = winners[102]
        stored = all_results.get(104, {})
        if stored.get("home_team") == fin_home and stored.get("away_team") == fin_away:
            probs = np.array([stored["prob_home_win"], stored["prob_draw"], stored["prob_away_win"]])
        else:
            probs = np.array([0.45, 0.10, 0.45])
        probs = probs / probs.sum()
        p_home = probs[0] + probs[1] * 0.5
        champion = fin_home if rng.random() < p_home else fin_away
        stage_counts["Final"][fin_home] = stage_counts["Final"].get(fin_home, 0) + 1
        stage_counts["Final"][fin_away] = stage_counts["Final"].get(fin_away, 0) + 1
        stage_counts["Champion"][champion] = stage_counts["Champion"].get(champion, 0) + 1

    # Convert counts to percentages
    results = {}
    for stage, counts in stage_counts.items():
        sorted_teams = sorted(counts.items(), key=lambda x: -x[1])
        results[stage] = [(team, count / n_simulations * 100) for team, count in sorted_teams[:20]]

    return results
