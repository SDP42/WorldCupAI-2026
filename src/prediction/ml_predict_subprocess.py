#!/usr/bin/env python3
"""WorldCupAI — Phase 8: Subprocess helper for ML predictions.

Loads and predicts with ML models (XGBoost, Gradient Boosting, etc.)
in a separate process to avoid OpenMP/PyTorch library conflict SIGSEGV on macOS.
"""
import os
import sys
import json
import pickle
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    if len(sys.argv) < 5:
        print("Usage: ml_predict_subprocess.py <model_name> <model_dir> <input_csv> <output_json>")
        sys.exit(1)

    model_name = sys.argv[1]
    model_dir = sys.argv[2]
    input_csv = sys.argv[3]
    output_json = sys.argv[4]

    # Load data
    test_df = pd.read_csv(input_csv)

    # Load artifacts
    prep_path = os.path.join(model_dir, "preprocessing.pkl")
    cal_path = os.path.join(model_dir, "calibrated_model.pkl")

    if not os.path.exists(prep_path) or not os.path.exists(cal_path):
        print(f"Error: Model artifacts not found in {model_dir}", file=sys.stderr)
        sys.exit(2)

    with open(prep_path, "rb") as f:
        pipeline = pickle.load(f)
    with open(cal_path, "rb") as f:
        model = pickle.load(f)

    # Transform and predict
    # Feature columns used in training
    feature_cols = [
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
    
    X_test = test_df[feature_cols]
    X_processed = pipeline.transform(X_test)
    y_prob = model.predict_proba(X_processed)
    y_pred = model.predict(X_processed)

    # Export results as JSON
    results = {
        "y_prob": y_prob.tolist(),
        "y_pred": [int(p) for p in y_pred]
    }

    with open(output_json, "w") as f:
        json.dump(results, f)

    print("SUCCESS")


if __name__ == "__main__":
    main()
