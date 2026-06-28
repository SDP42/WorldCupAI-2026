#!/usr/bin/env python3
"""WorldCupAI — Phase 6: Subprocess helper to export ML predictions.

This script runs in a separate process that does NOT import PyTorch,
preventing OpenMP/thread library conflicts (SIGSEGV/139) on macOS.
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.cross_validation import time_aware_split
from src.models.trainer import prepare_targets
from src.deep_learning.evaluation import export_ml_predictions

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
    "home_attack_rating", "away_attack_rating",
    "home_defence_rating", "away_defence_rating",
    "home_world_cup_titles_before", "away_world_cup_titles_before",
    "is_neutral", "is_world_cup", "is_friendly",
    "home_rest_days", "away_rest_days", "rest_difference",
]

ML_MODELS = {
    "XGBoost":            "models/xgboost_optimized",
    "Gradient Boosting":  "models/gradient_boosting_optimized",
    "Random Forest":      "models/random_forest_optimized",
    "Extra Trees":        "models/extra_trees_optimized",
    "Logistic Regression":"models/logistic_regression_optimized",
}

PREDICTIONS_DIR = "predictions"


def main():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    # Load and split
    df = pd.read_parquet("processed/feature_store.parquet")
    df["target"] = prepare_targets(df)
    df_modern = df[df["date"] >= pd.to_datetime("2005-01-01")].copy().sort_values("date").reset_index(drop=True)
    _, _, test_df = time_aware_split(df_modern, train_end="2018-12-31", val_end="2022-12-31")
    test_idx = test_df.index.values
    y_test = test_df["target"].values.astype(np.int64)

    ml_metrics_all = {}

    for ml_name, ml_dir in ML_MODELS.items():
        if not os.path.isdir(ml_dir):
            continue

        safe_name = ml_name.lower().replace(" ", "_")
        csv_path  = os.path.join(PREDICTIONS_DIR, f"{safe_name}_predictions.csv")
        prep_pkl  = os.path.join(ml_dir, "preprocessing.pkl")
        cal_pkl   = os.path.join(ml_dir, "calibrated_model.pkl")

        if not os.path.exists(cal_pkl):
            continue

        # Load models
        with open(prep_pkl, "rb") as f:
            ml_pipeline = pickle.load(f)

        X_ml_test = ml_pipeline.transform(test_df[FEATURE_COLS])
        
        # Run export
        ml_metrics = export_ml_predictions(
            model_name=ml_name,
            model_dir=ml_dir,
            X_test=X_ml_test,
            y_test=y_test,
            test_indices=test_idx,
            output_csv_path=csv_path,
        )
        ml_metrics_all[ml_name] = ml_metrics

    # Save metrics output to JSON for parent process
    out_path = os.path.join(PREDICTIONS_DIR, "ml_metrics_temp.json")
    with open(out_path, "w") as f:
        json.dump(ml_metrics_all, f, indent=4)
    print("SUCCESS")


if __name__ == "__main__":
    main()
