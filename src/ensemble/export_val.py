#!/usr/bin/env python3
"""WorldCupAI — Phase 7: Subprocess helper to export validation set ML predictions.

Run in a separate process (no torch) to avoid OpenMP/PyTorch library conflicts
(SIGSEGV/139) on macOS.  Produces predictions/val_{name}_predictions.csv for
every Phase 5 optimized ML model.
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.cross_validation import time_aware_split
from src.models.trainer import prepare_targets

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

    # Load and split data
    df = pd.read_parquet("processed/feature_store.parquet")
    df["target"] = prepare_targets(df)
    df_modern = (
        df[df["date"] >= pd.to_datetime("2005-01-01")]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )
    _, val_df, _ = time_aware_split(df_modern, train_end="2018-12-31", val_end="2022-12-31")
    val_idx = val_df.index.values
    y_val   = val_df["target"].values.astype(np.int64)

    for ml_name, ml_dir in ML_MODELS.items():
        if not os.path.isdir(ml_dir):
            print(f"[SKIP] {ml_name}: directory not found.")
            continue

        prep_pkl = os.path.join(ml_dir, "preprocessing.pkl")
        cal_pkl  = os.path.join(ml_dir, "calibrated_model.pkl")

        if not os.path.exists(cal_pkl):
            print(f"[SKIP] {ml_name}: calibrated_model.pkl not found.")
            continue

        with open(prep_pkl, "rb") as f:
            pipeline = pickle.load(f)
        with open(cal_pkl, "rb") as f:
            model = pickle.load(f)

        X_val = pipeline.transform(val_df[FEATURE_COLS])
        y_prob = model.predict_proba(X_val)
        y_pred = model.predict(X_val)

        safe_name = ml_name.lower().replace(" ", "_")
        out_path  = os.path.join(PREDICTIONS_DIR, f"val_{safe_name}_predictions.csv")

        pd.DataFrame({
            "match_id":        val_idx,
            "true_label":      y_val,
            "predicted_label": y_pred,
            "prob_away_win":   y_prob[:, 0].round(6),
            "prob_draw":       y_prob[:, 1].round(6),
            "prob_home_win":   y_prob[:, 2].round(6),
        }).to_csv(out_path, index=False)

        print(f"[OK] {ml_name} → {out_path} ({len(val_df):,} rows)")

    print("SUCCESS")


if __name__ == "__main__":
    main()
