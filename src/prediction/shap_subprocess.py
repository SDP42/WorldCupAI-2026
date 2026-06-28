#!/usr/bin/env python3
"""WorldCupAI — Phase 8: SHAP/Feature Importance subprocess helper.

Runs inside a clean subprocess (no PyTorch) to safely load XGBoost
and generate feature importances for a given match feature vector.
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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

MODEL_DIRS = {
    "XGBoost": "models/xgboost_optimized",
    "Gradient Boosting": "models/gradient_boosting_optimized"
}

def main():
    if len(sys.argv) < 3:
        print("Usage: shap_subprocess.py <input_csv> <output_json>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_json = sys.argv[2]

    df = pd.read_csv(input_csv)
    X = df[FEATURE_COLS]

    results = {}

    for model_name, m_dir in MODEL_DIRS.items():
        prep_path = os.path.join(m_dir, "preprocessing.pkl")
        cal_path = os.path.join(m_dir, "calibrated_model.pkl")

        if not os.path.exists(prep_path) or not os.path.exists(cal_path):
            continue

        with open(prep_path, "rb") as f:
            pipeline = pickle.load(f)
        with open(cal_path, "rb") as f:
            cal_model = pickle.load(f)

        X_proc = pipeline.transform(X)

        # XGBoost — use feature importances from the base estimator
        try:
            # CalibratedClassifierCV wraps the base estimator
            if hasattr(cal_model, "calibrated_classifiers_"):
                base_clf = cal_model.calibrated_classifiers_[0].estimator
            elif hasattr(cal_model, "base_estimator"):
                base_clf = cal_model.base_estimator
            else:
                base_clf = cal_model

            if hasattr(base_clf, "feature_importances_"):
                importances = base_clf.feature_importances_
                # Get transformed feature names if available
                try:
                    feat_names = pipeline.get_feature_names_out()
                except:
                    feat_names = [f"f_{i}" for i in range(len(importances))]

                # Map importances back to original feature names where possible
                importance_dict = {}
                for i, imp in enumerate(importances):
                    name = feat_names[i] if i < len(feat_names) else f"f_{i}"
                    importance_dict[str(name)] = float(imp)

                # Sort by importance
                sorted_imp = sorted(importance_dict.items(), key=lambda x: -x[1])
                results[model_name] = {
                    "top_features": sorted_imp[:15],
                    "total_features": len(importances)
                }
            else:
                # For linear models — use coefficients
                results[model_name] = {"note": "No feature_importances_ available (linear model)"}

        except Exception as e:
            results[model_name] = {"error": str(e)}

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print("SUCCESS")

if __name__ == "__main__":
    main()
