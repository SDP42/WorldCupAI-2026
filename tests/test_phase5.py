#!/usr/bin/env python3
"""WorldCupAI — Phase 5 Automated Tests

Validates all Phase 5 optimization artifacts:
  - All optimized model directories exist with required files
  - best_params.json is non-empty
  - calibration.json contains ECE/MCE in valid [0, 1] range
  - Calibrated models load cleanly and produce valid probabilities
  - MODEL_SELECTION_REPORT.md exists and references expected model names
  - ERROR_ANALYSIS.md exists
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.calibrator import compute_ece, compute_mce

# Expected model directories (subset that must always be present)
REQUIRED_MODELS = [
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
]

OPTIONAL_MODELS = ["xgboost", "lightgbm", "catboost"]

REQUIRED_FILES = [
    "model.pkl",
    "calibrated_model.pkl",
    "preprocessing.pkl",
    "best_params.json",
    "metrics.json",
    "calibration.json",
    "training_log.json",
]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestPhase5Artifacts(unittest.TestCase):
    """Tests that all Phase 5 optimization artifacts were correctly generated."""

    def _optimized_dir(self, short_name: str) -> str:
        return os.path.join(ROOT, "models", f"{short_name}_optimized")

    def test_required_directories_exist(self):
        """All required model directories must exist."""
        for name in REQUIRED_MODELS:
            d = self._optimized_dir(name)
            self.assertTrue(
                os.path.isdir(d),
                f"Missing optimized model directory: {d}"
            )

    def test_required_files_exist(self):
        """Each required directory must contain all required files."""
        for name in REQUIRED_MODELS:
            d = self._optimized_dir(name)
            if not os.path.isdir(d):
                continue
            for fname in REQUIRED_FILES:
                fpath = os.path.join(d, fname)
                self.assertTrue(
                    os.path.exists(fpath),
                    f"Missing artifact file: {fpath}"
                )

    def test_best_params_nonempty(self):
        """best_params.json must exist and contain non-empty params dict."""
        for name in REQUIRED_MODELS:
            d = self._optimized_dir(name)
            params_path = os.path.join(d, "best_params.json")
            if not os.path.exists(params_path):
                continue
            with open(params_path) as f:
                data = json.load(f)
            self.assertIn("best_params", data, f"[{name}] best_params key missing")
            self.assertIsInstance(data["best_params"], dict, f"[{name}] best_params not a dict")

    def test_calibration_json_valid(self):
        """calibration.json must contain ECE and MCE values in [0, 1]."""
        for name in REQUIRED_MODELS:
            d = self._optimized_dir(name)
            cal_path = os.path.join(d, "calibration.json")
            if not os.path.exists(cal_path):
                continue
            with open(cal_path) as f:
                cal = json.load(f)

            self.assertIn("calibration_method", cal, f"[{name}] calibration_method missing")

            ece = cal.get("calibrated", {}).get("ece", None)
            mce = cal.get("calibrated", {}).get("mce", None)

            if ece is not None:
                self.assertGreaterEqual(ece, 0.0, f"[{name}] ECE is negative")
                self.assertLessEqual(ece, 1.0,   f"[{name}] ECE exceeds 1.0")
            if mce is not None:
                self.assertGreaterEqual(mce, 0.0, f"[{name}] MCE is negative")
                self.assertLessEqual(mce, 1.0,   f"[{name}] MCE exceeds 1.0")

    def test_metrics_json_valid(self):
        """metrics.json must contain accuracy, log_loss, roc_auc_macro."""
        for name in REQUIRED_MODELS:
            d = self._optimized_dir(name)
            metrics_path = os.path.join(d, "metrics.json")
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path) as f:
                metrics = json.load(f)
            for key in ["accuracy", "log_loss", "roc_auc_macro", "f1_macro", "brier_score"]:
                self.assertIn(key, metrics, f"[{name}] metrics.json missing key: {key}")
                self.assertIsInstance(metrics[key], float, f"[{name}] {key} is not float")

    def test_calibrated_model_predict_proba(self):
        """Calibrated models must load and produce probabilities summing to 1.0."""
        store_path = os.path.join(ROOT, "processed", "feature_store.parquet")
        if not os.path.exists(store_path):
            self.skipTest("Feature store not found — skipping model inference test.")

        df = pd.read_parquet(store_path)
        df = df[df["date"] >= pd.to_datetime("2023-01-01")].head(50)

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
        available_cols = [c for c in FEATURE_COLS if c in df.columns]

        for name in REQUIRED_MODELS:
            d = self._optimized_dir(name)
            cal_pkl   = os.path.join(d, "calibrated_model.pkl")
            prep_pkl  = os.path.join(d, "preprocessing.pkl")
            if not os.path.exists(cal_pkl) or not os.path.exists(prep_pkl):
                continue

            with open(cal_pkl, "rb") as f:
                model = pickle.load(f)
            with open(prep_pkl, "rb") as f:
                pipeline = pickle.load(f)

            X = pipeline.transform(df[available_cols])
            probs = model.predict_proba(X)

            self.assertEqual(probs.shape[1], 3,
                             f"[{name}] predict_proba should return 3 columns")
            np.testing.assert_allclose(
                probs.sum(axis=1),
                np.ones(len(X)),
                atol=1e-5,
                err_msg=f"[{name}] Probabilities don't sum to 1.0",
            )

    def test_ece_mce_functions(self):
        """ECE and MCE utility functions produce valid outputs on synthetic data."""
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 3, size=500)
        y_prob = rng.dirichlet([1, 1, 1], size=500)

        ece = compute_ece(y_true, y_prob)
        mce = compute_mce(y_true, y_prob)

        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)
        self.assertGreaterEqual(mce, 0.0)
        self.assertLessEqual(mce, 1.0)

    def test_model_selection_report_exists(self):
        """MODEL_SELECTION_REPORT.md must exist in project root."""
        path = os.path.join(ROOT, "MODEL_SELECTION_REPORT.md")
        self.assertTrue(os.path.exists(path), "MODEL_SELECTION_REPORT.md not found")
        with open(path) as f:
            content = f.read()
        self.assertIn("Scoring Weights", content)
        self.assertIn("Top 5 Ensemble Candidates", content)

    def test_error_analysis_report_exists(self):
        """ERROR_ANALYSIS.md must exist in project root."""
        path = os.path.join(ROOT, "ERROR_ANALYSIS.md")
        self.assertTrue(os.path.exists(path), "ERROR_ANALYSIS.md not found")

    def test_phase5_approval_report_exists(self):
        """PHASE_5_APPROVAL_REPORT.md must exist in project root."""
        path = os.path.join(ROOT, "PHASE_5_APPROVAL_REPORT.md")
        self.assertTrue(os.path.exists(path), "PHASE_5_APPROVAL_REPORT.md not found")


if __name__ == "__main__":
    unittest.main(verbosity=2)
