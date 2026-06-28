#!/usr/bin/env python3
"""WorldCupAI — Phase 7 Test Suite

Validates all ensemble artifacts, configuration, metrics, and predictions
produced by train_ensemble.py.

Run:
    python3 tests/test_phase7.py
"""
import os
import sys
import json
import pickle
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ENSEMBLE_DIR = "models/ensemble"
PREDICTIONS_DIR = "predictions"


class TestEnsembleArtifacts(unittest.TestCase):
    """Verifies that all expected Phase 7 files are present."""

    def test_ensemble_dir_exists(self):
        self.assertTrue(os.path.isdir(ENSEMBLE_DIR), "models/ensemble/ directory missing")

    def test_ensemble_pkl_exists_and_loadable(self):
        path = os.path.join(ENSEMBLE_DIR, "ensemble.pkl")
        self.assertTrue(os.path.exists(path), "ensemble.pkl missing")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.assertIn("stacking_model",      obj, "stacking_model missing from ensemble.pkl")
        self.assertIn("blending_model",      obj, "blending_model missing from ensemble.pkl")
        self.assertIn("optimized_weights",   obj, "optimized_weights missing from ensemble.pkl")
        self.assertIn("candidates",          obj, "candidates missing from ensemble.pkl")
        self.assertIn("best_method",         obj, "best_method missing from ensemble.pkl")

    def test_ensemble_config_valid(self):
        path = os.path.join(ENSEMBLE_DIR, "ensemble_config.json")
        self.assertTrue(os.path.exists(path), "ensemble_config.json missing")
        with open(path) as f:
            cfg = json.load(f)
        self.assertIn("candidates",  cfg, "candidates key missing")
        self.assertIn("best_method", cfg, "best_method key missing")
        self.assertIn("n_models",    cfg, "n_models key missing")
        self.assertGreaterEqual(cfg["n_models"], 2, "ensemble must have at least 2 models")

    def test_optimized_weights_sum_to_one(self):
        path = os.path.join(ENSEMBLE_DIR, "optimized_weights.json")
        self.assertTrue(os.path.exists(path), "optimized_weights.json missing")
        with open(path) as f:
            weights = json.load(f)
        total = sum(weights.values())
        self.assertAlmostEqual(total, 1.0, places=4, msg=f"Weights sum to {total:.5f}, expected ~1.0")

    def test_optimized_weights_non_negative(self):
        with open(os.path.join(ENSEMBLE_DIR, "optimized_weights.json")) as f:
            weights = json.load(f)
        for name, w in weights.items():
            self.assertGreaterEqual(w, -1e-6, f"Negative weight for {name}: {w}")


class TestEnsemblePredictions(unittest.TestCase):
    """Validates the best-ensemble predictions.csv."""

    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(ENSEMBLE_DIR, "predictions.csv")

    def test_predictions_csv_exists(self):
        self.assertTrue(os.path.exists(self.path), "models/ensemble/predictions.csv missing")

    def test_predictions_required_columns(self):
        df = pd.read_csv(self.path)
        required = ["match_id", "true_label", "predicted_label",
                    "prob_away_win", "prob_draw", "prob_home_win"]
        for c in required:
            self.assertIn(c, df.columns, f"Column '{c}' missing from predictions.csv")

    def test_predictions_prob_sums_to_one(self):
        df = pd.read_csv(self.path)
        prob_sum = (df["prob_away_win"] + df["prob_draw"] + df["prob_home_win"])
        self.assertTrue((prob_sum.between(0.999, 1.001)).all(), "Probabilities do not sum to 1.0")

    def test_predictions_labels_valid(self):
        df = pd.read_csv(self.path)
        valid_labels = {0, 1, 2}
        self.assertTrue(set(df["true_label"].unique()).issubset(valid_labels))
        self.assertTrue(set(df["predicted_label"].unique()).issubset(valid_labels))

    def test_predictions_not_empty(self):
        df = pd.read_csv(self.path)
        self.assertGreater(len(df), 0, "predictions.csv is empty")


class TestEnsembleMetrics(unittest.TestCase):
    """Validates metrics.json produced by EnsembleEvaluator."""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(ENSEMBLE_DIR, "metrics.json")
        with open(path) as f:
            cls.metrics = json.load(f)

    def test_all_ensemble_methods_present(self):
        expected = ["Hard Voting", "Soft Voting", "Weighted Soft Voting", "Stacking", "Blending"]
        for name in expected:
            self.assertIn(name, self.metrics, f"'{name}' missing from metrics.json")

    def test_all_required_metric_keys(self):
        required_keys = ["accuracy", "roc_auc_macro", "pr_auc_macro",
                         "log_loss", "f1_macro", "brier_score"]
        for method, m in self.metrics.items():
            for k in required_keys:
                self.assertIn(k, m, f"Key '{k}' missing from {method} metrics")

    def test_accuracy_in_valid_range(self):
        for method, m in self.metrics.items():
            acc = m["accuracy"]
            self.assertGreaterEqual(acc, 0.0, f"{method} accuracy < 0")
            self.assertLessEqual(acc, 1.0,    f"{method} accuracy > 1")

    def test_roc_auc_in_valid_range(self):
        for method, m in self.metrics.items():
            auc = m["roc_auc_macro"]
            self.assertGreaterEqual(auc, 0.0, f"{method} ROC-AUC < 0")
            self.assertLessEqual(auc, 1.0,    f"{method} ROC-AUC > 1")

    def test_log_loss_positive(self):
        for method, m in self.metrics.items():
            self.assertGreater(m["log_loss"], 0.0, f"{method} log_loss <= 0")

    def test_ensemble_beats_random_baseline(self):
        """Each ensemble should beat a random classifier (ACC > 0.33)."""
        for method, m in self.metrics.items():
            self.assertGreater(m["accuracy"], 0.33,
                               f"{method} accuracy {m['accuracy']:.3f} ≤ random baseline")


class TestDiversityMatrices(unittest.TestCase):
    """Validates diversity analysis CSV outputs."""

    def _load_matrix(self, filename):
        path = os.path.join(ENSEMBLE_DIR, filename)
        self.assertTrue(os.path.exists(path), f"{filename} missing")
        return pd.read_csv(path, index_col=0)

    def test_correlation_matrix_exists(self):
        df = self._load_matrix("probability_correlations.csv")
        self.assertGreater(df.shape[0], 1)

    def test_agreement_matrix_exists(self):
        df = self._load_matrix("prediction_agreements.csv")
        self.assertGreater(df.shape[0], 1)

    def test_kappa_matrix_exists(self):
        df = self._load_matrix("cohens_kappa.csv")
        self.assertGreater(df.shape[0], 1)

    def test_error_overlap_matrix_exists(self):
        df = self._load_matrix("error_overlaps.csv")
        self.assertGreater(df.shape[0], 1)

    def test_heatmaps_saved(self):
        heatmaps = [
            "heatmap_correlation.png",
            "heatmap_agreement.png",
            "heatmap_kappa.png",
            "heatmap_error_overlap.png",
        ]
        for h in heatmaps:
            path = os.path.join(ENSEMBLE_DIR, h)
            self.assertTrue(os.path.exists(path), f"{h} not found")
            self.assertGreater(os.path.getsize(path), 1024, f"{h} suspiciously small")


class TestDocumentation(unittest.TestCase):
    """Validates all Phase 7 documentation files."""

    def test_readme_phase7_exists(self):
        self.assertTrue(os.path.exists("README_PHASE7.md"), "README_PHASE7.md missing")

    def test_readme_ensemble_exists(self):
        self.assertTrue(os.path.exists("README_ENSEMBLE.md"), "README_ENSEMBLE.md missing")

    def test_ensemble_comparison_exists(self):
        self.assertTrue(os.path.exists("ENSEMBLE_COMPARISON.md"), "ENSEMBLE_COMPARISON.md missing")

    def test_ensemble_selection_report_exists(self):
        self.assertTrue(os.path.exists("ENSEMBLE_SELECTION_REPORT.md"), "ENSEMBLE_SELECTION_REPORT.md missing")

    def test_phase7_approval_report_exists(self):
        self.assertTrue(os.path.exists("PHASE_7_APPROVAL_REPORT.md"), "PHASE_7_APPROVAL_REPORT.md missing")

    def test_changelog_updated(self):
        self.assertTrue(os.path.exists("CHANGELOG.md"), "CHANGELOG.md missing")
        with open("CHANGELOG.md") as f:
            content = f.read()
        self.assertIn("Phase 7", content, "Phase 7 not mentioned in CHANGELOG.md")

    def test_calibration_json_exists(self):
        path = os.path.join(ENSEMBLE_DIR, "calibration.json")
        self.assertTrue(os.path.exists(path), "calibration.json missing")
        with open(path) as f:
            cal = json.load(f)
        self.assertIn("ensemble", cal)
        self.assertIn("brier_score", cal)


class TestValPredictionCSVs(unittest.TestCase):
    """Validates that validation prediction CSVs were produced by export_val.py."""

    ML_NAMES = ["xgboost", "gradient_boosting", "random_forest", "extra_trees", "logistic_regression"]

    def test_val_csv_files_exist(self):
        for name in self.ML_NAMES:
            path = os.path.join(PREDICTIONS_DIR, f"val_{name}_predictions.csv")
            self.assertTrue(os.path.exists(path), f"val_{name}_predictions.csv missing")

    def test_val_csv_columns(self):
        required = ["match_id", "true_label", "predicted_label",
                    "prob_away_win", "prob_draw", "prob_home_win"]
        for name in self.ML_NAMES:
            path = os.path.join(PREDICTIONS_DIR, f"val_{name}_predictions.csv")
            if not os.path.exists(path):
                self.skipTest(f"Val CSV for {name} missing — skipping column check")
            df = pd.read_csv(path)
            for c in required:
                self.assertIn(c, df.columns, f"Column '{c}' missing from val_{name}_predictions.csv")

    def test_val_csv_prob_sums_to_one(self):
        for name in self.ML_NAMES:
            path = os.path.join(PREDICTIONS_DIR, f"val_{name}_predictions.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            s  = df["prob_away_win"] + df["prob_draw"] + df["prob_home_win"]
            self.assertTrue(s.between(0.998, 1.002).all(),
                            f"val_{name} probabilities don't sum to 1.0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
