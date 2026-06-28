#!/usr/bin/env python3
"""WorldCupAI — Phase 6 Automated Tests

Validates all Phase 6 deep learning artifacts:
  - models/ann/ and models/lstm/ directories + required files
  - model_config.json has correct keys
  - training_history.json has at least 5 epochs
  - metrics.json has all required metric keys
  - predictions/ CSVs exist with correct columns
  - Loaded ANN/LSTM produce valid probabilities
  - DL_VS_ML_COMPARISON.md and PHASE_6_APPROVAL_REPORT.md exist
"""
import os
import sys
import json
import pickle
import unittest
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

REQUIRED_MODEL_DIRS = {
    "ann":  "models/ann",
    "lstm": "models/lstm",
}

REQUIRED_FILES_PER_DIR = [
    "model.pt",
    "model_complete.pt",
    "model_config.json",
    "preprocessing.pkl",
    "metrics.json",
    "training_history.json",
    "training_log.json",
    "confusion_matrix.png",
    "roc_curve.png",
    "learning_curves.png",
]

REQUIRED_METRIC_KEYS = [
    "accuracy", "roc_auc_macro", "pr_auc_macro",
    "log_loss", "f1_macro", "brier_score", "ece",
]

PREDICTION_FILES = [
    "ann_predictions.csv",
    "lstm_predictions.csv",
    "xgboost_predictions.csv",
    "gradient_boosting_predictions.csv",
    "random_forest_predictions.csv",
]

PREDICTION_COLS = [
    "match_id", "true_label", "predicted_label",
    "prob_away_win", "prob_draw", "prob_home_win",
]


class TestPhase6Artifacts(unittest.TestCase):

    def _path(self, *parts) -> str:
        return os.path.join(ROOT, *parts)

    # ── Directory & file existence ────────────────────────────────────────────
    def test_model_directories_exist(self):
        for name, rel_dir in REQUIRED_MODEL_DIRS.items():
            full = self._path(rel_dir)
            self.assertTrue(os.path.isdir(full), f"Missing model dir: {full}")

    def test_required_files_exist(self):
        for name, rel_dir in REQUIRED_MODEL_DIRS.items():
            for fname in REQUIRED_FILES_PER_DIR:
                fpath = self._path(rel_dir, fname)
                self.assertTrue(os.path.exists(fpath),
                                f"[{name.upper()}] Missing file: {fpath}")

    # ── model_config.json ─────────────────────────────────────────────────────
    def test_model_config_keys(self):
        required_keys = ["model_name", "framework", "input_dim", "n_classes",
                         "optimizer", "lr", "batch_size"]
        for name, rel_dir in REQUIRED_MODEL_DIRS.items():
            cfg_path = self._path(rel_dir, "model_config.json")
            if not os.path.exists(cfg_path):
                continue
            with open(cfg_path) as f:
                cfg = json.load(f)
            for key in required_keys:
                self.assertIn(key, cfg,
                              f"[{name.upper()}] model_config.json missing key: {key}")

    # ── training_history.json ─────────────────────────────────────────────────
    def test_training_history_nonempty(self):
        for name, rel_dir in REQUIRED_MODEL_DIRS.items():
            hist_path = self._path(rel_dir, "training_history.json")
            if not os.path.exists(hist_path):
                continue
            with open(hist_path) as f:
                hist = json.load(f)
            self.assertGreaterEqual(len(hist), 5,
                                    f"[{name.upper()}] training_history has < 5 epochs")
            self.assertIn("train_loss", hist[0],
                          f"[{name.upper()}] training_history missing 'train_loss'")
            self.assertIn("val_loss",   hist[0],
                          f"[{name.upper()}] training_history missing 'val_loss'")

    # ── metrics.json ──────────────────────────────────────────────────────────
    def test_metrics_json_valid(self):
        for name, rel_dir in REQUIRED_MODEL_DIRS.items():
            m_path = self._path(rel_dir, "metrics.json")
            if not os.path.exists(m_path):
                continue
            with open(m_path) as f:
                m = json.load(f)
            for key in REQUIRED_METRIC_KEYS:
                self.assertIn(key, m,
                              f"[{name.upper()}] metrics.json missing key: {key}")
                self.assertIsInstance(m[key], (int, float),
                                      f"[{name.upper()}] {key} is not numeric")

    # ── Prediction CSVs ───────────────────────────────────────────────────────
    def test_prediction_csvs_exist(self):
        for fname in PREDICTION_FILES:
            fpath = self._path("predictions", fname)
            self.assertTrue(os.path.exists(fpath),
                            f"Prediction CSV not found: {fpath}")

    def test_prediction_csv_columns(self):
        for fname in PREDICTION_FILES:
            fpath = self._path("predictions", fname)
            if not os.path.exists(fpath):
                continue
            df = pd.read_csv(fpath)
            for col in PREDICTION_COLS:
                self.assertIn(col, df.columns,
                              f"[{fname}] missing column: {col}")
            # Probabilities should sum to ~1.0
            prob_sum = (df["prob_away_win"] + df["prob_draw"] + df["prob_home_win"])
            np.testing.assert_allclose(
                prob_sum.values,
                np.ones(len(df)),
                atol=1e-4,
                err_msg=f"[{fname}] probabilities don't sum to 1.0",
            )

    # ── Loaded model inference ────────────────────────────────────────────────
    def test_ann_model_inference(self):
        from src.deep_learning.ann_model import ANNModel
        cfg_path   = self._path("models/ann/model_config.json")
        state_path = self._path("models/ann/model.pt")
        if not (os.path.exists(cfg_path) and os.path.exists(state_path)):
            self.skipTest("ANN model files not found")

        with open(cfg_path) as f:
            cfg = json.load(f)

        model = ANNModel(
            input_dim=cfg["input_dim"],
            hidden_dims=cfg["hidden_dims"],
            dropout_rates=cfg["dropout_rates"],
            n_classes=cfg["n_classes"],
        )
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
        model.eval()

        X = torch.randn(16, cfg["input_dim"])
        probs = model.predict_proba(X)
        self.assertEqual(probs.shape, (16, 3))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(16), atol=1e-5)

    def test_lstm_model_inference(self):
        from src.deep_learning.lstm_model import LSTMModel
        cfg_path   = self._path("models/lstm/model_config.json")
        state_path = self._path("models/lstm/model.pt")
        if not (os.path.exists(cfg_path) and os.path.exists(state_path)):
            self.skipTest("LSTM model files not found")

        with open(cfg_path) as f:
            cfg = json.load(f)

        model = LSTMModel(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            lstm_dropout=cfg["lstm_dropout"],
            fc_dropout=cfg["fc_dropout"],
            seq_len=cfg["seq_len"],
        )
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
        model.eval()

        X = torch.randn(16, cfg["seq_len"], cfg["input_dim"])
        probs = model.predict_proba(X)
        self.assertEqual(probs.shape, (16, 3))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(16), atol=1e-5)

    # ── Documentation ─────────────────────────────────────────────────────────
    def test_comparison_report_exists(self):
        path = self._path("DL_VS_ML_COMPARISON.md")
        self.assertTrue(os.path.exists(path), "DL_VS_ML_COMPARISON.md not found")
        with open(path) as f:
            content = f.read()
        self.assertIn("ANN", content)
        self.assertIn("XGBoost", content)

    def test_approval_report_exists(self):
        path = self._path("PHASE_6_APPROVAL_REPORT.md")
        self.assertTrue(os.path.exists(path), "PHASE_6_APPROVAL_REPORT.md not found")

    def test_phase6_readme_exists(self):
        for fname in ["README_PHASE6.md", "README_ANN.md", "README_LSTM.md",
                      "README_DEEP_LEARNING.md", "DEEP_LEARNING_BENCHMARK.md"]:
            self.assertTrue(os.path.exists(self._path(fname)),
                            f"{fname} not found")


if __name__ == "__main__":
    unittest.main(verbosity=2)
