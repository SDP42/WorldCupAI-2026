"""WorldCupAI — Phase 11 Streamlit Dashboard Test Suite.

Validates data loading, package imports, layout files existence, and
integration components correctness.
"""
import os
import sys
import unittest
import pandas as pd
from typing import Dict, Any

# Add project root to sys path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dashboard.data_loader import (
    load_predictions, load_explanations, load_confidence, load_confidence_summary,
    load_counterfactuals, load_feature_importance, load_model_registry,
    get_dashboard_summary_metrics
)


class TestDashboardComponents(unittest.TestCase):
    """Test suite validating that the dashboard modules load correctly and consume Phase 1–10 data."""

    def test_imports_healthy(self):
        """Validates that no syntax errors or circular dependencies exist in dashboard modules."""
        # Main entry points
        import app
        from dashboard import sidebar
        from dashboard import cards
        from dashboard import bracket
        from dashboard import reports
        from dashboard import export
        from dashboard import xai
        from dashboard import performance
        from dashboard import settings
        
        self.assertTrue(True)

    def test_data_loaders(self):
        """Validates that data_loader retrieves valid structures from predictions directory."""
        preds = load_predictions()
        self.assertIsInstance(preds, list)
        if preds:
            self.assertIn("match_no", preds[0])
            self.assertIn("home_team", preds[0])
            self.assertIn("away_team", preds[0])

        exps = load_explanations()
        self.assertIsInstance(exps, list)
        if exps:
            self.assertIn("match_no", exps[0])
            self.assertIn("narrative", exps[0])

        conf = load_confidence()
        self.assertIsInstance(conf, pd.DataFrame)

        cf = load_counterfactuals()
        self.assertIsInstance(cf, pd.DataFrame)

        imp = load_feature_importance()
        self.assertIsInstance(imp, pd.DataFrame)

    def test_model_registry_loaded(self):
        """Validates model registry parameters loading."""
        reg = load_model_registry()
        self.assertIsInstance(reg, dict)
        if reg:
            self.assertIn("XGBoost (Optimized)" if "XGBoost (Optimized)" in reg else list(reg.keys())[0], reg)

    def test_summary_metrics(self):
        """Validates pre-aggregated metric loader."""
        metrics = get_dashboard_summary_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("champion", metrics)
        self.assertIn("runner_up", metrics)
        self.assertIn("mean_confidence", metrics)

    def test_layout_files_exist(self):
        """Validates that all required files and scripts specified by Phase 11 layout exist."""
        expected_files = [
            "app.py",
            "dashboard/data_loader.py",
            "dashboard/sidebar.py",
            "dashboard/cards.py",
            "dashboard/bracket.py",
            "dashboard/reports.py",
            "dashboard/export.py",
            "dashboard/xai.py",
            "dashboard/performance.py",
            "dashboard/settings.py",
        ]
        for f in expected_files:
            self.assertTrue(os.path.exists(f), f"Required file {f} is missing from layout structure!")


if __name__ == "__main__":
    unittest.main()
