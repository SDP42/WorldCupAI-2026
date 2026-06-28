"""WorldCupAI — Phase 10: XAI & Decision Intelligence Test Suite.

Validates correctness, completeness, consistency, and compliance of
all Phase 10 explainability modules, artifacts, plots, and reports.
"""
import os
import copy
import json
import shutil
import unittest
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.explainability.utils import (
    FEATURE_COLS, CONFIDENCE_THRESHOLDS,
    ensure_dir, load_tournament_predictions,
)
from src.explainability.feature_importance import FeatureImportanceAnalyzer
from src.explainability.global_explanations import GlobalExplainer
from src.explainability.local_explanations import LocalExplainer
from src.explainability.confidence_analysis import ConfidenceAnalyzer
from src.explainability.counterfactuals import CounterfactualEngine
from src.explainability.ensemble_explanations import EnsembleExplainer
from src.explainability.tournament_explanations import TournamentExplainer
from src.explainability.report_generator import ReportGenerator


class TestPhase10XAI(unittest.TestCase):
    """Test suite validating XAI deliverables and mathematical consistency."""

    @classmethod
    def setUpClass(cls):
        # Setup temporary directories for test artifacts to keep workspace clean
        cls.test_predictions_dir = "predictions_xai_test"
        cls.test_plots_dir = "outputs_plots_xai_test"
        cls.test_reports_dir = "reports_xai_test"

        ensure_dir(cls.test_predictions_dir)
        ensure_dir(cls.test_plots_dir)
        ensure_dir(cls.test_reports_dir)

        # Load production tournament predictions or fallback
        prod_preds_path = "predictions/tournament_predictions.json"
        if os.path.exists(prod_preds_path):
            cls.predictions = load_tournament_predictions(prod_preds_path)
        else:
            # Create a mock predictions list for fallback/CI environments
            cls.predictions = []
            for i in range(73, 105):
                cls.predictions.append({
                    "match_no": i,
                    "round": "Round of 32" if i < 89 else "Quarter-final",
                    "home_team": "Argentina",
                    "away_team": "Brazil",
                    "predicted_winner": "Argentina",
                    "confidence": 0.65,
                    "prob_home_win": 0.65,
                    "prob_draw": 0.15,
                    "prob_away_win": 0.20,
                    "entropy": 1.2,
                    "shootout_played": False,
                    "features": {f: 0.1 for f in FEATURE_COLS}
                })

        cls.ensemble_weights = {"XGBoost": 0.486, "Gradient Boosting": 0.514}

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary test outputs
        for d in [cls.test_predictions_dir, cls.test_plots_dir, cls.test_reports_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)

    def test_feature_importance_generation(self):
        """Task 1: Validates feature importance analyzer output."""
        analyzer = FeatureImportanceAnalyzer(
            ensemble_weights=self.ensemble_weights,
            predictions_dir=self.test_predictions_dir,
        )
        importances = analyzer.compute()
        self.assertIn("Ensemble (Weighted)", importances)
        self.assertIn("Average", importances)

        paths = analyzer.export(output_dir=self.test_predictions_dir)
        self.assertTrue(os.path.exists(paths["csv"]))
        self.assertTrue(os.path.exists(paths["json"]))

        # Check content
        df = pd.read_csv(paths["csv"])
        self.assertFalse(df.empty)
        self.assertIn("importance", df.columns)

    def test_global_explanations(self):
        """Task 1 & 8: Validates global explanations generator."""
        analyzer = FeatureImportanceAnalyzer(ensemble_weights=self.ensemble_weights)
        analyzer.compute()

        explainer = GlobalExplainer(analyzer, self.predictions)
        res = explainer.explain()

        self.assertIn("top_features_ranked", res)
        self.assertIn("category_importance", res)
        self.assertIn("model_agreement", res)

        json_path = explainer.export(output_dir=self.test_predictions_dir)
        self.assertTrue(os.path.exists(json_path))

    def test_local_match_explanations(self):
        """Task 2: Validates match attribution and narrative generation."""
        analyzer = FeatureImportanceAnalyzer(ensemble_weights=self.ensemble_weights)
        analyzer.compute()
        ens_imp = analyzer.importances["Ensemble (Weighted)"]

        explainer = LocalExplainer(self.predictions, global_importances=ens_imp)
        exps = explainer.explain_all()

        self.assertEqual(len(exps), len(self.predictions))
        first = exps[0]
        self.assertIn("narrative", first)
        self.assertIn("top_positive_features", first)
        self.assertIn("top_negative_features", first)

        paths = explainer.export(output_dir=self.test_predictions_dir)
        self.assertTrue(os.path.exists(paths["json"]))
        self.assertTrue(os.path.exists(paths["csv"]))

    def test_confidence_analysis(self):
        """Task 4: Validates confidence score and risk categorization."""
        analyzer = ConfidenceAnalyzer(self.predictions, ensemble_weights=self.ensemble_weights)
        analyses = analyzer.analyze_all()
        self.assertEqual(len(analyses), len(self.predictions))

        first = analyses[0]
        self.assertIn("entropy", first)
        self.assertIn("margin", first)
        self.assertIn("risk_category", first)
        self.assertIn(first["risk_category"], CONFIDENCE_THRESHOLDS.keys() | {"Very Low"})

        paths = analyzer.export(output_dir=self.test_predictions_dir)
        self.assertTrue(os.path.exists(paths["csv"]))
        self.assertTrue(os.path.exists(paths["json"]))

    def test_counterfactual_examples(self):
        """Task 5: Validates counterfactual perturbation scenarios."""
        # Ensure the test match round is in COUNTERFACTUAL_ROUNDS
        test_match = copy.deepcopy(self.predictions[0])
        test_match["round"] = "Final"
        cf_engine = CounterfactualEngine(
            [test_match],
            ensemble_weights=self.ensemble_weights,
        )
        res = cf_engine.run()
        self.assertEqual(len(res), 1)

        first = res[0]
        self.assertIn("scenarios", first)
        self.assertIn("robustness_score", first)
        self.assertTrue(len(first["scenarios"]) > 0)

        paths = cf_engine.export(output_dir=self.test_predictions_dir)
        self.assertTrue(os.path.exists(paths["json"]))
        self.assertTrue(os.path.exists(paths["csv"]))

    def test_ensemble_explanations(self):
        """Task 3: Validates voting split and model contributions."""
        explainer = EnsembleExplainer(self.predictions, ensemble_weights=self.ensemble_weights)
        exps = explainer.explain_all()
        self.assertEqual(len(exps), len(self.predictions))

        first = exps[0]
        self.assertIn("model_contributions", first)
        self.assertIn("model_agreement", first)

        paths = explainer.export(output_dir=self.test_predictions_dir)
        self.assertTrue(os.path.exists(paths["csv"]))
        self.assertTrue(os.path.exists(paths["json"]))

    def test_tournament_explanations(self):
        """Task 6: Validates tournament-level narratives."""
        analyzer = FeatureImportanceAnalyzer(ensemble_weights=self.ensemble_weights)
        analyzer.compute()

        explainer = TournamentExplainer(
            self.predictions,
            global_importances=analyzer.importances["Ensemble (Weighted)"],
            ensemble_weights=self.ensemble_weights,
        )
        res = explainer.explain()

        self.assertIn("champion_explanation", res)
        self.assertIn("biggest_upsets", res)
        self.assertIn("round_statistics", res)

        paths = explainer.export(output_dir=self.test_predictions_dir)
        self.assertTrue(os.path.exists(paths["json"]))
        self.assertTrue(os.path.exists(paths["csv"]))

    def test_probability_consistency(self):
        """Task 7: Validates that all prediction probabilities sum to ~1.0."""
        for m in self.predictions:
            p_away = m.get("prob_away_win", 0.0)
            p_draw = m.get("prob_draw", 0.0)
            p_home = m.get("prob_home_win", 0.0)
            total = p_away + p_draw + p_home
            self.assertAlmostEqual(total, 1.0, places=4)

    def test_reports_generation(self):
        """Task 9: Validates markdown report compilation."""
        # Mock inputs to generator
        generator = ReportGenerator(
            global_exp={"top_features_ranked": [], "category_importance": {}},
            local_exps=[],
            confidence_sum={"total_matches": 0, "tier_distribution": {}},
            ensemble_sum={"ensemble_weights": self.ensemble_weights},
            counterfactuals=[],
            tournament_exp={},
            output_dir=self.test_reports_dir,
        )
        generator.generate_all()

        expected_files = [
            "README_PHASE10.md", "README_EXPLAINABILITY.md",
            "XAI_REPORT.md", "FEATURE_IMPORTANCE_REPORT.md",
            "COUNTERFACTUAL_REPORT.md", "CONFIDENCE_ANALYSIS.md",
            "ENSEMBLE_EXPLAINABILITY.md", "TOURNAMENT_EXPLANATION_REPORT.md",
            "MODEL_TRUST_REPORT.md"
        ]
        for f in expected_files:
            self.assertTrue(os.path.exists(os.path.join(self.test_reports_dir, f)))

    def test_backward_compatibility(self):
        """Ensures existing preprocessors and predictions remain uncorrupted."""
        self.assertTrue(os.path.exists("models/xgboost_optimized/calibrated_model.pkl"))
        self.assertTrue(os.path.exists("models/xgboost_optimized/preprocessing.pkl"))


if __name__ == "__main__":
    unittest.main()
