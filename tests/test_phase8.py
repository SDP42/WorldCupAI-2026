#!/usr/bin/env python3
"""WorldCupAI — Phase 8 Test Suite

Validates fixture loading, bracket progression, winner advancement,
probability conservation, export generation, and deterministic execution.
"""
import os
import sys
import json
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prediction.knockout_engine import KnockoutEngine, map_team_name


class TestKnockoutEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = KnockoutEngine(
            fixtures_path="configs/knockout_fixtures.json",
            ensemble_pkl_path="models/ensemble/ensemble.pkl",
            feature_store_path="processed/feature_store.parquet",
            master_dataset_path="processed/master_dataset.parquet"
        )
        cls.results = cls.engine.run_tournament()

    def test_fixtures_loaded_correctly(self):
        self.assertEqual(len(self.engine.round_32_fixtures), 16)
        for f in self.engine.round_32_fixtures:
            self.assertIn("match_no", f)
            self.assertIn("round", f)
            self.assertIn("home_team", f)
            self.assertIn("away_team", f)

    def test_complete_bracket_played(self):
        # Must have exactly 32 matches predicted: 73 to 104 inclusive
        self.assertEqual(len(self.results), 32)
        match_keys = sorted(list(self.results.keys()))
        self.assertEqual(match_keys, list(range(73, 105)))

    def test_probability_conservation(self):
        for m_no, m in self.results.items():
            prob_sum = m["prob_home_win"] + m["prob_draw"] + m["prob_away_win"]
            self.assertAlmostEqual(prob_sum, 1.0, places=5, msg=f"Match {m_no} probabilities do not sum to 1")

    def test_shootout_flag_and_draw_outcomes(self):
        for m_no, m in self.results.items():
            is_draw = (m["predicted_outcome"] == "Draw")
            self.assertEqual(m["shootout_played"], is_draw, f"Match {m_no} shootout flag mismatch")
            if is_draw:
                self.assertTrue(len(m["shootout_reason"]) > 0)
                self.assertIn(m["predicted_winner"], [m["home_team"], m["away_team"]])

    def test_winner_progression_integrity(self):
        # Round of 16 Match 89 = Winner Match 73 vs Winner Match 75
        w_73 = self.results[73]["predicted_winner"]
        w_75 = self.results[75]["predicted_winner"]
        self.assertEqual(self.results[89]["home_team"], w_73)
        self.assertEqual(self.results[89]["away_team"], w_75)

        # Match 104 (Final) = Winner Match 101 vs Winner Match 102
        w_101 = self.results[101]["predicted_winner"]
        w_102 = self.results[102]["predicted_winner"]
        self.assertEqual(self.results[104]["home_team"], w_101)
        self.assertEqual(self.results[104]["away_team"], w_102)

        # Match 103 (Third Place Play-off) uses losers
        l_101_home = self.results[101]["home_team"]
        l_101_away = self.results[101]["away_team"]
        w_101_val = self.results[101]["predicted_winner"]
        l_101 = l_101_home if w_101_val != l_101_home else l_101_away

        l_102_home = self.results[102]["home_team"]
        l_102_away = self.results[102]["away_team"]
        w_102_val = self.results[102]["predicted_winner"]
        l_102 = l_102_home if w_102_val != l_102_home else l_102_away

        self.assertEqual(self.results[103]["home_team"], l_101)
        self.assertEqual(self.results[103]["away_team"], l_102)

    def test_output_files_exist_and_valid(self):
        # Check generated predictions folder exports
        self.assertTrue(os.path.exists("predictions/tournament_predictions.csv"))
        self.assertTrue(os.path.exists("predictions/tournament_predictions.json"))
        self.assertTrue(os.path.exists("predictions/tournament_predictions.parquet"))

        # Check dashboard JSON files
        self.assertTrue(os.path.exists("bracket.json"))
        self.assertTrue(os.path.exists("tournament_tree.json"))
        self.assertTrue(os.path.exists("prediction_summary.json"))

        # Verify tree matches bracket structure
        with open("tournament_tree.json") as f:
            tree = json.load(f)
        self.assertEqual(len(tree), 32)
        self.assertIn("match_104", tree)

    def test_documentation_generated(self):
        docs = [
            "README_PHASE8.md",
            "README_KNOCKOUT_ENGINE.md",
            "README_MATCH_PREDICTIONS.md",
            "TOURNAMENT_PREDICTION_REPORT.md",
            "MATCH_EXPLANATIONS.md",
            "PHASE_8_APPROVAL_REPORT.md"
        ]
        for d in docs:
            self.assertTrue(os.path.exists(d), f"Documentation file {d} is missing")
            self.assertGreater(os.path.getsize(d), 0, f"Documentation file {d} is empty")


class TestTeamNameMapping(unittest.TestCase):
    def test_team_mapping(self):
        self.assertEqual(map_team_name("Ivory Coast"), "Cote d'Ivoire")
        self.assertEqual(map_team_name("DR Congo"), "Democratic Republic of the Congo")
        self.assertEqual(map_team_name("Bosnia & Herzegovina"), "Bosnia and Herzegovina")
        self.assertEqual(map_team_name("Germany"), "Germany")


if __name__ == "__main__":
    unittest.main(verbosity=2)
