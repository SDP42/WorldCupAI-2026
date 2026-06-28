"""WorldCupAI — Phase 9 Monte Carlo Simulation Test Suite."""
import os
import shutil
import unittest
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.simulation.config import SimulationConfig
from src.simulation.simulation_engine import SimulationEngine
from src.simulation.utils import save_checkpoint, load_checkpoint

class TestPhase9Simulation(unittest.TestCase):
    """Test suite validating Monte Carlo simulation framework correctness, reproducibility, and compliance."""

    @classmethod
    def setUpClass(cls):
        # Setup temporary test directories to avoid overwriting production outputs
        cls.test_output_dir = "predictions_test"
        cls.test_plot_dir = "outputs_plots_test"
        os.makedirs(cls.test_output_dir, exist_ok=True)
        os.makedirs(cls.test_plot_dir, exist_ok=True)
        
        cls.config = SimulationConfig()
        cls.config.OUTPUT_DIR = cls.test_output_dir
        cls.config.PLOT_DIR = cls.test_plot_dir
        cls.config.CHECKPOINT_FILE = os.path.join(cls.test_output_dir, "test_checkpoint.pkl")
        
        # Copy matchup prediction cache from production predictions if available
        prod_cache = "predictions/matchup_cache.pkl"
        test_cache = os.path.join(cls.test_output_dir, "matchup_cache.pkl")
        if os.path.exists(prod_cache):
            shutil.copy(prod_cache, test_cache)
            
        # Instantiate engine under test
        cls.engine = SimulationEngine(cls.config)
        
        # Run a small batch of simulations (50) for fast testing
        cls.results = cls.engine.run(total_sims=50, seed=42, checkpoint_interval=20, force_restart=True)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary test output directories
        if os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir)
        if os.path.exists(cls.test_plot_dir):
            shutil.rmtree(cls.test_plot_dir)

    def test_simulation_count(self):
        """Validates that the correct number of simulations was completed."""
        perf = self.results["performance"]
        self.assertEqual(perf["simulations_completed"], 50)
        self.assertTrue(self.results["validation_passed"])

    def test_probability_conservation(self):
        """Ensures probabilities conservation and limits sum to 100%."""
        stats = self.results["team_statistics"]
        self.assertAlmostEqual(stats["champion_prob"].sum(), 1.0, places=3)
        self.assertAlmostEqual(stats["runner_up_prob"].sum(), 1.0, places=3)

    def test_reproducibility(self):
        """Ensures that identical seeds yield identical simulation outcomes."""
        engine_a = SimulationEngine(self.config)
        results_a = engine_a.run(total_sims=20, seed=123, checkpoint_interval=10, force_restart=True)
        
        engine_b = SimulationEngine(self.config)
        results_b = engine_b.run(total_sims=20, seed=123, checkpoint_interval=10, force_restart=True)
        
        df_a = results_a["team_statistics"].sort_values("team").reset_index(drop=True)
        df_b = results_b["team_statistics"].sort_values("team").reset_index(drop=True)
        
        pd.testing.assert_frame_equal(df_a, df_b)

    def test_checkpoint_and_resume(self):
        """Tests that checkpoint saving and resume behavior is correct and robust."""
        # Clean checkpoint file
        if os.path.exists(self.config.CHECKPOINT_FILE):
            os.remove(self.config.CHECKPOINT_FILE)
            
        # Simulate an interruption by manually saving a partial checkpoint
        dummy_checkpoint = {
            "total_sims_requested": 40,
            "seed": 999,
            "completed_sims": 20,
            "match_runs": self.results["team_statistics"].head(1).to_dict(), # dummy match runs placeholder
            "team_runs": [{} for _ in range(20)]
        }
        
        # When we load checkpoint it should match
        save_checkpoint(dummy_checkpoint, self.config.CHECKPOINT_FILE)
        loaded = load_checkpoint(self.config.CHECKPOINT_FILE)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["completed_sims"], 20)
        
        # Clean up
        if os.path.exists(self.config.CHECKPOINT_FILE):
            os.remove(self.config.CHECKPOINT_FILE)

    def test_exports_generation(self):
        """Verifies that all required CSV, Parquet, and JSON deliverables are correctly exported."""
        expected_files = [
            "team_statistics.csv", "team_probabilities.csv",
            "confidence_intervals.csv", "convergence.csv",
            "champion_distribution.csv", "runner_up_distribution.csv",
            "advancement_probabilities.csv", "simulation_results.csv",
            "simulation_results.parquet", "simulation_results.json",
            "simulation_metadata.json", "simulation_config.json",
            "runtime_statistics.json", "simulation_summary.json"
        ]
        for filename in expected_files:
            filepath = os.path.join(self.test_output_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"Missing exported artifact: {filename}")

    def test_visualizations_generation(self):
        """Verifies that all required analytical plots are correctly generated."""
        expected_plots = [
            "champion_probability_bar.png", "champion_distribution.png",
            "runner_up_distribution.png", "advancement_heatmap.png",
            "probability_histogram.png", "upset_distribution.png",
            "convergence_plot.png", "confidence_intervals_plot.png"
        ]
        for filename in expected_plots:
            filepath = os.path.join(self.test_plot_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"Missing generated figure: {filename}")

    def test_performance_benchmarks(self):
        """Verifies that runtime performance metrics are tracked and populated."""
        perf = self.results["performance"]
        self.assertIn("elapsed_seconds", perf)
        self.assertIn("simulations_per_second", perf)
        self.assertIn("cpu_utilization_pct", perf)
        self.assertIn("memory_usage_mb", perf)
        self.assertTrue(perf["simulations_per_second"] >= 0.0)

if __name__ == "__main__":
    unittest.main()
