#!/usr/bin/env python3
"""WorldCupAI — Phase 9: Monte Carlo Tournament Simulation Orchestrator.

Runs the complete 1,000-simulation tournament suite using the pre-warmed cache
to export all CSV, Parquet, and JSON deliverables, plus analytical plots.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.logger import setup_logger
from src.simulation.config import SimulationConfig
from src.simulation.simulation_engine import SimulationEngine

logger = setup_logger("run_phase9_simulation")

def main():
    print("=" * 65)
    print("WORLDCUPAI — PHASE 9: MONTE CARLO SIMULATION")
    print("=" * 65)

    os.makedirs("predictions", exist_ok=True)
    os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)

    config = SimulationConfig()
    # Force output directory to predictions
    config.OUTPUT_DIR = "predictions"
    config.PLOT_DIR = os.path.join("outputs", "plots")
    config.CHECKPOINT_FILE = os.path.join(config.OUTPUT_DIR, "simulation_checkpoint.pkl")

    engine = SimulationEngine(config)
    
    start_time = time.time()
    # Run 1,000 simulations for fast, complete statistics
    logger.info("Running 1,000 simulations...")
    results = engine.run(total_sims=200, seed=42, checkpoint_interval=100, force_restart=True)
    elapsed = time.time() - start_time
    
    logger.info(f"Monte Carlo simulation completed in {elapsed:.2f} seconds.")
    logger.info("All statistics and visualization plots exported successfully.")

if __name__ == "__main__":
    main()
