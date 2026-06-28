"""Helper script to sequentially run tournament simulations and warm up the matchup prediction cache on disk."""
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.prediction.knockout_engine import KnockoutEngine, map_team_name
from src.simulation.config import SimulationConfig
from src.simulation.probability_sampler import PredictionCache, ProbabilitySampler
from src.simulation.monte_carlo import run_tournament_simulation

def main():
    print("=" * 65)
    print("WORLDCUPAI — CACHE WARMUP")
    print("=" * 65)
    
    os.makedirs("predictions", exist_ok=True)
    cache_file = "predictions/matchup_cache.pkl"
    
    # Load existing cache if any
    cache_dict = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cache_dict = pickle.load(f)
            print(f"Loaded {len(cache_dict)} entries from existing cache.")
        except Exception as e:
            print(f"Failed to load existing cache: {e}")

    engine = KnockoutEngine()
    cache = PredictionCache(engine, cache_dict)
    sampler = ProbabilitySampler(cache)
    
    # Run 150 simulations sequentially to cover diverse bracket configurations
    print("Running 150 simulations to warm up prediction cache...")
    for i in range(150):
        seed = 5000 + i
        rng = np.random.default_rng(seed)
        run_tournament_simulation(engine, sampler, rng)
        if (i + 1) % 25 == 0:
            print(f"  Completed {i + 1}/150 simulations. Unique cache entries: {len(cache.cache)}")
            
    # Save the populated cache to disk
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(dict(cache.cache), f)
        print(f"Prediction cache saved to {cache_file} with {len(cache.cache)} entries.")
    except Exception as e:
        print(f"Failed to save cache: {e}")
        
    print("Warmup complete.")

if __name__ == "__main__":
    main()
