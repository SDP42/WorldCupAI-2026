"""WorldCupAI — Phase 9 Simulation Orchestrator & Multiprocessing Engine."""
import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from multiprocessing import Pool

from src.utils.logger import setup_logger
from src.prediction.knockout_engine import KnockoutEngine, map_team_name
from src.simulation.config import SimulationConfig
from src.simulation.utils import PerformanceTracker, save_checkpoint, load_checkpoint, get_git_commit_hash
from src.simulation.probability_sampler import PredictionCache, ProbabilitySampler
from src.simulation.monte_carlo import run_tournament_simulation, analyze_simulation_run
from src.simulation.statistics import (
    calculate_team_statistics, calculate_tournament_statistics,
    calculate_uncertainty_analysis, calculate_convergence, validate_probabilities
)
from src.simulation.visualization import (
    plot_champion_probability_bar, plot_champion_distribution_donut,
    plot_runner_up_distribution_donut, plot_advancement_heatmap,
    plot_probability_histogram, plot_upset_distribution,
    plot_convergence, plot_confidence_intervals
)

logger = setup_logger("simulation_engine")

# ─────────────────────────────────────────────────────────────────────────────
# Worker Process Target
# ─────────────────────────────────────────────────────────────────────────────
_global_engine: Optional[KnockoutEngine] = None
_global_cache: Optional[PredictionCache] = None
_global_sampler: Optional[ProbabilitySampler] = None

def init_worker(shared_dict):
    """Initializes KnockoutEngine once per worker process and wires the shared dictionary."""
    global _global_engine, _global_cache, _global_sampler
    _global_engine = KnockoutEngine()
    _global_cache = PredictionCache(_global_engine, shared_dict)
    _global_sampler = ProbabilitySampler(_global_cache)

def run_sim_chunk_worker(args: Tuple[int, int, Any]) -> Tuple[List[Dict[int, Dict[str, Any]]], List[Dict[str, Dict[str, Any]]]]:
    """Runs a batch of simulations inside a worker process."""
    global _global_engine, _global_sampler
    num_sims, start_seed, shared_dict = args
    if _global_engine is None or _global_sampler is None:
        init_worker(shared_dict)
        
    match_runs = []
    team_runs = []
    
    for i in range(num_sims):
        # Ensure distinct but reproducible seed for each simulation run
        seed = int(start_seed + i)
        rng = np.random.default_rng(seed)
        
        match_res = run_tournament_simulation(_global_engine, _global_sampler, rng)
        team_res = analyze_simulation_run(match_res, SimulationConfig.TEAMS)
        
        match_runs.append(match_res)
        team_runs.append(team_res)
        
    return match_runs, team_runs

# ─────────────────────────────────────────────────────────────────────────────
# Orchestration Engine
# ─────────────────────────────────────────────────────────────────────────────
class SimulationEngine:
    """Manages full Monte Carlo bracket simulation runs, checkpointing, and exports."""
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.config.PLOT_DIR, exist_ok=True)
        self.tracker = PerformanceTracker()

    def run(self, total_sims: Optional[int] = None, seed: Optional[int] = None, 
            checkpoint_interval: Optional[int] = None, force_restart: bool = False) -> Dict[str, Any]:
        """Runs the complete Monte Carlo simulation orchestrator."""
        sims_to_run = total_sims or self.config.DEFAULT_SIMULATIONS
        base_seed = seed or self.config.DEFAULT_SEED
        chk_interval = checkpoint_interval or self.config.DEFAULT_CHECKPOINT_INTERVAL
        
        logger.info(f"Preparing Monte Carlo Simulation: sims={sims_to_run}, seed={base_seed}, workers={self.config.NUM_WORKERS}")

        # Try to resume from checkpoint
        checkpoint = None
        if not force_restart:
            checkpoint = load_checkpoint(self.config.CHECKPOINT_FILE)
            
        if checkpoint is not None and checkpoint.get("total_sims_requested") == sims_to_run and checkpoint.get("seed") == base_seed:
            completed_sims = checkpoint["completed_sims"]
            all_match_runs = checkpoint["match_runs"]
            all_team_runs = checkpoint["team_runs"]
            logger.info(f"Resuming simulation from checkpoint: {completed_sims}/{sims_to_run} completed.")
        else:
            completed_sims = 0
            all_match_runs = []
            all_team_runs = []
            if force_restart and os.path.exists(self.config.CHECKPOINT_FILE):
                os.remove(self.config.CHECKPOINT_FILE)

        # ── Pre-prediction Cache Warm-up ──────────────────────────────────────
        logger.info("Initializing prediction cache...")
        from multiprocessing import Manager
        manager = Manager()
        shared_dict = manager.dict()
        
        cache_file = os.path.join(self.config.OUTPUT_DIR, "matchup_cache.pkl")
        if os.path.exists(cache_file):
            logger.info("Loading prediction cache from disk...")
            try:
                with open(cache_file, "rb") as f:
                    disk_cache = pickle.load(f)
                for k, v in disk_cache.items():
                    shared_dict[k] = v
                logger.info(f"Loaded {len(shared_dict)} cached predictions from disk.")
            except Exception as e:
                logger.warning(f"Failed to load cache from disk: {e}")
        else:
            # Warm up the cache using a local KnockoutEngine to pre-populate predictions
            # for the most common/deterministic paths (the starting 16 matches).
            logger.info("Pre-populating cache with starting Round of 32 predictions...")
            temp_engine = KnockoutEngine()
            for fixture in temp_engine.round_32_fixtures:
                m_no = fixture["match_no"]
                h = map_team_name(fixture["home_team"])
                a = map_team_name(fixture["away_team"])
                date = fixture.get("date", "2026-06-28")
                pred = temp_engine.predict_match(m_no, "Round of 32", h, a, date)
                key = (h, a, m_no)
                shared_dict[key] = {
                    "prob_home_win": pred["prob_home_win"],
                    "prob_draw": pred["prob_draw"],
                    "prob_away_win": pred["prob_away_win"],
                    "predicted_outcome": pred["predicted_outcome"],
                    "predicted_winner": pred["predicted_winner"],
                    "confidence": pred["confidence"],
                    "entropy": pred["entropy"],
                    "shootout_played": pred["shootout_played"],
                    "shootout_reason": pred["shootout_reason"]
                }
            # Save the pre-populated cache immediately
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(dict(shared_dict), f)
            except Exception as e:
                logger.warning(f"Failed to save initial cache: {e}")

        # Run chunks in parallel
        chunk_size = 200  # Size of chunk sent to each worker
        sims_per_batch = chk_interval
        
        while completed_sims < sims_to_run:
            current_batch_size = min(sims_per_batch, sims_to_run - completed_sims)
            logger.info(f"Running batch: simulations {completed_sims} to {completed_sims + current_batch_size}...")
            
            # Prepare task arguments for worker pool
            num_workers = self.config.NUM_WORKERS
            tasks = []
            remaining = current_batch_size
            worker_seed = base_seed + completed_sims
            
            while remaining > 0:
                alloc = min(chunk_size, remaining)
                tasks.append((alloc, worker_seed, shared_dict))
                worker_seed += alloc
                remaining -= alloc
                
            # Execute batch in parallel worker pool
            with Pool(processes=num_workers, initializer=init_worker, initargs=(shared_dict,)) as pool:
                batch_results = pool.map(run_sim_chunk_worker, tasks)
                
            # Aggregate batch results
            for m_runs, t_runs in batch_results:
                all_match_runs.extend(m_runs)
                all_team_runs.extend(t_runs)
                
            completed_sims += current_batch_size
            
            # Save checkpoint
            chk_data = {
                "total_sims_requested": sims_to_run,
                "seed": base_seed,
                "completed_sims": completed_sims,
                "match_runs": all_match_runs,
                "team_runs": all_team_runs
            }
            save_checkpoint(chk_data, self.config.CHECKPOINT_FILE)
            logger.info(f"Saved checkpoint for {completed_sims} simulations. Unique cache entries: {len(shared_dict)}")

        # Compute runtime metrics
        perf_metrics = self.tracker.get_metrics(completed_sims)
        logger.info(f"Simulations finished. Elapsed time: {perf_metrics['elapsed_seconds']}s ({perf_metrics['simulations_per_second']} sims/s)")

        # Remove checkpoint file on successful completion
        if os.path.exists(self.config.CHECKPOINT_FILE):
            os.remove(self.config.CHECKPOINT_FILE)

        # Save cache back to disk
        try:
            logger.info("Saving prediction cache to disk...")
            normal_dict = dict(shared_dict)
            with open(cache_file, "wb") as f:
                pickle.dump(normal_dict, f)
        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {e}")

        # ── Compute Analysis & Statistics ────────────────────────────────────
        logger.info("Computing team and tournament statistics...")
        df_team_stats = calculate_team_statistics(all_team_runs, self.config.TEAMS)
        tourney_stats = calculate_tournament_statistics(all_match_runs)
        df_uncertainty = calculate_uncertainty_analysis(all_team_runs, self.config.TEAMS)
        
        # Calculate convergence across checkpoints: 100, 500, 1000, 5000, 10000, etc.
        checkpoints_list = [100, 500, 1000, 5000, 10000, 50000]
        checkpoints_to_run = [c for c in checkpoints_list if c <= sims_to_run]
        if sims_to_run not in checkpoints_to_run:
            checkpoints_to_run.append(sims_to_run)
            
        df_conv = calculate_convergence(all_team_runs, self.config.TEAMS, checkpoints_to_run)

        # Validate probabilities consistency
        valid, validation_errors = validate_probabilities(df_team_stats)
        if not valid:
            logger.warning(f"Probability validation failed with {len(validation_errors)} errors!")
            for err in validation_errors:
                logger.warning(f"  Validation Error: {err}")
        else:
            logger.info("Probability validation passed successfully.")

        # ── Generate Deliverable Reports & Artifacts ──────────────────────────
        logger.info("Generating plots...")
        plot_champion_probability_bar(df_team_stats, self.config.PLOT_DIR)
        plot_champion_distribution_donut(df_team_stats, self.config.PLOT_DIR)
        plot_runner_up_distribution_donut(df_team_stats, self.config.PLOT_DIR)
        plot_advancement_heatmap(df_team_stats, self.config.PLOT_DIR)
        plot_probability_histogram(all_match_runs, self.config.PLOT_DIR)
        plot_upset_distribution(all_match_runs, self.config.PLOT_DIR)
        plot_convergence(df_conv, self.config.PLOT_DIR)
        plot_confidence_intervals(df_uncertainty, self.config.PLOT_DIR)

        # ── Exporting outputs (Task 9) ───────────────────────────────────────
        logger.info("Exporting data files...")
        
        # Save team statistics
        df_team_stats.to_csv(os.path.join(self.config.OUTPUT_DIR, "team_statistics.csv"), index=False)
        df_team_stats.to_csv(os.path.join(self.config.OUTPUT_DIR, "team_probabilities.csv"), index=False)
        
        # Save uncertainty confidence intervals
        df_uncertainty.to_csv(os.path.join(self.config.OUTPUT_DIR, "confidence_intervals.csv"), index=False)
        
        # Save convergence analysis
        df_conv.to_csv(os.path.join(self.config.OUTPUT_DIR, "convergence.csv"), index=False)
        
        # Save distributions
        df_champs = df_team_stats[["team", "champion_prob"]].sort_values("champion_prob", ascending=False)
        df_champs.to_csv(os.path.join(self.config.OUTPUT_DIR, "champion_distribution.csv"), index=False)
        
        df_runners = df_team_stats[["team", "runner_up_prob"]].sort_values("runner_up_prob", ascending=False)
        df_runners.to_csv(os.path.join(self.config.OUTPUT_DIR, "runner_up_distribution.csv"), index=False)
        
        # Save advancement probabilities
        adv_probs = []
        for _, row in df_team_stats.iterrows():
            p_c = row["champion_prob"]
            adv_probs.append({
                "team": row["team"],
                "round_of_16": row["round_of_16_prob"],
                "quarter_final": row["quarter_final_prob"],
                "semi_final": row["semi_final_prob"],
                "final": row["runner_up_prob"] + p_c,
                "champion": p_c
            })
        pd.DataFrame(adv_probs).to_csv(os.path.join(self.config.OUTPUT_DIR, "advancement_probabilities.csv"), index=False)

        # Flattened matches simulation results export
        flat_results = []
        for idx, run in enumerate(all_match_runs):
            if idx >= 100:  # limit details to first 100 simulations to prevent huge files
                break
            for m_no, m in run.items():
                flat_results.append({
                    "simulation_id": idx,
                    "match_no": m_no,
                    "home_team": m["home_team"],
                    "away_team": m["away_team"],
                    "winner": m["winner"],
                    "sampled_outcome": m["sampled_outcome"],
                    "shootout_played": m["shootout_played"]
                })
        df_flat = pd.DataFrame(flat_results)
        df_flat.to_csv(os.path.join(self.config.OUTPUT_DIR, "simulation_results.csv"), index=False)
        df_flat.to_parquet(os.path.join(self.config.OUTPUT_DIR, "simulation_results.parquet"), index=False)
        df_flat.to_json(os.path.join(self.config.OUTPUT_DIR, "simulation_results.json"), orient="records", indent=4)

        # Metadata & summaries export
        metadata = {
            "simulation_version": "1.0.0",
            "timestamp": pd.Timestamp.now().isoformat(),
            "random_seed": base_seed,
            "git_commit_hash": get_git_commit_hash(),
            "config": {
                "num_simulations": sims_to_run,
                "workers": num_workers
            }
        }
        with open(os.path.join(self.config.OUTPUT_DIR, "simulation_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
        with open(os.path.join(self.config.OUTPUT_DIR, "simulation_config.json"), "w") as f:
            json.dump(metadata["config"], f, indent=4)

        # Runtime stats
        with open(os.path.join(self.config.OUTPUT_DIR, "runtime_statistics.json"), "w") as f:
            json.dump(perf_metrics, f, indent=4)

        # Summary statistics
        summary = {
            "performance": perf_metrics,
            "validation": {
                "passed": valid,
                "errors": validation_errors
            },
            "tournament_statistics": tourney_stats
        }
        with open(os.path.join(self.config.OUTPUT_DIR, "simulation_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

        logger.info("Simulation engine run completed successfully.")
        
        return {
            "team_statistics": df_team_stats,
            "tournament_statistics": tourney_stats,
            "uncertainty_analysis": df_uncertainty,
            "convergence": df_conv,
            "performance": perf_metrics,
            "validation_passed": valid
        }
