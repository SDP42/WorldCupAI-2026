"""WorldCupAI — Phase 9 Simulation Utilities."""
import os
import sys
import json
import time
import pickle
import psutil
import subprocess
from typing import Dict, Any, Optional

def get_git_commit_hash() -> str:
    """Helper to get current git commit hash if available."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"

class PerformanceTracker:
    """Utility to measure runtime, memory, CPU, and worker stats."""
    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        
    def get_metrics(self, sim_count: int) -> Dict[str, Any]:
        """Calculates current performance metrics."""
        end_time = time.time()
        elapsed = end_time - self.start_time
        sims_per_sec = sim_count / elapsed if elapsed > 0 else 0.0
        
        cpu_util = psutil.cpu_percent()
        mem_info = self.process.memory_info()
        mem_mb = mem_info.rss / (1024 * 1024)
        
        return {
            "elapsed_seconds": round(elapsed, 2),
            "simulations_per_second": round(sims_per_sec, 2),
            "cpu_utilization_pct": cpu_util,
            "memory_usage_mb": round(mem_mb, 2),
            "simulations_completed": sim_count
        }

def save_checkpoint(data: Any, filepath: str):
    """Saves checkpoint data to disk safely."""
    temp_filepath = filepath + ".tmp"
    with open(temp_filepath, "wb") as f:
        pickle.dump(data, f)
    os.replace(temp_filepath, filepath)

def load_checkpoint(filepath: str) -> Optional[Any]:
    """Loads checkpoint data if available."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None
