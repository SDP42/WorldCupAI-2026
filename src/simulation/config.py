"""WorldCupAI — Phase 9 Simulation Configuration."""
import os
import multiprocessing

class SimulationConfig:
    """Config settings for Monte Carlo tournament simulations."""
    
    # Execution parameters
    DEFAULT_SIMULATIONS = 10000
    DEFAULT_SEED = 42
    DEFAULT_CHECKPOINT_INTERVAL = 1000
    
    # Multiprocessing
    NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)
    
    # Paths
    OUTPUT_DIR = "predictions"
    PLOT_DIR = os.path.join("outputs", "plots")
    CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "simulation_checkpoint.pkl")
    
    # Teams list in tournament (canonical)
    TEAMS = [
        "South Africa", "Canada", "Germany", "Paraguay", "Netherlands", "Morocco",
        "Brazil", "Japan", "France", "Sweden", "Cote d'Ivoire", "Norway",
        "Mexico", "Ecuador", "England", "Democratic Republic of the Congo",
        "United States", "Bosnia and Herzegovina", "Belgium", "Senegal",
        "Portugal", "Croatia", "Spain", "Austria", "Switzerland", "Algeria",
        "Argentina", "Cape Verde", "Colombia", "Ghana", "Australia", "Egypt"
    ]
