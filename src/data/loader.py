import os
import pandas as pd
import yaml
from typing import Dict, Any, Generator
from src.utils.logger import setup_logger

logger = setup_logger("loader")

class DataLoader:
    def __init__(self, config_path: str = "configs/data_config.yaml", project_root: str = "."):
        self.project_root = project_root
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
    def get_absolute_path(self, rel_path: str) -> str:
        return os.path.join(self.project_root, rel_path)
        
    def load_csv(self, key: str) -> pd.DataFrame:
        """Loads a full CSV dataset from the config key."""
        if key not in self.config["raw_paths"]:
            raise KeyError(f"Dataset key '{key}' not found in data_config.yaml raw_paths.")
            
        path = self.get_absolute_path(self.config["raw_paths"][key])
        logger.info(f"Loading dataset '{key}' from {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
            
        try:
            df = pd.read_csv(path, low_memory=False)
            logger.info(f"Successfully loaded '{key}' with {len(df):,} rows and {len(df.columns)} columns.")
            return df
        except Exception as e:
            logger.error(f"Error loading '{key}': {e}")
            raise e

    def load_csv_in_chunks(self, key: str, chunksize: int = 100000) -> Generator[pd.DataFrame, None, None]:
        """Generator to load large CSV datasets in chunks (e.g., Transfermarkt)."""
        if key not in self.config["raw_paths"]:
            raise KeyError(f"Dataset key '{key}' not found in data_config.yaml raw_paths.")
            
        path = self.get_absolute_path(self.config["raw_paths"][key])
        logger.info(f"Loading dataset '{key}' in chunks of {chunksize:,} from {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
            
        try:
            for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
                yield chunk
        except Exception as e:
            logger.error(f"Error loading chunked '{key}': {e}")
            raise e
            
    def get_config_value(self, keys_path: str) -> Any:
        """Helper to get nested config values."""
        parts = keys_path.split(".")
        val = self.config
        for p in parts:
            val = val[p]
        return val
