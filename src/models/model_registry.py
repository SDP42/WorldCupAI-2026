import os
import yaml
from typing import Dict, Any
from src.utils.logger import setup_logger

logger = setup_logger("model_registry")

class ModelRegistry:
    def __init__(self, registry_path: str = "models/model_registry.yaml"):
        self.registry_path = registry_path
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        self.registry = self.load_registry()

    def load_registry(self) -> Dict[str, Any]:
        """Loads the model registry from file or returns an empty dict if not found."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as f:
                try:
                    return yaml.safe_load(f) or {}
                except Exception as e:
                    logger.error(f"Error loading registry file: {e}")
                    return {}
        return {}

    def register_model(self, model_name: str, model_dir: str, metrics: Dict[str, Any], params: Dict[str, Any] = None):
        """Registers a new model run with its metadata and metrics."""
        self.registry[model_name] = {
            "model_directory": model_dir,
            "trained_at": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') if 'pd' in globals() else "2026-06-28 16:47:00",
            "metrics": metrics,
            "hyperparameters": params or {}
        }
        
        # Save to file
        with open(self.registry_path, "w") as f:
            yaml.safe_dump(self.registry, f, default_flow_style=False)
            
        logger.info(f"Registered model '{model_name}' in model_registry.yaml")
import pandas as pd
