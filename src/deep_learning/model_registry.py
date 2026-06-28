"""WorldCupAI — Phase 6: DL Model Registry

Extends the Phase 5 ModelRegistry to register DL models
alongside ML models in models/model_registry.yaml.
"""
import os
import json
from typing import Any, Dict

from src.models.model_registry import ModelRegistry  # reuse Phase 4/5
from src.utils.logger import setup_logger

logger = setup_logger("dl_model_registry")


class DLModelRegistry(ModelRegistry):
    """Extends Phase 5 ModelRegistry for deep learning models.

    DL entries are namespaced as 'ANN (DL)' and 'LSTM (DL)' to avoid
    conflicts with any ML model of the same simple name.
    """

    def register_dl_model(
        self,
        model_name: str,
        model_dir:  str,
        metrics:    Dict[str, Any],
        config:     Dict[str, Any],
    ):
        """Registers a DL model with its config and test metrics."""
        import pandas as pd
        entry_name = f"{model_name} (DL)"
        self.registry[entry_name] = {
            "model_directory": model_dir,
            "trained_at":      pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "framework":       "PyTorch",
            "metrics":         metrics,
            "config":          config,
            "hyperparameters": config,  # kept for schema consistency with Phase 5
        }

        import yaml
        with open(self.registry_path, "w") as f:
            yaml.safe_dump(self.registry, f, default_flow_style=False)

        logger.info(f"Registered '{entry_name}' in model_registry.yaml")
