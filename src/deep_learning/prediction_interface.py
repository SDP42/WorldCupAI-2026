"""WorldCupAI — Phase 6: Prediction Interface for DL models.

Provides a unified predict_proba interface identical to Phase 5
PredictionInterface, enabling DL models to slot into the Phase 7 ensemble.
"""
import os
import json
import numpy as np
import torch
from typing import List, Tuple

from src.deep_learning.base_model import get_device
from src.utils.logger import setup_logger

logger = setup_logger("dl_prediction_interface")

CLASS_NAMES = ["Away Win", "Draw", "Home Win"]


class DLPredictionInterface:
    """Loads a saved DL model and provides predict_proba.

    Compatible with Phase 5 PredictionInterface contract.

    Args:
        model_dir:   e.g. 'models/ann'
        model_class: ANNModel or LSTMModel class (not instance)
    """

    def __init__(self, model_dir: str, model_class):
        self.model_dir   = model_dir
        self.device      = get_device()

        # Load config
        config_path = os.path.join(model_dir, "model_config.json")
        with open(config_path) as f:
            self.config = json.load(f)

        # Reconstruct model from config
        self.model = self._load_model(model_class)

    def _load_model(self, model_class):
        cfg = self.config
        if cfg.get("model_name") == "ANN":
            model = model_class(
                input_dim=cfg["input_dim"],
                hidden_dims=cfg["hidden_dims"],
                dropout_rates=cfg["dropout_rates"],
                n_classes=cfg["n_classes"],
            )
        elif cfg.get("model_name") == "LSTM":
            model = model_class(
                input_dim=cfg["input_dim"],
                hidden_dim=cfg["hidden_dim"],
                num_layers=cfg["num_layers"],
                lstm_dropout=cfg["lstm_dropout"],
                fc_dropout=cfg["fc_dropout"],
                seq_len=cfg["seq_len"],
                n_classes=cfg["n_classes"],
            )
        else:
            raise ValueError(f"Unknown model_name in config: {cfg.get('model_name')}")

        state_path = os.path.join(self.model_dir, "model.pt")
        model.load_state_dict(torch.load(state_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        logger.info(f"[{cfg['model_name']}] Loaded from {state_path}")
        return model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns softmax probabilities (N, 3) for flat feature input."""
        tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        return self.model.predict_proba(tensor)

    def predict_match(
        self,
        features: np.ndarray,
    ) -> Tuple[str, List[float]]:
        """Convenience wrapper: returns (outcome_name, [p_away, p_draw, p_home])."""
        probs    = self.predict_proba(features)[0]
        pred_idx = int(np.argmax(probs))
        return CLASS_NAMES[pred_idx], probs.tolist()
