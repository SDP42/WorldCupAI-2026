"""WorldCupAI — Phase 6: Abstract Base Model for all PyTorch DL models."""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from src.utils.logger import setup_logger

logger = setup_logger("dl_base_model")


class BaseDLModel(ABC, nn.Module):
    """Abstract base class for all WorldCupAI deep learning models.

    Provides a unified interface for:
      - forward pass
      - predict_proba (softmax output, numpy)
      - predict (argmax)
      - save / load (state_dict + complete model)
      - config serialization
    """

    def __init__(self, model_name: str, n_classes: int = 3):
        super().__init__()
        self.model_name = model_name
        self.n_classes = n_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns raw logits (batch, n_classes)."""
        ...

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Returns a JSON-serialisable dict of all model hyperparameters."""
        ...

    # ── Inference helpers ─────────────────────────────────────────────────────
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Returns softmax probabilities as a numpy array (batch, n_classes)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Returns hard class predictions (argmax) as a numpy array (batch,)."""
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, output_dir: str):
        """Saves model.pt (state_dict), model_complete.pt, and model_config.json."""
        os.makedirs(output_dir, exist_ok=True)

        # 1. State dict
        state_path = os.path.join(output_dir, "model.pt")
        torch.save(self.state_dict(), state_path)
        logger.info(f"[{self.model_name}] state_dict saved → {state_path}")

        # 2. Complete model (torch.save on full module)
        complete_path = os.path.join(output_dir, "model_complete.pt")
        torch.save(self, complete_path)
        logger.info(f"[{self.model_name}] complete model saved → {complete_path}")

        # 3. Config JSON
        config_path = os.path.join(output_dir, "model_config.json")
        with open(config_path, "w") as f:
            json.dump(self.get_config(), f, indent=4)
        logger.info(f"[{self.model_name}] config saved → {config_path}")

    @classmethod
    def load_state(cls, model_instance: "BaseDLModel", state_path: str, device: torch.device) -> "BaseDLModel":
        """Loads state_dict into an existing model instance."""
        model_instance.load_state_dict(torch.load(state_path, map_location=device))
        model_instance.eval()
        return model_instance


def get_device() -> torch.device:
    """Returns the best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Device: Apple MPS (Metal GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logger.info("Device: CPU")
    return device
