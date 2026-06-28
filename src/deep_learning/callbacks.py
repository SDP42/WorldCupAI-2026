"""WorldCupAI — Phase 6: DL Training Callbacks (PyTorch)

Pure-Python callback implementations:
  - EarlyStopping
  - ModelCheckpoint
  - ReduceLROnPlateau wrapper
  - TrainingLogger
"""
import os
import json
import math
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
from src.utils.logger import setup_logger

logger = setup_logger("dl_callbacks")


class EarlyStopping:
    """Stops training when monitored metric stops improving.

    Args:
        patience:  Epochs to wait with no improvement before stopping.
        min_delta: Minimum change to qualify as improvement.
        mode:      'min' (loss) or 'max' (accuracy, AUC).
    """
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "min"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best      = math.inf if mode == "min" else -math.inf
        self.triggered = False

    def __call__(self, value: float) -> bool:
        """Returns True if training should stop."""
        improved = (
            (self.mode == "min" and value < self.best - self.min_delta) or
            (self.mode == "max" and value > self.best + self.min_delta)
        )
        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                logger.info(f"EarlyStopping: patience {self.patience} exhausted. Stopping.")
        return self.triggered


class ModelCheckpoint:
    """Saves model state_dict whenever monitored metric improves.

    Args:
        path:  Full path to save the .pt file.
        mode:  'min' (loss) or 'max' (accuracy, AUC).
    """
    def __init__(self, path: str, mode: str = "min"):
        self.path  = path
        self.mode  = mode
        self.best  = math.inf if mode == "min" else -math.inf
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    def __call__(self, model: nn.Module, value: float) -> bool:
        """Saves model if value improved. Returns True if saved."""
        improved = (
            (self.mode == "min" and value < self.best) or
            (self.mode == "max" and value > self.best)
        )
        if improved:
            self.best = value
            torch.save(model.state_dict(), self.path)
            return True
        return False


class TrainingLogger:
    """Accumulates per-epoch metrics and exports to JSON.

    Args:
        output_path: Path to write training_history.json.
    """
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.history: List[Dict[str, Any]] = []

    def log(self, epoch: int, metrics: Dict[str, Any]):
        """Appends one epoch's metrics."""
        entry = {"epoch": epoch, **metrics}
        self.history.append(entry)

    def save(self):
        """Writes accumulated history to JSON."""
        os.makedirs(os.path.dirname(self.output_path) if os.path.dirname(self.output_path) else ".", exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved → {self.output_path}")

    def last(self) -> Dict[str, Any]:
        return self.history[-1] if self.history else {}
