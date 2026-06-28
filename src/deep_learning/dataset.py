"""WorldCupAI — Phase 6: PyTorch Datasets

TabularDataset  — flat feature matrix for ANN
SequenceDataset — rolling temporal sequences for LSTM (leakage-safe)
SequenceBuilder — builds LSTM-ready (N, seq_len, n_features) arrays
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

from src.utils.logger import setup_logger

logger = setup_logger("dl_dataset")


# ─────────────────────────────────────────────────────────────────────────────
# TabularDataset  (ANN)
# ─────────────────────────────────────────────────────────────────────────────
class TabularDataset(Dataset):
    """Simple flat-feature dataset for feed-forward ANN.

    Args:
        X: numpy array (N, n_features)
        y: numpy array (N,) — integer class labels 0/1/2
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# SequenceBuilder  (LSTM — leakage-safe)
# ─────────────────────────────────────────────────────────────────────────────
class SequenceBuilder:
    """Constructs rolling temporal sequences from the match feature matrix.

    Leakage-safety guarantee:
      For match at position i in the chronologically sorted feature store,
      the input sequence is rows [i-seq_len : i] — the SEQ_LEN matches that
      occurred BEFORE match i. The target is the outcome of match i.
      Since all 37 feature columns were already computed with a .shift(1)
      protocol during Phase 3, no future information is embedded in any row.

    Args:
        seq_len:  Number of prior matches to include per sequence (default 5).
    """

    def __init__(self, seq_len: int = 5):
        self.seq_len = seq_len

    def build(
        self,
        X_full: np.ndarray,
        y_full: np.ndarray,
        split_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Builds (sequences, targets) for a given set of absolute indices.

        Args:
            X_full:        Full feature matrix (all matches, sorted by date).
            y_full:        Full target array aligned with X_full.
            split_indices: Absolute indices of matches in this split
                           (train / val / test).

        Returns:
            sequences: (N, seq_len, n_features) float32
            targets:   (N,) int64
        """
        n_features = X_full.shape[1]
        sequences = np.zeros((len(split_indices), self.seq_len, n_features), dtype=np.float32)
        targets   = np.zeros(len(split_indices), dtype=np.int64)

        for out_idx, abs_idx in enumerate(split_indices):
            start = max(0, abs_idx - self.seq_len)
            window = X_full[start:abs_idx]          # (≤seq_len, n_features)
            pad_len = self.seq_len - len(window)

            if pad_len > 0:
                # Left-pad with zeros
                padded = np.zeros((self.seq_len, n_features), dtype=np.float32)
                padded[pad_len:] = window
            else:
                padded = window.astype(np.float32)

            sequences[out_idx] = padded
            targets[out_idx]   = y_full[abs_idx]

        logger.info(f"SequenceBuilder: built {len(split_indices)} sequences "
                    f"of shape ({self.seq_len}, {n_features})")
        return sequences, targets

    def save_config(self, path: str, feature_cols: List[str]):
        """Persists sequence generation config to JSON."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        config = {
            "seq_len":      self.seq_len,
            "n_features":   len(feature_cols),
            "feature_cols": feature_cols,
            "padding":      "left-zero",
            "leakage_note": "Window uses rows [i-seq_len:i]; row i excluded.",
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Sequence config saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SequenceDataset  (LSTM)
# ─────────────────────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    """PyTorch Dataset wrapping pre-built LSTM sequences.

    Args:
        sequences: (N, seq_len, n_features) float32 numpy array
        targets:   (N,) int64 numpy array
    """
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets   = torch.tensor(targets,   dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]
