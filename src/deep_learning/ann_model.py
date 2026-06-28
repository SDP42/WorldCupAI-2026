"""WorldCupAI — Phase 6: ANN Feed-Forward Neural Network

Architecture:
  Input(n_features)
    → Linear(n_features, 256) + BatchNorm1d + ReLU + Dropout(0.3)
    → Linear(256, 128)        + BatchNorm1d + ReLU + Dropout(0.3)
    → Linear(128, 64)         + BatchNorm1d + ReLU + Dropout(0.2)
    → Linear(64, 3)           [logits]

Regularisation: L2 via weight_decay in AdamW, Dropout, BatchNorm.
Initialisation: He (Kaiming uniform) for all Linear layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

from src.deep_learning.base_model import BaseDLModel


class ANNModel(BaseDLModel):
    """Production-grade feed-forward ANN for 3-class match outcome prediction.

    Args:
        input_dim:     Number of input features (default 37).
        hidden_dims:   List of hidden layer widths.
        dropout_rates: Dropout probability per hidden layer.
        n_classes:     Number of output classes (default 3).
    """

    def __init__(
        self,
        input_dim:     int        = 37,
        hidden_dims:   List[int]  = None,
        dropout_rates: List[float]= None,
        n_classes:     int        = 3,
    ):
        super().__init__(model_name="ANN", n_classes=n_classes)

        self.input_dim     = input_dim
        self.hidden_dims   = hidden_dims   or [256, 128, 64]
        self.dropout_rates = dropout_rates or [0.3, 0.3, 0.2]

        # Build layers
        layers = []
        prev_dim = input_dim
        for hidden_dim, dropout in zip(self.hidden_dims, self.dropout_rates):
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)

        # He (Kaiming) initialisation
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits (batch, n_classes)."""
        return self.network(x)

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name":     self.model_name,
            "framework":      "PyTorch",
            "input_dim":      self.input_dim,
            "hidden_dims":    self.hidden_dims,
            "dropout_rates":  self.dropout_rates,
            "n_classes":      self.n_classes,
            "activation":     "ReLU",
            "batch_norm":     True,
            "weight_init":    "kaiming_uniform (He)",
            "optimizer":      "AdamW",
            "lr":             1e-3,
            "weight_decay":   1e-4,
            "batch_size":     256,
            "loss":           "LabelSmoothingCrossEntropy(smoothing=0.05)",
            "lr_scheduler":   "ReduceLROnPlateau(factor=0.5, patience=5)",
            "early_stopping": "patience=15, min_delta=1e-4",
            "max_epochs":     200,
        }
