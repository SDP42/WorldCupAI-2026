"""WorldCupAI — Phase 6: LSTM Sequence Classifier

Architecture:
  Input: (batch, SEQ_LEN, n_features)
    → LSTM(n_features → hidden_dim=128, num_layers=2, dropout=0.3)
    → LayerNorm(hidden_dim)
    → last hidden state
    → Linear(128, 64) + ReLU + Dropout(0.2)
    → Linear(64, 3)           [logits]

Sequences are built by SequenceBuilder using chronologically prior matches
only — no future information leaks into the input window.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from src.deep_learning.base_model import BaseDLModel


class LSTMModel(BaseDLModel):
    """Production-grade LSTM classifier for temporal match sequence modelling.

    Args:
        input_dim:   Number of features per timestep (default 37).
        hidden_dim:  LSTM hidden state size (default 128).
        num_layers:  Number of stacked LSTM layers (default 2).
        lstm_dropout: Dropout between LSTM layers (default 0.3).
        fc_dropout:  Dropout before final linear layer (default 0.2).
        seq_len:     Sequence length (used in config only; not enforced here).
        n_classes:   Output classes (default 3).
    """

    def __init__(
        self,
        input_dim:    int   = 37,
        hidden_dim:   int   = 128,
        num_layers:   int   = 2,
        lstm_dropout: float = 0.3,
        fc_dropout:   float = 0.2,
        seq_len:      int   = 5,
        n_classes:    int   = 3,
    ):
        super().__init__(model_name="LSTM", n_classes=n_classes)

        self.input_dim    = input_dim
        self.hidden_dim   = hidden_dim
        self.num_layers   = num_layers
        self.lstm_dropout = lstm_dropout
        self.fc_dropout   = fc_dropout
        self.seq_len      = seq_len

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(64, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.kaiming_uniform_(param, nonlinearity="relu")
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1.0 (helps with long-range memory)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, n_features)
        Returns:
            logits: (batch, n_classes)
        """
        # LSTM output: (batch, seq_len, hidden_dim), (h_n, c_n)
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers, batch, hidden_dim) — take last layer
        last_hidden = h_n[-1]                    # (batch, hidden_dim)
        normalized  = self.layer_norm(last_hidden)
        return self.fc(normalized)

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name":     self.model_name,
            "framework":      "PyTorch",
            "input_dim":      self.input_dim,
            "hidden_dim":     self.hidden_dim,
            "num_layers":     self.num_layers,
            "lstm_dropout":   self.lstm_dropout,
            "fc_dropout":     self.fc_dropout,
            "seq_len":        self.seq_len,
            "n_classes":      self.n_classes,
            "layer_norm":     True,
            "weight_init":    "kaiming_uniform (ih), orthogonal (hh), forget_bias=1",
            "optimizer":      "AdamW",
            "lr":             1e-3,
            "weight_decay":   1e-4,
            "batch_size":     256,
            "loss":           "LabelSmoothingCrossEntropy(smoothing=0.05)",
            "lr_scheduler":   "ReduceLROnPlateau(factor=0.5, patience=5)",
            "early_stopping": "patience=15, min_delta=1e-4",
            "max_epochs":     200,
            "sequence_mode":  "global_rolling_window",
            "padding":        "left-zero",
        }
