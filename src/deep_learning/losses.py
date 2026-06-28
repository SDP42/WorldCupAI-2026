"""WorldCupAI — Phase 6: Custom loss functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with optional label smoothing.

    Label smoothing prevents overconfident predictions and acts as
    a form of regularisation for multiclass classifiers.

    Args:
        smoothing: Label smoothing factor ε. 0.0 = standard CE loss.
        n_classes: Number of output classes.
    """
    def __init__(self, smoothing: float = 0.05, n_classes: int = 3):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (batch, n_classes) — raw unnormalised scores
            targets: (batch,) — integer class indices
        Returns:
            scalar loss
        """
        if self.smoothing == 0.0:
            return F.cross_entropy(logits, targets)

        # Build smoothed one-hot targets
        confidence = 1.0 - self.smoothing
        n = self.n_classes
        smooth_val = self.smoothing / (n - 1)

        one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
        smooth_one_hot = one_hot * confidence + (1 - one_hot) * smooth_val

        log_probs = F.log_softmax(logits, dim=1)
        loss = -(smooth_one_hot * log_probs).sum(dim=1).mean()
        return loss
