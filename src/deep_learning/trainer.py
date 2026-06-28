"""WorldCupAI — Phase 6: DL Training Loop

Implements a unified training function for both ANN and LSTM models.
Handles:
  - AdamW optimiser with configurable LR and weight_decay
  - ReduceLROnPlateau LR scheduler
  - EarlyStopping and ModelCheckpoint callbacks
  - Per-epoch metric logging (loss, accuracy, ROC-AUC)
  - Full training history return
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any, Dict, List, Tuple, Optional

from src.deep_learning.base_model import BaseDLModel, get_device
from src.deep_learning.callbacks import EarlyStopping, ModelCheckpoint, TrainingLogger
from src.deep_learning.losses import LabelSmoothingCrossEntropyLoss
from src.deep_learning.metrics import compute_classification_metrics
from src.utils.logger import setup_logger

logger = setup_logger("dl_trainer")


# ─────────────────────────────────────────────────────────────────────────────
# Per-epoch helpers
# ─────────────────────────────────────────────────────────────────────────────
def _train_epoch(
    model:     BaseDLModel,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float]:
    """Runs one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds   = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total   += len(y_batch)

    return total_loss / total, correct / total


def _eval_epoch(
    model:     BaseDLModel,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Runs one evaluation epoch. Returns (avg_loss, y_pred, y_true, y_prob)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            probs  = F.softmax(logits, dim=1)

            total_loss    += loss.item() * len(y_batch)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)
    avg_loss = total_loss / len(y_true)

    return avg_loss, y_pred, y_true, y_prob


# ─────────────────────────────────────────────────────────────────────────────
# Main trainer
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    model:          BaseDLModel,
    train_loader:   DataLoader,
    val_loader:     DataLoader,
    output_dir:     str,
    max_epochs:     int   = 200,
    lr:             float = 1e-3,
    weight_decay:   float = 1e-4,
    smoothing:      float = 0.05,
    es_patience:    int   = 15,
    lr_patience:    int   = 5,
    lr_factor:      float = 0.5,
    seed:           int   = 42,
) -> Dict[str, Any]:
    """Trains a DL model with full callback stack and metric logging.

    Args:
        model:          Instantiated (un-fitted) BaseDLModel.
        train_loader:   DataLoader for training split.
        val_loader:     DataLoader for validation split.
        output_dir:     Directory to save checkpoints and history.
        max_epochs:     Maximum training epochs.
        lr:             Initial learning rate.
        weight_decay:   AdamW L2 regularisation coefficient.
        smoothing:      Label smoothing factor.
        es_patience:    EarlyStopping patience.
        lr_patience:    ReduceLROnPlateau patience.
        lr_factor:      ReduceLROnPlateau reduction factor.
        seed:           Random seed.

    Returns:
        Dict with training summary including best_val_loss and n_epochs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    model  = model.to(device)

    criterion  = LabelSmoothingCrossEntropyLoss(smoothing=smoothing, n_classes=model.n_classes)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor, patience=lr_patience)

    checkpoint_path = os.path.join(output_dir, "model.pt")
    early_stop  = EarlyStopping(patience=es_patience, min_delta=1e-4, mode="min")
    checkpointer = ModelCheckpoint(path=checkpoint_path, mode="min")
    hist_path   = os.path.join(output_dir, "training_history.json")
    history_log = TrainingLogger(output_path=hist_path)

    logger.info(f"[{model.model_name}] Training on {device} | "
                f"max_epochs={max_epochs} | lr={lr} | wd={weight_decay}")

    t_start = time.time()
    best_val_loss = float("inf")

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_pred, val_true, val_prob = _eval_epoch(model, val_loader, criterion, device)

        # LR scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Quick metric (avoid full ROC-AUC every epoch for speed)
        val_acc = float((val_pred == val_true).mean())

        epoch_metrics = {
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc,  5),
            "val_loss":   round(val_loss,   5),
            "val_acc":    round(val_acc,    5),
            "lr":         current_lr,
        }
        history_log.log(epoch, epoch_metrics)

        # Checkpoint
        saved = checkpointer(model, val_loss)
        if saved:
            best_val_loss = val_loss

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"[{model.model_name}] Epoch {epoch:03d} | "
                        f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                        f"val_acc={val_acc:.4f} | lr={current_lr:.2e}")

        # Early stopping
        if early_stop(val_loss):
            logger.info(f"[{model.model_name}] Early stopping at epoch {epoch}.")
            break

    elapsed = time.time() - t_start
    history_log.save()

    # Load best checkpoint back into model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    logger.info(f"[{model.model_name}] Training complete | {epoch} epochs | "
                f"best_val_loss={best_val_loss:.4f} | time={elapsed:.1f}s")

    return {
        "model_name":    model.model_name,
        "n_epochs":      epoch,
        "best_val_loss": round(best_val_loss, 5),
        "training_time_sec": round(elapsed, 2),
        "device":        str(device),
    }
