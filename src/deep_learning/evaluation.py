"""WorldCupAI — Phase 6: DL Model Evaluation

Reuses Phase 4/5 ModelEvaluator and metrics modules — no duplication.
Adds:
  - Learning curve plots (train vs val loss / accuracy)
  - Prediction CSV export (match_id, true_label, pred_label, class probs)
"""
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional

from src.deep_learning.base_model import BaseDLModel, get_device
from src.deep_learning.metrics import compute_classification_metrics, compute_ece, compute_mce
from src.models.evaluator import ModelEvaluator  # reuse Phase 4/5 plots
from src.utils.logger import setup_logger

logger = setup_logger("dl_evaluation")

CLASS_NAMES = ["Away Win", "Draw", "Home Win"]


# ─────────────────────────────────────────────────────────────────────────────
# DL Evaluator
# ─────────────────────────────────────────────────────────────────────────────
class DLEvaluator:
    """Evaluates a trained DL model on the test set and saves all artifacts.

    Reuses Phase 4/5 ModelEvaluator for confusion matrix and ROC curve plots.

    Args:
        model_name:  Canonical model name ('ANN' or 'LSTM').
        output_dir:  Directory to save evaluation artifacts.
    """

    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate(
        self,
        model:       BaseDLModel,
        test_loader: DataLoader,
        training_time_sec: float,
        test_indices: Optional[np.ndarray] = None,
        predictions_csv_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full evaluation suite on test set.

        Returns:
            metrics dict (identical schema to Phase 5 metrics.json)
        """
        logger.info(f"[{self.model_name}] Evaluating on test set …")

        device = get_device()
        model  = model.to(device)
        model.eval()

        t0 = time.time()
        all_preds, all_targets, all_probs = [], [], []

        import torch.nn.functional as F
        with torch.no_grad():
            for batch in test_loader:
                X_batch, y_batch = batch
                X_batch = X_batch.to(device)
                logits  = model(X_batch)
                probs   = F.softmax(logits, dim=1)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_targets.append(y_batch.numpy())
                all_probs.append(probs.cpu().numpy())

        inference_time = time.time() - t0

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        y_prob = np.concatenate(all_probs)

        # Core metrics — identical to Phase 5
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        metrics["training_time_sec"]   = training_time_sec
        metrics["prediction_time_sec"] = round(inference_time, 4)
        metrics["ece"]  = compute_ece(y_true, y_prob)
        metrics["mce"]  = compute_mce(y_true, y_prob)
        metrics["model_name"] = self.model_name

        # Save metrics.json
        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        # Save classification report
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
        with open(os.path.join(self.output_dir, "classification_report.txt"), "w") as f:
            f.write(report)

        # Reuse Phase 4/5 evaluator for confusion matrix + ROC curve
        ph45_eval = ModelEvaluator(self.model_name, self.output_dir)
        ph45_eval._plot_confusion_matrix(y_true, y_pred)
        ph45_eval._plot_roc_curve(y_true, y_prob)

        # Export predictions CSV
        if predictions_csv_path:
            self._save_predictions_csv(
                y_true, y_pred, y_prob,
                test_indices, predictions_csv_path
            )

        logger.info(
            f"[{self.model_name}] TEST | Acc={metrics['accuracy']:.4f} | "
            f"LogLoss={metrics['log_loss']:.4f} | ROC-AUC={metrics['roc_auc_macro']:.4f} | "
            f"ECE={metrics['ece']:.4f}"
        )
        return metrics

    def plot_learning_curves(self, history_path: str):
        """Plots and saves train vs val loss/accuracy learning curves."""
        if not os.path.exists(history_path):
            logger.warning(f"training_history.json not found: {history_path}")
            return

        with open(history_path) as f:
            history = json.load(f)

        epochs     = [h["epoch"]      for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss   = [h["val_loss"]   for h in history]
        train_acc  = [h["train_acc"]  for h in history]
        val_acc    = [h["val_acc"]    for h in history]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Learning Curves — {self.model_name}", fontsize=13)

        axes[0].plot(epochs, train_loss, label="Train Loss", color="steelblue")
        axes[0].plot(epochs, val_loss,   label="Val Loss",   color="darkorange")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, train_acc, label="Train Acc", color="steelblue")
        axes[1].plot(epochs, val_acc,   label="Val Acc",   color="darkorange")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy Curve"); axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "learning_curves.png")
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close()
        logger.info(f"[{self.model_name}] Learning curves saved → {path}")

    def _save_predictions_csv(
        self,
        y_true:       np.ndarray,
        y_pred:       np.ndarray,
        y_prob:       np.ndarray,
        test_indices: Optional[np.ndarray],
        path:         str,
    ):
        """Exports prediction CSV: match_id | true_label | pred_label | prob_away_win | prob_draw | prob_home_win"""
        df = pd.DataFrame({
            "match_id":      test_indices if test_indices is not None else np.arange(len(y_true)),
            "true_label":    y_true,
            "predicted_label": y_pred,
            "prob_away_win": y_prob[:, 0].round(6),
            "prob_draw":     y_prob[:, 1].round(6),
            "prob_home_win": y_prob[:, 2].round(6),
        })
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"[{self.model_name}] Predictions CSV saved → {path} ({len(df):,} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# ML model prediction export (Phase 5 models)
# ─────────────────────────────────────────────────────────────────────────────
def export_ml_predictions(
    model_name:      str,
    model_dir:       str,
    X_test:          np.ndarray,
    y_test:          np.ndarray,
    test_indices:    np.ndarray,
    output_csv_path: str,
) -> Dict[str, Any]:
    """Loads a Phase 5 calibrated model and exports its test-set predictions.

    Args:
        model_name:      e.g. 'XGBoost'
        model_dir:       e.g. 'models/xgboost_optimized'
        X_test:          Pre-processed test features (using the same pipeline as training)
        y_test:          True labels
        test_indices:    Original match indices for CSV export
        output_csv_path: Where to save {model}_predictions.csv

    Returns:
        metrics dict loaded from models/{name}/metrics.json
    """
    import pickle, json

    cal_pkl  = os.path.join(model_dir, "calibrated_model.pkl")
    prep_pkl = os.path.join(model_dir, "preprocessing.pkl")

    if not os.path.exists(cal_pkl):
        logger.warning(f"Calibrated model not found: {cal_pkl}")
        return {}

    with open(cal_pkl, "rb") as f:
        cal_model = pickle.load(f)

    # Use the already-transformed X_test (same preprocessing pipeline)
    # Load the model's own preprocessing if needed
    if os.path.exists(prep_pkl):
        with open(prep_pkl, "rb") as f:
            pipeline = pickle.load(f)
        # We pass X_test already transformed by a matching pipeline from the caller.
        # But for safety, transform with the model's own pipeline using the raw features.
        # The caller provides the already-transformed array, so use it directly.
        pass

    t0 = time.time()
    y_prob = cal_model.predict_proba(X_test)
    y_pred = cal_model.predict(X_test)
    inf_time = time.time() - t0

    df = pd.DataFrame({
        "match_id":        test_indices,
        "true_label":      y_test,
        "predicted_label": y_pred,
        "prob_away_win":   y_prob[:, 0].round(6),
        "prob_draw":       y_prob[:, 1].round(6),
        "prob_home_win":   y_prob[:, 2].round(6),
    })
    os.makedirs(os.path.dirname(output_csv_path) if os.path.dirname(output_csv_path) else ".", exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"[{model_name}] Predictions CSV saved → {output_csv_path} ({len(df):,} rows)")

    # Load existing metrics
    metrics_path = os.path.join(model_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return {}
