import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_auc_score, average_precision_score
)
from typing import Dict, Any

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Computes all classification metrics for 3-class target (0: Away, 1: Draw, 2: Home).
    
    Metrics computed:
        accuracy, precision_macro, recall_macro, f1_macro,
        roc_auc_macro (OvR), pr_auc_macro (OvR), log_loss, brier_score
    """
    n_classes = y_prob.shape[1]
    
    # ROC-AUC (macro OvR)
    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        roc_auc = np.nan

    # PR-AUC (macro OvR) — average_precision_score requires one-hot encoding
    try:
        y_true_onehot = np.eye(n_classes)[y_true]
        pr_auc = float(np.mean([
            average_precision_score(y_true_onehot[:, c], y_prob[:, c])
            for c in range(n_classes)
        ]))
    except Exception:
        pr_auc = np.nan

    # Log Loss
    loss = log_loss(y_true, y_prob, labels=list(range(n_classes)))

    # Multiclass Brier Score: mean over samples of sum of squared prob errors per class
    y_true_one_hot = np.eye(n_classes)[y_true]
    brier_score = float(np.mean(np.sum((y_prob - y_true_one_hot) ** 2, axis=1)))

    return {
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "precision_macro":  float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "recall_macro":     float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_macro":         float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "roc_auc_macro":    float(roc_auc),
        "pr_auc_macro":     float(pr_auc),
        "log_loss":         float(loss),
        "brier_score":      brier_score,
    }
