"""WorldCupAI — Phase 7: Ensemble Performance Evaluation.

Evaluates voting, stacking, and blending ensembles on the test set
and computes all canonical Phase 4/5/6 evaluation metrics.
"""
import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.models.metrics import compute_classification_metrics
from src.models.calibrator import compute_ece, compute_mce
from src.models.evaluator import ModelEvaluator  # reuse Phase 4/5 plots
from src.utils.logger import setup_logger

logger = setup_logger("ensemble_evaluator")


class EnsembleEvaluator:
    """Evaluates multiple ensemble strategies and compares them to baselines.

    Args:
        output_dir: Directory to save evaluation reports and plots.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate_ensemble(
        self,
        ensemble_name:   str,
        y_prob:          np.ndarray,
        y_pred:          np.ndarray,
        y_true:          np.ndarray,
        inference_time:  float,
        training_time:   float = 0.0,
    ) -> Dict[str, Any]:
        """Computes evaluation metrics and saves confusion matrix and ROC curve plots.

        Args:
            ensemble_name:  Name of ensemble strategy (e.g., 'Soft Voting').
            y_prob:         Soft class probabilities (N, 3).
            y_pred:         Hard predictions (N,).
            y_true:         True labels (N,).
            inference_time: Total prediction time in seconds.
            training_time:  Weight optimization or meta-learner training time.

        Returns:
            metrics dictionary
        """
        logger.info(f"Evaluating ensemble: {ensemble_name} …")

        # Reuse canonical metrics
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        metrics["training_time_sec"]   = training_time
        metrics["prediction_time_sec"] = round(inference_time, 5)
        metrics["ece"]  = compute_ece(y_true, y_prob)
        metrics["mce"]  = compute_mce(y_true, y_prob)
        metrics["model_name"] = ensemble_name

        # Save metrics file
        safe_name = ensemble_name.lower().replace(" ", "_")
        m_dir = os.path.join(self.output_dir, safe_name)
        os.makedirs(m_dir, exist_ok=True)

        with open(os.path.join(m_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        # Reuse plots
        ph45_eval = ModelEvaluator(ensemble_name, m_dir)
        ph45_eval._plot_confusion_matrix(y_true, y_pred)
        ph45_eval._plot_roc_curve(y_true, y_prob)

        logger.info(
            f"[{ensemble_name}] TEST | Acc: {metrics['accuracy']:.4f} | "
            f"LogLoss: {metrics['log_loss']:.4f} | ECE: {metrics['ece']:.4f}"
        )
        return metrics

    def select_best_ensembles(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, str]:
        """Identifies best overall, fastest, most calibrated, and most robust models.

        Uses Phase 5 composite ranking logic.
        """
        df = pd.DataFrame(all_metrics)

        # Min-max normalization for composite scoring
        # Composite components (same weights as Phase 5):
        #   Accuracy: 20%, ROC-AUC: 20%, Log Loss: 20%, F1: 15%, Brier: 15%, ECE: 10%
        # Directions: Acc ↑, ROC ↑, F1 ↑, Log Loss ↓, Brier ↓, ECE ↓
        normalized = pd.DataFrame(index=df.index)
        
        metrics_meta = {
            "accuracy":      (0.20, True),
            "roc_auc_macro": (0.20, True),
            "log_loss":      (0.20, False),
            "f1_macro":      (0.15, True),
            "brier_score":   (0.15, False),
            "ece":           (0.10, False),
        }

        for col, (weight, ascending) in metrics_meta.items():
            if col not in df.columns:
                continue
            vals = df[col].values
            v_min, v_max = vals.min(), vals.max()
            
            if v_max == v_min:
                norm_vals = np.ones_like(vals)
            else:
                if ascending:
                    # Higher is better
                    norm_vals = (vals - v_min) / (v_max - v_min)
                else:
                    # Lower is better
                    norm_vals = (v_max - vals) / (v_max - v_min)
            
            normalized[col] = norm_vals * weight

        df["composite_score"] = normalized.sum(axis=1)
        df_ranked = df.sort_values("composite_score", ascending=False)

        best_overall = df_ranked.iloc[0]["model_name"]
        
        # Fastest (min prediction_time_sec)
        fastest = df.sort_values("prediction_time_sec").iloc[0]["model_name"]
        
        # Most Robust (max roc_auc_macro)
        robust = df.sort_values("roc_auc_macro", ascending=False).iloc[0]["model_name"]
        
        # Most Calibrated (min ECE)
        calibrated = df.sort_values("ece").iloc[0]["model_name"]

        return {
            "best_overall":    best_overall,
            "fastest":         fastest,
            "most_robust":     robust,
            "most_calibrated": calibrated,
            "composite_rankings": df_ranked[["model_name", "composite_score"]].to_dict(orient="records")
        }
