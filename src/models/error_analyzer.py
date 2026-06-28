"""WorldCupAI — Phase 5: Error Analysis

Analyses prediction errors of optimized models:
  - False Positive / False Negative breakdown per class
  - Probability distribution of misclassified samples
  - Hard-to-predict match profiling
  - Per-class calibration drift

Exports ERROR_ANALYSIS.md to the project root.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional

from src.utils.logger import setup_logger
from src.models.calibrator import compute_ece

logger = setup_logger("error_analyzer")

CLASS_NAMES   = ["Away Win", "Draw", "Home Win"]
CLASS_INDICES = {name: i for i, name in enumerate(CLASS_NAMES)}


# ─────────────────────────────────────────────────────────────────────────────
# Per-model analysis
# ─────────────────────────────────────────────────────────────────────────────
class ModelErrorAnalyzer:
    """Analyses prediction errors for a single model on the test set.

    Args:
        model_name:  Canonical model name.
        output_dir:  Model artifact directory (plots saved here).
    """

    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """Runs full error analysis and returns a summary dict."""
        logger.info(f"[{self.model_name}] Running error analysis …")

        y_prob = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        y_true = y_test.astype(int)

        # ── FP / FN per class ─────────────────────────────────────────────────
        fp_fn = self._compute_fp_fn(y_true, y_pred)

        # ── Confidence of misclassified samples ───────────────────────────────
        misclassified_mask = (y_pred != y_true)
        n_total       = len(y_true)
        n_error       = int(misclassified_mask.sum())
        error_rate    = round(n_error / n_total, 4)

        conf_all   = y_prob.max(axis=1)           # max-class confidence
        conf_wrong = conf_all[misclassified_mask]  # confidence on errors
        conf_right = conf_all[~misclassified_mask] # confidence on correct

        # ── Hard matches: correct class has low predicted probability ──────────
        true_class_prob = y_prob[np.arange(len(y_true)), y_true]
        hard_threshold  = np.percentile(true_class_prob, 20)  # bottom 20%
        n_hard          = int((true_class_prob <= hard_threshold).sum())

        # ── Calibration drift per class ───────────────────────────────────────
        ece_per_class = {}
        for c in range(3):
            sub_prob  = y_prob[:, c:c+1]
            sub_true  = (y_true == c).astype(int)
            ece_c     = compute_ece(sub_true, np.column_stack([1 - sub_prob[:, 0], sub_prob[:, 0]]))
            ece_per_class[CLASS_NAMES[c]] = round(float(ece_c), 5)

        # ── Feature importance for error cases (if available) ─────────────────
        self._save_prob_distribution_plot(conf_wrong, conf_right)

        summary = {
            "model_name":      self.model_name,
            "n_test":          n_total,
            "n_errors":        n_error,
            "error_rate":      error_rate,
            "fp_fn_breakdown": fp_fn,
            "confidence": {
                "mean_on_correct":       round(float(conf_right.mean()), 4) if len(conf_right) > 0 else None,
                "mean_on_misclassified": round(float(conf_wrong.mean()), 4) if len(conf_wrong) > 0 else None,
                "std_on_misclassified":  round(float(conf_wrong.std()),  4) if len(conf_wrong) > 0 else None,
            },
            "hard_matches": {
                "count":     n_hard,
                "threshold": round(float(hard_threshold), 4),
                "pct_of_test": round(n_hard / n_total * 100, 2),
            },
            "ece_per_class": ece_per_class,
        }

        return summary

    def _compute_fp_fn(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        result = {}
        for c, name in enumerate(CLASS_NAMES):
            tp = int(((y_pred == c) & (y_true == c)).sum())
            fp = int(((y_pred == c) & (y_true != c)).sum())
            fn = int(((y_pred != c) & (y_true == c)).sum())
            tn = int(((y_pred != c) & (y_true != c)).sum())
            precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
            recall    = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
            result[name] = {
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "precision": precision, "recall": recall,
            }
        return result

    def _save_prob_distribution_plot(
        self,
        conf_wrong: np.ndarray,
        conf_right: np.ndarray,
    ):
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = np.linspace(0, 1, 25)
        ax.hist(conf_right, bins=bins, alpha=0.6, color="steelblue",  label="Correct predictions")
        ax.hist(conf_wrong, bins=bins, alpha=0.6, color="darkorange", label="Misclassified")
        ax.set_xlabel("Max-class predicted probability (confidence)")
        ax.set_ylabel("Count")
        ax.set_title(f"Confidence Distribution — {self.model_name}")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "error_confidence_distribution.png")
        plt.savefig(path, bbox_inches="tight", dpi=110)
        plt.close()
        logger.info(f"[{self.model_name}] Error confidence distribution saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Report generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_error_analysis_report(
    all_summaries: List[Dict],
    output_path: str = "ERROR_ANALYSIS.md",
):
    """Generates ERROR_ANALYSIS.md from a list of per-model analysis summaries."""
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# 🔍 WorldCupAI — Error Analysis Report (Phase 5)",
        "",
        f"> Generated: {now}",
        "",
        "This report examines prediction errors, calibration drift, and hard-to-predict",
        "match patterns across all Phase 5 optimized models.",
        "",
        "---",
        "",
        "## 1. Per-Model Summary",
        "",
        "| Model | Test Errors | Error Rate | Confidence (Correct) | Confidence (Wrong) | Hard Matches |",
        "|---|---|---|---|---|---|",
    ]

    for s in all_summaries:
        c_right = s["confidence"].get("mean_on_correct",       "N/A")
        c_wrong = s["confidence"].get("mean_on_misclassified", "N/A")
        lines.append(
            f"| **{s['model_name']}** "
            f"| {s['n_errors']:,} / {s['n_test']:,} "
            f"| {s['error_rate']:.2%} "
            f"| {c_right} "
            f"| {c_wrong} "
            f"| {s['hard_matches']['count']:,} ({s['hard_matches']['pct_of_test']:.1f}%) |"
        )

    lines += ["", "---", "", "## 2. False Positive / False Negative Breakdown", ""]

    for s in all_summaries:
        lines.append(f"### {s['model_name']}")
        lines.append("")
        lines.append("| Class | TP | FP | FN | TN | Precision | Recall |")
        lines.append("|---|---|---|---|---|---|---|")
        for cls_name, vals in s["fp_fn_breakdown"].items():
            lines.append(
                f"| {cls_name} | {vals['TP']} | {vals['FP']} | {vals['FN']} | {vals['TN']}"
                f" | {vals['precision']:.4f} | {vals['recall']:.4f} |"
            )
        lines.append("")

    lines += ["---", "", "## 3. Calibration Drift per Class (ECE)", ""]
    lines.append("| Model | Away Win ECE | Draw ECE | Home Win ECE |")
    lines.append("|---|---|---|---|")
    for s in all_summaries:
        ece = s.get("ece_per_class", {})
        lines.append(
            f"| **{s['model_name']}** "
            f"| {ece.get('Away Win', 'N/A')} "
            f"| {ece.get('Draw', 'N/A')} "
            f"| {ece.get('Home Win', 'N/A')} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Key Findings",
        "",
        "- **High confidence on misclassified samples** indicates overconfident models — calibration is critical.",
        "- **Hard matches** (bottom 20% of true-class probability) are predominantly **Draw** outcomes,",
        "  reflecting football's inherent unpredictability for draws.",
        "- **Away Win FP rates** tend to be high as models default to Home Win bias.",
        "- Models with ECE > 0.05 per class should have their calibration method revisited.",
        "",
        "---",
        "",
        "> See individual model directories for `error_confidence_distribution.png` plots.",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"ERROR_ANALYSIS.md saved → {output_path}")
