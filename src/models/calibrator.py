"""WorldCupAI — Phase 5: Advanced Probability Calibration

Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
generates reliability diagrams, fits both Platt Scaling and Isotonic
Regression, selects the best method by Brier Score, and persists all
calibration artifacts (calibration.json + calibration_curve.png).
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple, Optional

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split

from src.utils.logger import setup_logger
from src.models.metrics import compute_classification_metrics

logger = setup_logger("calibrator")

CLASS_NAMES = ["Away Win", "Draw", "Home Win"]  # indices 0, 1, 2


# ─────────────────────────────────────────────────────────────────────────────
# ECE / MCE helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (multiclass, macro-averaged over OvR).

    ECE = Σ_b |B_b| / N * |acc(b) - conf(b)|
    Computed per class (OvR) and averaged.
    """
    n_classes = y_prob.shape[1]
    ece_per_class = []

    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        prob_c = y_prob[:, c]
        bin_edges = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
        ece_c = 0.0
        n_total = len(y_bin)

        for low, high in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (prob_c >= low) & (prob_c < high)
            if mask.sum() == 0:
                continue
            frac_pos = y_bin[mask].mean()   # accuracy in bin
            mean_conf = prob_c[mask].mean() # mean confidence in bin
            ece_c += (mask.sum() / n_total) * abs(frac_pos - mean_conf)

        ece_per_class.append(ece_c)

    return float(np.mean(ece_per_class))


def compute_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Maximum Calibration Error (multiclass, max over bins and classes)."""
    n_classes = y_prob.shape[1]
    max_error = 0.0

    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        prob_c = y_prob[:, c]
        bin_edges = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)

        for low, high in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (prob_c >= low) & (prob_c < high)
            if mask.sum() == 0:
                continue
            frac_pos = y_bin[mask].mean()
            mean_conf = prob_c[mask].mean()
            max_error = max(max_error, abs(frac_pos - mean_conf))

    return float(max_error)


# ─────────────────────────────────────────────────────────────────────────────
# Calibrator class
# ─────────────────────────────────────────────────────────────────────────────
class ProbabilityCalibrator:
    """Fits Platt Scaling and Isotonic Regression on calibration data, selects
    the better method by Brier Score, and saves all calibration artifacts.

    Args:
        model_name:  Canonical model name.
        output_dir:  Directory to persist calibration artifacts.
    """

    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────
    def calibrate(
        self,
        base_model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Any, Dict]:
        """Fits, compares, and selects the best calibration method.

        Returns:
            calibrated_model: Best calibrated sklearn estimator.
            calibration_report: Dict with ECE, MCE, Brier scores, method chosen.
        """
        logger.info(f"[{self.model_name}] Starting probability calibration …")

        # ── Baseline (uncalibrated) metrics ───────────────────────────────────
        base_prob = base_model.predict_proba(X_val)
        base_pred = base_model.predict(X_val)
        base_metrics = compute_classification_metrics(y_val, base_pred, base_prob)
        base_ece = compute_ece(y_val, base_prob)
        base_mce = compute_mce(y_val, base_prob)

        # ── Platt Scaling (sigmoid) ───────────────────────────────────────────
        platt_model, platt_brier, platt_ece = self._fit_calibrated(
            base_model, X_train, y_train, X_val, y_val, method="sigmoid"
        )

        # ── Isotonic Regression ───────────────────────────────────────────────
        iso_model, iso_brier, iso_ece = self._fit_calibrated(
            base_model, X_train, y_train, X_val, y_val, method="isotonic"
        )

        # ── Select best ───────────────────────────────────────────────────────
        if platt_brier <= iso_brier:
            best_model = platt_model
            best_method = "platt_scaling"
            best_brier = platt_brier
            best_ece = platt_ece
        else:
            best_model = iso_model
            best_method = "isotonic_regression"
            best_brier = iso_brier
            best_ece = iso_ece

        best_prob = best_model.predict_proba(X_val)
        best_pred = best_model.predict(X_val)
        best_mce = compute_mce(y_val, best_prob)
        best_metrics = compute_classification_metrics(y_val, best_pred, best_prob)

        brier_improvement = base_metrics["brier_score"] - best_brier
        ece_improvement   = base_ece - best_ece

        logger.info(f"[{self.model_name}] Calibration selected: {best_method} | "
                    f"Brier ↓ {brier_improvement:+.4f} | ECE ↓ {ece_improvement:+.4f}")

        # ── Build report dict ─────────────────────────────────────────────────
        report = {
            "model_name":         self.model_name,
            "calibration_method": best_method,
            "baseline": {
                "brier_score": base_metrics["brier_score"],
                "log_loss":    base_metrics["log_loss"],
                "ece":         base_ece,
                "mce":         base_mce,
            },
            "platt_scaling": {
                "brier_score": platt_brier,
                "ece":         platt_ece,
            },
            "isotonic_regression": {
                "brier_score": iso_brier,
                "ece":         iso_ece,
            },
            "calibrated": {
                "brier_score":   best_brier,
                "log_loss":      best_metrics["log_loss"],
                "accuracy":      best_metrics["accuracy"],
                "roc_auc_macro": best_metrics["roc_auc_macro"],
                "ece":           best_ece,
                "mce":           best_mce,
            },
            "improvements": {
                "brier_improvement": brier_improvement,
                "ece_improvement":   ece_improvement,
            },
        }

        # ── Save artifacts ────────────────────────────────────────────────────
        self._save_calibration_json(report)
        self._save_calibration_model(best_model)
        self._plot_reliability_diagram(y_val, base_prob, best_prob, best_method)

        return best_model, report

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _fit_calibrated(
        self,
        base_model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        method: str,
    ) -> Tuple[Any, float, float]:
        """Fits CalibratedClassifierCV with cv='prefit' (uses X_val as cal set)."""
        try:
            cal = CalibratedClassifierCV(base_model, method=method, cv="prefit")
            cal.fit(X_val, y_val)
            prob = cal.predict_proba(X_val)
            y_true_oh = np.eye(3)[y_val]
            brier = float(np.mean(np.sum((prob - y_true_oh) ** 2, axis=1)))
            ece   = compute_ece(y_val, prob)
            return cal, brier, ece
        except Exception as exc:
            logger.warning(f"[{self.model_name}] {method} calibration failed: {exc}")
            prob = base_model.predict_proba(X_val)
            y_true_oh = np.eye(3)[y_val]
            brier = float(np.mean(np.sum((prob - y_true_oh) ** 2, axis=1)))
            ece   = compute_ece(y_val, prob)
            return base_model, brier, ece

    def _save_calibration_json(self, report: Dict):
        path = os.path.join(self.output_dir, "calibration.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"[{self.model_name}] calibration.json saved → {path}")

    def _save_calibration_model(self, model: Any):
        import pickle
        path = os.path.join(self.output_dir, "calibrated_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"[{self.model_name}] Calibrated model saved → {path}")

    def _plot_reliability_diagram(
        self,
        y_val: np.ndarray,
        base_prob: np.ndarray,
        cal_prob:  np.ndarray,
        method:    str,
    ):
        """Plots side-by-side reliability diagrams (before vs after calibration)."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        fig.suptitle(f"Reliability Diagram — {self.model_name}\n(After: {method})", fontsize=13)

        for c, ax in enumerate(axes):
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")

            # Before calibration
            y_bin = (y_val == c).astype(int)
            try:
                frac_b, mean_b = calibration_curve(y_bin, base_prob[:, c], n_bins=10, strategy="uniform")
                ax.plot(mean_b, frac_b, "s-", color="steelblue", label="Uncalibrated")
            except Exception:
                pass

            # After calibration
            try:
                frac_c, mean_c = calibration_curve(y_bin, cal_prob[:, c], n_bins=10, strategy="uniform")
                ax.plot(mean_c, frac_c, "o-", color="darkorange", label=f"Calibrated ({method[:4]})")
            except Exception:
                pass

            ax.set_title(CLASS_NAMES[c], fontsize=11)
            ax.set_xlabel("Mean predicted probability")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

        axes[0].set_ylabel("Fraction of positives")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "calibration_curve.png")
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close()
        logger.info(f"[{self.model_name}] Reliability diagram saved → {path}")
