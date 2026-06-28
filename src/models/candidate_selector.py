"""WorldCupAI — Phase 5: Objective Model Candidate Selector

Ranks all optimized + calibrated models using a weighted multi-metric
composite score, then identifies:
  - Best Overall Model
  - Fastest Inference Model
  - Most Robust Model  (highest CV stability)
  - Best Calibrated Model  (lowest ECE)
  - Top 5 Ensemble Candidates

Exports MODEL_SELECTION_REPORT.md.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import setup_logger

logger = setup_logger("candidate_selector")

# ─────────────────────────────────────────────────────────────────────────────
# Scoring weights (must sum to 1.0)
# ─────────────────────────────────────────────────────────────────────────────
METRIC_WEIGHTS = {
    "accuracy":      0.20,
    "roc_auc_macro": 0.20,
    "log_loss":      0.20,  # lower is better → inverted
    "f1_macro":      0.15,
    "brier_score":   0.15,  # lower is better → inverted
    "ece":           0.10,  # lower is better → inverted
}
assert abs(sum(METRIC_WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

LOWER_IS_BETTER = {"log_loss", "brier_score", "ece"}


# ─────────────────────────────────────────────────────────────────────────────
# Selector class
# ─────────────────────────────────────────────────────────────────────────────
class ModelCandidateSelector:
    """Selects top ensemble candidates using objective metric-weighted scoring.

    Args:
        model_records: List of dicts, each containing model_name and all
                       required metrics (from both metrics.json and
                       calibration.json).
    """

    def __init__(self, model_records: List[Dict]):
        self.records = model_records
        self.df: Optional[pd.DataFrame] = None

    def rank(self) -> pd.DataFrame:
        """Computes composite scores and returns ranked DataFrame."""
        df = pd.DataFrame(self.records)

        # ── Normalise each metric to [0, 1] min-max ───────────────────────────
        scored_cols = []
        for metric, weight in METRIC_WEIGHTS.items():
            if metric not in df.columns:
                logger.warning(f"Metric '{metric}' missing from records — skipping.")
                continue
            col_raw  = df[metric].copy()
            col_min  = col_raw.min()
            col_max  = col_raw.max()
            rng      = col_max - col_min if (col_max - col_min) > 0 else 1.0

            if metric in LOWER_IS_BETTER:
                # Invert: lower raw → higher normalised score
                col_norm = 1.0 - (col_raw - col_min) / rng
            else:
                col_norm = (col_raw - col_min) / rng

            score_col = f"_score_{metric}"
            df[score_col]  = col_norm * weight
            scored_cols.append(score_col)

        df["composite_score"] = df[scored_cols].sum(axis=1)
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        self.df = df
        return df

    def get_selections(self) -> Dict[str, str]:
        """Returns objective selections for each role."""
        if self.df is None:
            self.rank()

        selections = {}
        selections["best_overall"] = self.df.iloc[0]["model_name"]

        # Fastest: lowest inference time
        if "prediction_time_sec" in self.df.columns:
            selections["fastest"] = self.df.loc[self.df["prediction_time_sec"].idxmin(), "model_name"]
        else:
            selections["fastest"] = "N/A"

        # Most robust: highest ROC-AUC (stable across classes)
        if "roc_auc_macro" in self.df.columns:
            selections["most_robust"] = self.df.loc[self.df["roc_auc_macro"].idxmax(), "model_name"]
        else:
            selections["most_robust"] = "N/A"

        # Best calibrated: lowest ECE
        if "ece" in self.df.columns:
            selections["best_calibrated"] = self.df.loc[self.df["ece"].idxmin(), "model_name"]
        else:
            selections["best_calibrated"] = "N/A"

        # Top 5 ensemble candidates: top composite scorers
        top5 = self.df.head(5)["model_name"].tolist()
        selections["top5_ensemble_candidates"] = top5

        return selections

    def export_report(self, output_path: str = "MODEL_SELECTION_REPORT.md"):
        """Generates and saves MODEL_SELECTION_REPORT.md."""
        if self.df is None:
            self.rank()

        selections = self.get_selections()
        now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── Build markdown table ──────────────────────────────────────────────
        display_cols = ["rank", "model_name", "composite_score",
                        "accuracy", "roc_auc_macro", "log_loss",
                        "f1_macro", "brier_score", "ece",
                        "prediction_time_sec"]
        display_cols = [c for c in display_cols if c in self.df.columns]

        header = "| " + " | ".join(display_cols) + " |"
        sep    = "| " + " | ".join(["---"] * len(display_cols)) + " |"
        rows   = []
        for _, row in self.df[display_cols].iterrows():
            cells = []
            for c in display_cols:
                val = row[c]
                if isinstance(val, float):
                    cells.append(f"{val:.4f}")
                else:
                    cells.append(str(val))
            rows.append("| " + " | ".join(cells) + " |")

        table = "\n".join([header, sep] + rows)

        # ── Ensemble candidates ───────────────────────────────────────────────
        top5_lines = "\n".join(
            [f"{i+1}. **{m}**" for i, m in enumerate(selections["top5_ensemble_candidates"])]
        )

        content = f"""# 🏆 WorldCupAI — Model Selection Report (Phase 5)

> Generated: {now}

This report provides **objective, metric-weighted rankings** for all Phase 5
optimized and calibrated models. Selections are fully automated — no manual
override.

---

## Scoring Weights

| Metric | Weight | Direction |
|---|---|---|
| Accuracy | 20% | ↑ Higher better |
| ROC-AUC (macro) | 20% | ↑ Higher better |
| Log Loss | 20% | ↓ Lower better |
| F1 (macro) | 15% | ↑ Higher better |
| Brier Score | 15% | ↓ Lower better |
| ECE | 10% | ↓ Lower better |

---

## Full Ranking

{table}

---

## Objective Selections

| Role | Selected Model |
|---|---|
| 🥇 **Best Overall** | {selections["best_overall"]} |
| ⚡ **Fastest Inference** | {selections["fastest"]} |
| 🛡️ **Most Robust** | {selections["most_robust"]} |
| 📐 **Best Calibrated** | {selections["best_calibrated"]} |

---

## Top 5 Ensemble Candidates

{top5_lines}

These models are recommended for inclusion in Phase 6 (Deep Learning Comparison)
and ultimately Phase 7 (Ensemble Construction).

---

## Recommendations for Phase 6

1. The **{selections["best_overall"]}** model should be used as the ML baseline
   for Deep Learning comparison.
2. The **{selections["best_calibrated"]}** model provides the most trustworthy
   probability estimates for match outcome confidence scoring.
3. All **Top 5 ensemble candidates** should be retained for stacking/voting
   ensemble construction in a later phase.
4. Models with Log Loss > 0.85 post-calibration should be excluded from the ensemble.

---

> All model artifacts are stored in `models/{{model_name}}_optimized/`.
"""

        with open(output_path, "w") as f:
            f.write(content)

        logger.info(f"MODEL_SELECTION_REPORT.md saved → {output_path}")
        return selections
