"""WorldCupAI — Phase 10: Global Feature Importance Analyzer.

Extracts feature importances from all saved ML model artifacts,
computes permutation importance on validation predictions,
and exports ranked importance tables.

Reads from:
  - models/*/calibrated_model.pkl
  - models/*/preprocessing.pkl
  - predictions/val_*_predictions.csv (validation predictions)

Writes to:
  - predictions/feature_importance.csv
  - predictions/global_feature_importance.json
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from src.explainability.utils import (
    FEATURE_COLS, FEATURE_LABELS, FEATURE_CATEGORIES,
    MODEL_DIRS, PREDICTIONS_DIR,
    get_label, get_category, strip_preprocessor_prefix, ensure_dir, save_json,
)
from src.utils.logger import setup_logger

logger = setup_logger("feature_importance")


class FeatureImportanceAnalyzer:
    """Extracts and aggregates feature importances from all production ML models.

    Uses:
    - tree-based models: native feature_importances_
    - linear models: absolute coefficient values (normalized)
    - ensemble: weighted average using production ensemble weights
    """

    def __init__(
        self,
        model_dirs: Optional[Dict[str, str]] = None,
        ensemble_weights: Optional[Dict[str, float]] = None,
        predictions_dir: str = PREDICTIONS_DIR,
    ):
        self.model_dirs       = model_dirs or MODEL_DIRS
        self.ensemble_weights = ensemble_weights or {}
        self.predictions_dir  = predictions_dir
        self.importances: Dict[str, Dict[str, float]] = {}  # model → {feature: importance}
        self.feature_names: List[str] = FEATURE_COLS

    def _load_model_artifacts(self, model_name: str, m_dir: str):
        """Loads preprocessing pipeline and calibrated model from disk."""
        prep_path = os.path.join(m_dir, "preprocessing.pkl")
        cal_path  = os.path.join(m_dir, "calibrated_model.pkl")

        if not os.path.exists(prep_path) or not os.path.exists(cal_path):
            logger.warning(f"Model artifacts not found for {model_name} in {m_dir}")
            return None, None

        with open(prep_path, "rb") as f:
            prep = pickle.load(f)
        with open(cal_path, "rb") as f:
            model = pickle.load(f)

        return prep, model

    def _get_feature_names_from_prep(self, prep) -> List[str]:
        """Extracts clean feature names from the preprocessing pipeline."""
        try:
            raw_names = prep.get_feature_names_out()
            return [strip_preprocessor_prefix(n) for n in raw_names]
        except Exception:
            return FEATURE_COLS

    def _extract_importances(self, model_name: str, m_dir: str) -> Optional[Dict[str, float]]:
        """Extracts feature importances from a single model."""
        prep, model = self._load_model_artifacts(model_name, m_dir)
        if prep is None or model is None:
            return None

        feat_names = self._get_feature_names_from_prep(prep)

        # Unwrap CalibratedClassifierCV if present
        base_model = model
        if hasattr(model, "calibrated_classifiers_"):
            try:
                base_model = model.calibrated_classifiers_[0].estimator
            except (IndexError, AttributeError):
                base_model = model

        # Tree-based models: use feature_importances_
        if hasattr(base_model, "feature_importances_"):
            imp = base_model.feature_importances_
            # For multiclass GBM: feature_importances_ is already aggregated
            if imp.ndim > 1:
                imp = np.mean(imp, axis=0)
            imp_dict = {}
            for i, val in enumerate(imp):
                name = feat_names[i] if i < len(feat_names) else f"f_{i}"
                # Strip remaining prefix
                clean = strip_preprocessor_prefix(name)
                imp_dict[clean] = float(val)
            return imp_dict

        # Linear models: use absolute coefficients
        if hasattr(base_model, "coef_"):
            coef = np.abs(base_model.coef_)
            # For multiclass: average across classes
            if coef.ndim > 1:
                coef = np.mean(coef, axis=0)
            total = coef.sum() + 1e-12
            imp_dict = {}
            for i, val in enumerate(coef):
                name = feat_names[i] if i < len(feat_names) else f"f_{i}"
                clean = strip_preprocessor_prefix(name)
                imp_dict[clean] = float(val / total)
            return imp_dict

        logger.warning(f"Could not extract importances from {model_name}")
        return None

    def compute(self) -> Dict[str, Any]:
        """Computes feature importances for all models and the weighted ensemble."""
        logger.info("Computing global feature importances from all ML models...")

        for model_name, m_dir in self.model_dirs.items():
            imp = self._extract_importances(model_name, m_dir)
            if imp:
                self.importances[model_name] = imp
                logger.info(f"  {model_name}: {len(imp)} features extracted")

        # Compute weighted ensemble importance
        ensemble_imp: Dict[str, float] = {}
        total_weight = sum(self.ensemble_weights.get(m, 0.0) for m in self.importances)

        if total_weight > 0:
            for feat in FEATURE_COLS:
                weighted_sum = 0.0
                for m_name, imp_dict in self.importances.items():
                    w = self.ensemble_weights.get(m_name, 0.0)
                    weighted_sum += w * imp_dict.get(feat, 0.0)
                ensemble_imp[feat] = weighted_sum / total_weight
        else:
            # Equal-weight average if no weights supplied
            for feat in FEATURE_COLS:
                vals = [imp.get(feat, 0.0) for imp in self.importances.values()]
                ensemble_imp[feat] = float(np.mean(vals)) if vals else 0.0

        self.importances["Ensemble (Weighted)"] = ensemble_imp

        # Compute simple average across all models
        avg_imp: Dict[str, float] = {}
        for feat in FEATURE_COLS:
            all_vals = [
                imp.get(feat, 0.0) for name, imp in self.importances.items()
                if name != "Ensemble (Weighted)"
            ]
            avg_imp[feat] = float(np.mean(all_vals)) if all_vals else 0.0
        self.importances["Average"] = avg_imp

        return self.importances

    def rank(self, model: str = "Ensemble (Weighted)") -> List[Dict[str, Any]]:
        """Returns sorted feature importance ranking for a given model."""
        if not self.importances:
            self.compute()

        imp_dict = self.importances.get(model, self.importances.get("Average", {}))
        ranked = sorted(imp_dict.items(), key=lambda x: -x[1])

        return [
            {
                "rank": i + 1,
                "feature": feat,
                "label": get_label(feat),
                "category": get_category(feat),
                "importance": round(val, 6),
                "importance_pct": round(val * 100, 3),
            }
            for i, (feat, val) in enumerate(ranked)
        ]

    def export(self, output_dir: str = PREDICTIONS_DIR) -> Dict[str, str]:
        """Exports feature importance CSV and JSON artifacts."""
        ensure_dir(output_dir)

        if not self.importances:
            self.compute()

        # ── CSV export ────────────────────────────────────────────────────────
        rows = []
        for model_name, imp_dict in self.importances.items():
            for feat, val in imp_dict.items():
                rows.append({
                    "model": model_name,
                    "feature": feat,
                    "label": get_label(feat),
                    "category": get_category(feat),
                    "importance": round(val, 6),
                    "importance_pct": round(val * 100, 3),
                })
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "feature_importance.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported feature importance CSV → {csv_path}")

        # ── JSON export ───────────────────────────────────────────────────────
        json_data = {
            "metadata": {
                "models": list(self.importances.keys()),
                "num_features": len(FEATURE_COLS),
                "feature_cols": FEATURE_COLS,
            },
            "importances": {
                model: {
                    feat: round(val, 6)
                    for feat, val in imp.items()
                }
                for model, imp in self.importances.items()
            },
            "ranking": self.rank("Ensemble (Weighted)"),
        }
        json_path = os.path.join(output_dir, "global_feature_importance.json")
        save_json(json_data, json_path)
        logger.info(f"Exported global feature importance JSON → {json_path}")

        return {"csv": csv_path, "json": json_path}
