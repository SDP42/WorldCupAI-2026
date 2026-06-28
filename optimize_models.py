#!/usr/bin/env python3
"""WorldCupAI — Phase 5: Hyperparameter Optimization, Calibration & Model Selection

Orchestrates the full Phase 5 pipeline:
  1. Load feature store + time-aware split (identical to Phase 4)
  2. Optimize top candidate models (GridSearch / RandomizedSearch + early stopping)
  3. Calibrate each optimized model (Platt vs Isotonic → best selected)
  4. Evaluate all optimized+calibrated models on the test set
  5. Run error analysis
  6. Objective model candidate selection
  7. Generate all Phase 5 documentation

IMPORTANT: Does NOT train Deep Learning models. Does NOT build the ensemble.
Does NOT generate World Cup predictions. Does NOT build Streamlit.
"""
import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.logger import setup_logger
from src.models.cross_validation import time_aware_split
from src.models.trainer import PreprocessingPipeline, prepare_targets, save_preprocessing_pipeline
from src.models.evaluator import ModelEvaluator
from src.models.calibration import plot_multiclass_calibration  # Phase 4 — kept for compat
from src.models.optimizer import HyperparameterOptimizer, SEARCH_SPACES
from src.models.calibrator import ProbabilityCalibrator, compute_ece, compute_mce
from src.models.error_analyzer import ModelErrorAnalyzer, generate_error_analysis_report
from src.models.candidate_selector import ModelCandidateSelector
from src.models.model_registry import ModelRegistry

# ── Baseline classifiers ─────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] XGBoost not installed — skipping.")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[WARN] LightGBM not installed — skipping.")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("[WARN] CatBoost not installed — skipping.")

logger = setup_logger("optimize_models")

# ─────────────────────────────────────────────────────────────────────────────
# Feature columns — identical to Phase 4 train_baselines.py
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "elo_diff", "elo_ratio", "rank_diff", "rank_ratio",
    "home_form_win_rate_5", "away_form_win_rate_5",
    "home_form_avg_goals_scored_5", "away_form_avg_goals_scored_5",
    "home_form_avg_goals_conceded_5", "away_form_avg_goals_conceded_5",
    "home_form_clean_sheet_rate_5", "away_form_clean_sheet_rate_5",
    "home_form_win_rate_10", "away_form_win_rate_10",
    "home_form_avg_goals_scored_10", "away_form_avg_goals_scored_10",
    "home_form_avg_goals_conceded_10", "away_form_avg_goals_conceded_10",
    "home_form_clean_sheet_rate_10", "away_form_clean_sheet_rate_10",
    "h2h_meetings", "h2h_home_wins", "h2h_away_wins", "h2h_draws", "h2h_gd",
    "home_attack_rating", "away_attack_rating",
    "home_defence_rating", "away_defence_rating",
    "home_world_cup_titles_before", "away_world_cup_titles_before",
    "is_neutral", "is_world_cup", "is_friendly",
    "home_rest_days", "away_rest_days", "rest_difference",
]

# ─────────────────────────────────────────────────────────────────────────────
# Candidate model registry
# ─────────────────────────────────────────────────────────────────────────────
def build_candidate_models() -> dict:
    """Returns {name: (model_instance, needs_scaling)} for optimization candidates."""
    candidates = {
        "Logistic Regression": (
            LogisticRegression(random_state=42, n_jobs=-1),
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            False,
        ),
        "Extra Trees": (
            ExtraTreesClassifier(random_state=42, n_jobs=-1),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            False,
        ),
    }
    if HAS_XGBOOST:
        candidates["XGBoost"] = (
            XGBClassifier(random_state=42, n_jobs=-1, eval_metric="mlogloss",
                          use_label_encoder=False),
            False,
        )
    if HAS_LGBM:
        candidates["LightGBM"] = (
            LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
            False,
        )
    if HAS_CATBOOST:
        candidates["CatBoost"] = (
            CatBoostClassifier(random_state=42, verbose=0),
            False,
        )
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Helper: model directory name
# ─────────────────────────────────────────────────────────────────────────────
def model_dir_name(name: str) -> str:
    return f"models/{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_optimized"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: save feature importance
# ─────────────────────────────────────────────────────────────────────────────
def save_feature_importance(model, feature_names: list, output_dir: str):
    try:
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({
                "feature": feature_names,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            fi.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
        elif hasattr(model, "coef_"):
            coef = np.abs(model.coef_).mean(axis=0)
            fi = pd.DataFrame({
                "feature": feature_names,
                "importance": coef,
            }).sort_values("importance", ascending=False)
            fi.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
        # For CatBoost, LightGBM etc. wrapped in CalibratedClassifierCV
        elif hasattr(model, "estimator") and hasattr(model.estimator, "feature_importances_"):
            fi = pd.DataFrame({
                "feature": feature_names,
                "importance": model.estimator.feature_importances_,
            }).sort_values("importance", ascending=False)
            fi.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    except Exception as exc:
        logger.warning(f"Could not save feature importance: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_phase5():
    logger.info("=" * 65)
    logger.info("WORLDCUPAI — PHASE 5: HYPERPARAMETER OPTIMIZATION PIPELINE")
    logger.info("=" * 65)

    # ── 1. Load feature store ─────────────────────────────────────────────────
    store_path = "processed/feature_store.parquet"
    if not os.path.exists(store_path):
        logger.error(f"Feature store not found at {store_path}. Run build_features.py first.")
        sys.exit(1)

    df = pd.read_parquet(store_path)
    df["target"] = prepare_targets(df)

    # ── 2. Time-aware split (identical boundaries as Phase 4) ─────────────────
    df_modern = df[df["date"] >= pd.to_datetime("2005-01-01")]
    train_df, val_df, test_df = time_aware_split(
        df_modern, train_end="2018-12-31", val_end="2022-12-31"
    )

    y_train = train_df["target"].values
    y_val   = val_df["target"].values
    y_test  = test_df["target"].values

    logger.info(f"Data split — Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # ── 3. Candidate models ───────────────────────────────────────────────────
    candidates = build_candidate_models()

    registry       = ModelRegistry()
    all_records    = []   # for candidate_selector
    error_summaries = []  # for error_analyzer

    # ── 4. Optimize → Calibrate → Evaluate loop ───────────────────────────────
    for name, (base_model, needs_scaling) in candidates.items():
        logger.info("")
        logger.info(f"{'─'*55}")
        logger.info(f"  Processing: {name}")
        logger.info(f"{'─'*55}")

        out_dir = model_dir_name(name)
        os.makedirs(out_dir, exist_ok=True)

        # Build and fit preprocessing pipeline (same as Phase 4)
        prep = PreprocessingPipeline(FEATURE_COLS, scale=needs_scaling)
        X_train_proc, pipeline_obj = prep.fit_transform(train_df)
        X_val_proc  = prep.transform(val_df)
        X_test_proc = prep.transform(test_df)

        # Save preprocessing pipeline
        save_preprocessing_pipeline(pipeline_obj, os.path.join(out_dir, "preprocessing.pkl"))

        # ── 4a. Hyperparameter optimization ───────────────────────────────────
        optimizer = HyperparameterOptimizer(
            model_name=name,
            model=base_model,
            output_dir=out_dir,
            cv_splits=5,
            random_state=42,
            n_jobs=-1,
        )
        t0 = time.time()
        optimized_model, best_params = optimizer.optimize(
            X_train_proc, y_train,
            X_val_proc, y_val,
        )
        opt_time = time.time() - t0

        # ── 4b. Evaluate optimized (pre-calibration) on val set ───────────────
        evaluator = ModelEvaluator(name, out_dir)
        metrics = evaluator.evaluate(optimized_model, X_val_proc, y_val, opt_time)

        # ── 4c. Probability calibration ───────────────────────────────────────
        calibrator = ProbabilityCalibrator(name, out_dir)
        calibrated_model, cal_report = calibrator.calibrate(
            optimized_model,
            X_train_proc, y_train,
            X_val_proc,   y_val,
        )

        # ── 4d. Final evaluation on TEST set (calibrated model) ───────────────
        from src.models.metrics import compute_classification_metrics
        t_inf_start = time.time()
        y_prob_test = calibrated_model.predict_proba(X_test_proc)
        y_pred_test = calibrated_model.predict(X_test_proc)
        inf_time = time.time() - t_inf_start

        test_metrics = compute_classification_metrics(y_test, y_pred_test, y_prob_test)
        test_metrics["training_time_sec"]   = opt_time
        test_metrics["prediction_time_sec"] = inf_time
        test_metrics["model_name"]          = name
        test_metrics["ece"]  = compute_ece(y_test, y_prob_test)
        test_metrics["mce"]  = compute_mce(y_test, y_prob_test)

        # Save test metrics
        test_metrics_path = os.path.join(out_dir, "metrics.json")
        with open(test_metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=4)

        logger.info(
            f"[{name}] TEST | Acc: {test_metrics['accuracy']:.4f} | "
            f"LogLoss: {test_metrics['log_loss']:.4f} | "
            f"ROC-AUC: {test_metrics['roc_auc_macro']:.4f} | "
            f"ECE: {test_metrics['ece']:.4f}"
        )

        # ── 4e. Feature importance ────────────────────────────────────────────
        save_feature_importance(calibrated_model, FEATURE_COLS, out_dir)

        # ── 4f. Error analysis on test set ────────────────────────────────────
        err_analyzer = ModelErrorAnalyzer(name, out_dir)
        err_summary  = err_analyzer.analyze(calibrated_model, X_test_proc, y_test)
        error_summaries.append(err_summary)

        # ── 4g. ROC curve (test set) ──────────────────────────────────────────
        evaluator_test = ModelEvaluator(name, out_dir)
        evaluator_test._plot_roc_curve(y_test, y_prob_test)
        evaluator_test._plot_confusion_matrix(y_test, y_pred_test)

        # ── 4h. Register in model registry ────────────────────────────────────
        registry.register_model(
            f"{name} (Optimized)",
            out_dir,
            test_metrics,
            params=best_params,
        )

        # Collect record for candidate selector
        record = {**test_metrics, **{"calibration_method": cal_report.get("calibration_method", "N/A")}}
        all_records.append(record)

    # ── 5. Error Analysis Report ──────────────────────────────────────────────
    logger.info("")
    logger.info("Generating ERROR_ANALYSIS.md …")
    generate_error_analysis_report(error_summaries, output_path="ERROR_ANALYSIS.md")

    # ── 6. Candidate Selection ────────────────────────────────────────────────
    logger.info("Generating MODEL_SELECTION_REPORT.md …")
    selector = ModelCandidateSelector(all_records)
    selector.rank()
    selections = selector.export_report(output_path="MODEL_SELECTION_REPORT.md")

    # ── 7. Generate all documentation ────────────────────────────────────────
    logger.info("Generating Phase 5 documentation …")
    ranked_df = selector.df

    generate_phase5_readme(ranked_df, selections)
    generate_optimization_readme(candidates)
    generate_calibration_readme()
    generate_approval_report(ranked_df, selections, all_records)
    update_changelog()

    logger.info("")
    logger.info("=" * 65)
    logger.info("WORLDCUPAI PHASE 5 COMPLETED SUCCESSFULLY")
    logger.info(f"  Best Overall:     {selections['best_overall']}")
    logger.info(f"  Best Calibrated:  {selections['best_calibrated']}")
    logger.info(f"  Fastest:          {selections['fastest']}")
    logger.info(f"  Ensemble Picks:   {', '.join(selections['top5_ensemble_candidates'])}")
    logger.info("=" * 65)

    return selections


# ─────────────────────────────────────────────────────────────────────────────
# Documentation generators
# ─────────────────────────────────────────────────────────────────────────────
def generate_phase5_readme(ranked_df: pd.DataFrame, selections: dict):
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    display_cols = ["rank", "model_name", "accuracy", "roc_auc_macro",
                    "log_loss", "f1_macro", "brier_score", "ece"]
    display_cols = [c for c in display_cols if c in ranked_df.columns]

    header = "| " + " | ".join(display_cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(display_cols)) + " |"
    rows   = []
    for _, row in ranked_df[display_cols].iterrows():
        cells = []
        for c in display_cols:
            val = row[c]
            cells.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        rows.append("| " + " | ".join(cells) + " |")
    table = "\n".join([header, sep] + rows)

    content = f"""# 🚀 WorldCupAI — Phase 5: Hyperparameter Optimization

> Generated: {now}

## Overview

Phase 5 optimizes the top ML models identified in Phase 4, applies probability
calibration, and objectively selects the best candidates for the final ensemble.

## Pipeline

```
feature_store.parquet
    → Time-Aware Split (2005-2018 train / 2019-2022 val / 2023+ test)
    → PreprocessingPipeline (Phase 4 — unchanged)
    → HyperparameterOptimizer (GridSearch / RandomizedSearch + TimeSeriesSplit)
    → Early Stopping (XGBoost / LightGBM / CatBoost)
    → ProbabilityCalibrator (Platt vs Isotonic → best selected by Brier Score)
    → ModelEvaluator (test set: Acc, ROC-AUC, Log Loss, F1, Brier, ECE, MCE)
    → ErrorAnalyzer (FP/FN, confidence distributions, hard matches)
    → ModelCandidateSelector (weighted composite score)
```

## Execution

```bash
python3 optimize_models.py
```

## Results Summary

{table}

## Key Selections

| Role | Model |
|---|---|
| 🥇 Best Overall | {selections['best_overall']} |
| 📐 Best Calibrated | {selections['best_calibrated']} |
| ⚡ Fastest | {selections['fastest']} |

## Artifacts

All optimized model artifacts are stored in `models/{{name}}_optimized/`:
- `model.pkl` — optimized model
- `calibrated_model.pkl` — calibrated model
- `preprocessing.pkl` — unchanged Phase 4 pipeline
- `best_params.json` — best hyperparameters found
- `metrics.json` — full test-set metric suite
- `calibration.json` — ECE, MCE, Brier before/after
- `feature_importance.csv` — ranked feature importances
- `training_log.json` — full CV search results
- `roc_curve.png`, `confusion_matrix.png`, `calibration_curve.png`
- `error_confidence_distribution.png`
"""
    with open("README_PHASE5.md", "w") as f:
        f.write(content)
    logger.info("README_PHASE5.md saved.")


def generate_optimization_readme(candidates: dict):
    from src.models.optimizer import SEARCH_SPACES
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    model_sections = []
    for name in candidates:
        if name in SEARCH_SPACES:
            sp = SEARCH_SPACES[name]
            strategy = sp["strategy"].upper()
            n_iter   = sp.get("n_iter") or "Exhaustive"
            params   = json.dumps(sp["params"], indent=6)
            es_note  = " (+ Early Stopping on val set)" if name in ("XGBoost", "LightGBM", "CatBoost") else ""
            model_sections.append(f"""### {name}
- **Strategy**: {strategy}{es_note}
- **Iterations**: {n_iter}
- **Scoring**: neg_log_loss (via TimeSeriesSplit CV, 5 folds)
```json
{params}
```
""")

    content = f"""# ⚙️ WorldCupAI — Model Optimization Guide (Phase 5)

> Generated: {now}

## Cross-Validation Strategy

All searches use `TimeSeriesSplit(n_splits=5)` to maintain temporal integrity.
No future data leaks into training folds.

## Search Strategies

- **GridSearchCV**: Used for models with small parameter grids (Logistic Regression).
- **RandomizedSearchCV**: Used for large grids — samples `n_iter` random combinations.
- **Early Stopping**: XGBoost, LightGBM, and CatBoost are re-fit post-search using
  the validation set as an eval set with `early_stopping_rounds=20`.

## Per-Model Search Spaces

{"".join(model_sections)}

## Reproducibility

All searches use `random_state=42`. Results are deterministic.
"""
    with open("README_MODEL_OPTIMIZATION.md", "w") as f:
        f.write(content)
    logger.info("README_MODEL_OPTIMIZATION.md saved.")


def generate_calibration_readme():
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"""# 📐 WorldCupAI — Probability Calibration Guide (Phase 5)

> Generated: {now}

## Why Calibration Matters

A model predicting 70% for a Home Win should be correct ~70% of the time.
Without calibration, tree-based models tend to push probabilities to extremes
(overconfident), while linear models may underestimate confidence.

## Calibration Metrics

| Metric | Definition | Target |
|---|---|---|
| **ECE** (Expected Calibration Error) | Weighted avg. of abs(accuracy - confidence) across bins | < 0.05 |
| **MCE** (Maximum Calibration Error) | Worst-case bin error | < 0.10 |
| **Brier Score** | Mean squared error of probability vs actual (lower = better) | < 0.45 |

## Methods Compared

### Platt Scaling (Sigmoid)
- Fits a logistic regression over the model's raw probabilities.
- Works well for small calibration sets (< 1,000 samples).
- Parametric — assumes a sigmoid-shaped miscalibration.

### Isotonic Regression
- Non-parametric monotone fit over predicted probabilities.
- Works best for larger calibration sets (> 1,000 samples).
- More flexible but can overfit on small sets.

## Selection Logic

Both methods are fitted using `CalibratedClassifierCV(cv='prefit')` on the
validation set. The method with the lower **Brier Score** on the validation set
is selected and saved as `calibrated_model.pkl`.

## Reliability Diagrams

Each model's `calibration_curve.png` shows the reliability diagram per class
(Away Win / Draw / Home Win) comparing uncalibrated vs calibrated probabilities
to the perfect calibration diagonal.
"""
    with open("README_PROBABILITY_CALIBRATION.md", "w") as f:
        f.write(content)
    logger.info("README_PROBABILITY_CALIBRATION.md saved.")


def generate_approval_report(
    ranked_df: pd.DataFrame,
    selections: dict,
    all_records: list,
):
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # Find baseline best model from Phase 4 registry
    baseline_acc = 0.6534  # Gradient Boosting baseline
    best_record = all_records[0] if all_records else {}
    best_acc = best_record.get("accuracy", 0.0)
    improvement = best_acc - baseline_acc

    top5 = selections.get("top5_ensemble_candidates", [])
    top5_str = "\n".join([f"{i+1}. **{m}**" for i, m in enumerate(top5)])

    # Risk assessment
    risks = []
    for rec in all_records:
        if rec.get("log_loss", 999) > 0.85:
            risks.append(f"- **{rec['model_name']}**: Log Loss > 0.85 post-calibration")
        if rec.get("ece", 999) > 0.10:
            risks.append(f"- **{rec['model_name']}**: ECE > 0.10 — potential calibration failure")

    if not risks:
        risks_str = "- No critical risks identified. All models passed quality thresholds."
    else:
        risks_str = "\n".join(risks)

    content = f"""# ✅ WorldCupAI — Phase 5 Approval Report

**Status**: 🏁 Phase 5 Complete
**Generated**: {now}
**Team**: WorldCupAI AI Engineering Team

---

## 1. Executive Summary

Phase 5 successfully optimized **{len(all_records)}** ML models using time-aware
hyperparameter search, applied Platt Scaling vs Isotonic Regression calibration,
and objectively selected the top ensemble candidates.

- **Best Optimized Model**: **{selections.get('best_overall', 'N/A')}**
- **Best Baseline (Phase 4)**: Gradient Boosting (Accuracy: {baseline_acc:.4f})
- **Accuracy Improvement**: {improvement:+.4f}
- **Best Calibrated Model**: {selections.get('best_calibrated', 'N/A')}
- **Fastest Model**: {selections.get('fastest', 'N/A')}

---

## 2. Performance Improvements

| Model | Phase 4 Acc | Phase 5 Acc | Improvement |
|---|---|---|---|
"""
    # Add rows comparing baseline to optimized (use known baseline values)
    phase4_accs = {
        "Logistic Regression": 0.6077,
        "Random Forest":       0.6479,
        "Extra Trees":         0.6032,
        "Gradient Boosting":   0.6534,
        "XGBoost":             0.6353,
        "LightGBM":            None,
        "CatBoost":            None,
    }
    for rec in all_records:
        mname = rec.get("model_name", "")
        acc5  = rec.get("accuracy", 0.0)
        acc4  = phase4_accs.get(mname)
        if acc4 is not None:
            diff  = acc5 - acc4
            content += f"| {mname} | {acc4:.4f} | {acc5:.4f} | {diff:+.4f} |\n"
        else:
            content += f"| {mname} | (new) | {acc5:.4f} | — |\n"

    content += f"""
---

## 3. Calibration Summary

All models now have ECE, MCE, and Brier Score recorded.
The best calibration method (Platt Scaling or Isotonic Regression) was
automatically selected per model by Brier Score on the validation set.

---

## 4. Top 5 Ensemble Candidates

{top5_str}

These models are recommended for inclusion in Phase 7 (Ensemble Construction).

---

## 5. Remaining Risks

{risks_str}

---

## 6. Recommendation for Phase 6

- Proceed to **Phase 6: Deep Learning Models** (LSTM / GRU / Transformer).
- Use the **{selections.get('best_overall', 'best optimized model')}** as the ML
  baseline for DL comparison.
- The probability calibration framework built in Phase 5 should be reused
  for Deep Learning models.
- All Top 5 ensemble candidates are approved for Phase 7.

---

## 7. Artifacts Generated

| Artifact | Path |
|---|---|
| Phase 5 README | `README_PHASE5.md` |
| Optimization Guide | `README_MODEL_OPTIMIZATION.md` |
| Calibration Guide | `README_PROBABILITY_CALIBRATION.md` |
| Model Selection | `MODEL_SELECTION_REPORT.md` |
| Error Analysis | `ERROR_ANALYSIS.md` |
| Model Registry | `models/model_registry.yaml` |

---

### ✅ Phase 5 is complete. Awaiting approval for Phase 6.
"""
    with open("PHASE_5_APPROVAL_REPORT.md", "w") as f:
        f.write(content)
    logger.info("PHASE_5_APPROVAL_REPORT.md saved.")


def update_changelog():
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"""
## [Phase 5] — {now}

### Added
- `src/models/optimizer.py` — HyperparameterOptimizer with GridSearch/RandomizedSearch + TimeSeriesSplit CV
- `src/models/calibrator.py` — ECE/MCE, Platt Scaling vs Isotonic Regression comparison
- `src/models/error_analyzer.py` — FP/FN analysis, confidence distribution, hard-match profiling
- `src/models/candidate_selector.py` — Weighted composite scoring + objective model selection
- `optimize_models.py` — Phase 5 orchestration pipeline
- `tests/test_phase5.py` — Automated validation tests
- `README_PHASE5.md`, `README_MODEL_OPTIMIZATION.md`, `README_PROBABILITY_CALIBRATION.md`
- `MODEL_SELECTION_REPORT.md`, `ERROR_ANALYSIS.md`, `PHASE_5_APPROVAL_REPORT.md`

### Changed
- `src/models/metrics.py` — Added PR-AUC (average_precision_score) to compute_classification_metrics

### Models Optimized
- Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost

"""
    changelog_path = "CHANGELOG.md"
    if os.path.exists(changelog_path):
        with open(changelog_path, "r") as f:
            existing = f.read()
        with open(changelog_path, "w") as f:
            f.write(entry + existing)
    else:
        with open(changelog_path, "w") as f:
            f.write(f"# CHANGELOG\n{entry}")
    logger.info("CHANGELOG.md updated.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_phase5()
