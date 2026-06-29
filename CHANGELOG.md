# CHANGELOG

## [Neutral Venue Fix] — 2026-06-29 19:04:27

### Fixed
- `is_neutral` hardcoded to `1` in `src/features/context_features.py` for all FIFA WC 2026 matches
- Symmetric prediction averaging implemented in `src/prediction/knockout_engine.py`
- `PredictionCache` swap logic updated in `src/simulation/probability_sampler.py`
- Dashboard bracket display updated to show `prob_team1_win`/`prob_team2_win`
- `xai.py` and `app.py` updated to use neutral team1/team2 labels
- Outcome strings in probability sampler now use team name (e.g. "France Win" vs "Home Win")


## [Phase 8] — 2026-06-29 17:59:11

### Added
- Predefined authoritative JSON knockout schedule `configs/knockout_fixtures.json`
- `src/prediction/` prediction package containing `KnockoutEngine`
- `predict_tournament.py` orchestrator script
- JSON bracket, tree, and summary files for Streamlit dashboard
- `tests/test_phase8.py` automated test suite
- Exhaustive documentation files (`README_PHASE8.md`, `README_KNOCKOUT_ENGINE.md`, `README_MATCH_PREDICTIONS.md`, `TOURNAMENT_PREDICTION_REPORT.md`, `MATCH_EXPLANATIONS.md`, `PHASE_8_APPROVAL_REPORT.md`)
# CHANGELOG

## [Phase 11] - Production Streamlit Dashboard

### Added
- Streamlit web platform main entry router `app.py`.
- Complete `dashboard/` directory modules:
  - `data_loader.py` for caching files and models registry.
  - `sidebar.py` for Royal Blue & Gold theme navigation.
  - `cards.py` for glassmorphism summaries.
  - `bracket.py` for widescreen tree grids.
  - `reports.py` for Markdown and CSV downloads.
  - `export.py` for tabular exports.
  - `xai.py` for interactive Plotly explanations.
  - `performance.py` for leaderboard comparisons.
  - `settings.py` for folder configs and cache clearing.
- Multi-page validation test suite `tests/test_dashboard.py`.
- Platform guides (`README_STREAMLIT.md`, `README_DASHBOARD.md`, `DASHBOARD_USER_GUIDE.md`, `PHASE_11_APPROVAL_REPORT.md`).

## [Phase 10] - Explainable AI & Decision Intelligence

### Added
- Complete `src/explainability/` suite:
  - `feature_importance.py` for global metrics extraction.
  - `global_explanations.py` for correlation and stability checks.
  - `local_explanations.py` for match attribution and natural language narratives.
  - `confidence_analysis.py` for entropy risk classification.
  - `counterfactuals.py` for what-if scenarios.
  - `ensemble_explanations.py` for voter probability breakdowns.
  - `tournament_explanations.py` for champion path analytics.
  - `visualizations.py` for 8 production plots.
  - `report_generator.py` for markdown reporting.
- High-quality XAI dashboard reports under root.
## [Phase 8] — 2026-06-28 20:25:51

### Added
- Predefined authoritative JSON knockout schedule `configs/knockout_fixtures.json`
- `src/prediction/` prediction package containing `KnockoutEngine`
- `predict_tournament.py` orchestrator script
- JSON bracket, tree, and summary files for Streamlit dashboard
- `tests/test_phase8.py` automated test suite
- Exhaustive documentation files (`README_PHASE8.md`, `README_KNOCKOUT_ENGINE.md`, `README_MATCH_PREDICTIONS.md`, `TOURNAMENT_PREDICTION_REPORT.md`, `MATCH_EXPLANATIONS.md`, `PHASE_8_APPROVAL_REPORT.md`)
# CHANGELOG

## [Phase 8] — 2026-06-28 20:18:24

### Added
- Predefined authoritative JSON knockout schedule `configs/knockout_fixtures.json`
- `src/prediction/` prediction package containing `KnockoutEngine`
- `predict_tournament.py` orchestrator script
- JSON bracket, tree, and summary files for Streamlit dashboard
- `tests/test_phase8.py` automated test suite
- Exhaustive documentation files (`README_PHASE8.md`, `README_KNOCKOUT_ENGINE.md`, `README_MATCH_PREDICTIONS.md`, `TOURNAMENT_PREDICTION_REPORT.md`, `MATCH_EXPLANATIONS.md`, `PHASE_8_APPROVAL_REPORT.md`)
# CHANGELOG

## [Phase 7.1] — 2026-06-28 19:52:08

### Added
- Multi-objective weight optimization strategy in `EnsembleWeightOptimizer`
- Constrained optimization bounds (Top-K, min_weight, max_weight)
- Expanded diversity metrics (disagreement rate, Q-statistic, double-fault, pairwise KL)
- McNemar test & bootstrap confidence interval statistical tools
- STATISTICAL_ANALYSIS.md, MODEL_DIVERSITY_REPORT.md, ENSEMBLE_EXPLAINABILITY.md, PRODUCTION_VALIDATION.md
- New test suite at `tests/test_phase7_improvements.py`

# CHANGELOG

## [Phase 7] — 2026-06-28 19:44:29

### Added
- `src/ensemble/` — ensemble package (diversity, voting, stacking, optimizer, evaluator, pipeline, export_val)
- `train_ensemble.py` — Phase 7 orchestration script
- `tests/test_phase7.py` — automated validation tests
- `models/ensemble/` — all ensemble artifacts (ensemble.pkl, config, weights, metrics, predictions, calibration, heatmaps)
- `README_PHASE7.md`, `README_ENSEMBLE.md`, `ENSEMBLE_COMPARISON.md`, `ENSEMBLE_SELECTION_REPORT.md`
- `PHASE_7_APPROVAL_REPORT.md`

# CHANGELOG

## [Phase 6] — 2026-06-28 17:37:57

### Added
- `src/deep_learning/` — full DL package (base_model, ann_model, lstm_model, dataset, losses, metrics, callbacks, trainer, evaluation, model_registry, prediction_interface)
- `train_deep_learning.py` — Phase 6 orchestration script
- `tests/test_phase6.py` — automated validation tests
- `models/ann/` — ANN model artifacts (model.pt, model_complete.pt, model_config.json, metrics.json, ...)
- `models/lstm/` — LSTM model artifacts
- `predictions/` — prediction CSVs for all models (DL + ML, 7 total)
- `README_PHASE6.md`, `README_DEEP_LEARNING.md`, `README_ANN.md`, `README_LSTM.md`
- `DEEP_LEARNING_BENCHMARK.md`, `DL_VS_ML_COMPARISON.md`, `PHASE6_ERROR_ANALYSIS.md`
- `PHASE_6_APPROVAL_REPORT.md`

### Framework
- PyTorch 2.9.0 with Apple MPS (Metal GPU) acceleration


## [Phase 5] — 2026-06-28 17:05:20

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

# Changelog

All notable changes to the WorldCupAI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.3.0] — 2026-06-28

### Phase 3 — Advanced Feature Engineering & Model-Ready Feature Store

#### Added
- Modular feature engineering package in `src/features/` (`base_features.py`, `team_strength.py`, `recent_form.py`, `head_to_head.py`, `attack_features.py`, `defence_features.py`, `tournament_features.py`, `context_features.py`, `feature_store.py`, `feature_validator.py`, `feature_selector.py`, `feature_dictionary.py`).
- 72 engineered features spanning Team Strength, Recent Form, Head-to-Head, Attack, Defence, Tournament Experience, and Context.
- Chronological shift protocol (`.shift(1)`) to guarantee 100% temporal safety and prevent leakage.
- Versioned feature store at `processed/feature_store_v1.0.parquet` and `processed/feature_store_v1.0.csv` containing 41,491 rows and 80 columns.
- Feature selection analysis using Correlation, Mutual Information, and Decision Tree baseline importances.
- Documentation and validation reports:
  - `README_PHASE3.md`
  - `README_FEATURE_ENGINEERING.md`
  - `README_FEATURE_STORE.md`
  - `README_FEATURE_VALIDATION.md`
  - `FEATURE_DICTIONARY.md`
  - `FEATURE_SELECTION_REPORT.md`
  - `FEATURE_LEAKAGE_REPORT.md`
  - `PHASE_3_APPROVAL_REPORT.md`

## [0.2.0] — 2026-06-28

### Phase 2 — Data Engineering Pipeline & Master Dataset Construction

#### Added
- Modular ETL pipeline package in `src/data/` (`loader.py`, `cleaner.py`, `harmonizer.py`, `validator.py`, `merger.py`, `exporter.py`).
- YAML-based data configuration in `configs/data_config.yaml`.
- Team name entity resolution mapping saved to `mappings/team_name_mapping.csv`.
- Strict temporal validation checking for future leakage.
- Master dataset generated in Parquet and CSV formats at `processed/master_dataset.parquet` and `processed/master_dataset.csv`.
- Automated test suite in `tests/test_pipeline.py` verifying primary keys, dates, scores, and leakage.
- Documentation suite for Phase 2:
  - `README_PHASE2.md`
  - `README_MASTER_DATASET.md`
  - `README_MAPPING.md`
  - `README_DATA_QUALITY.md`
  - `reports/README_DATA_PIPELINE.md`
  - `reports/README_TEMPORAL_VALIDATION.md`
  - `PIPELINE_SUMMARY.md`

## [0.1.0] — 2026-06-28

### Phase 1 — Foundation, Architecture & Dataset Intelligence

#### Added
- Complete workspace discovery: 113 CSV files identified across 5 data source families
- Full dataset audit: rows, columns, memory, data types, missing values, duplicates profiled for every CSV
- Duplicate analysis: 43 exact-duplicate file pairs detected between `FIFA World Cup Historical Dataset/` and `worldcup-master/`
- Data leakage analysis: 80+ columns flagged and classified as Allowed / Conditional / Forbidden
- Relationship discovery: shared columns mapped across all datasets; foreign key candidates identified
- Dataset catalogue: every dataset classified as Canonical / Supporting / Validation / Ignore
- Production repository architecture proposed with 12 top-level directories
- Documentation suite created:
  - `README_PROJECT.md` — Master project overview
  - `README_DATASETS.md` — Dataset catalogue
  - `README_DATA_AUDIT.md` — Full audit report
  - `README_ARCHITECTURE.md` — Repository structure
  - `README_PHASE1.md` — Phase 1 approval report
  - `CHANGELOG.md` — This file

#### Identified Issues
- `worldcup-master/` is a byte-for-byte duplicate of `FIFA World Cup Historical Dataset/` (43 file pairs, all MD5-verified)
- `test.csv` has 100% missing target columns (winner, finalist, semi_finalist, quarter_finalist) — expected for 2026 prediction
- Transfermarkt `game_lineups.csv` (335 MB, 3.17M rows) — very large; requires chunked processing
- Team name inconsistencies across data sources (e.g., "USA" vs "United States" vs country codes)

#### Recommendations
- Designate `FIFA World Cup Historical Dataset/data-csv/` as canonical; archive `worldcup-master/`
- Proceed to Phase 2: Data Engineering, Cleaning & Feature Store
