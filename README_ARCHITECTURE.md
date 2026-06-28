# 🏗️ WorldCupAI — Production Repository Architecture

> Recommended directory structure for the WorldCupAI platform, designed for production ML workflows.

---

## Directory Layout

```
FIFA World Cup Prediction Project/
│
├── data/                          # 📦 Raw immutable datasets
│   ├── raw/                       #    Original datasets — never modified
│   │   ├── fjelstul_worldcup/     #    Fjelstul World Cup Historical Database (canonical)
│   │   ├── international_results/ #    International Football Results 1872–2026
│   │   ├── transfermarkt/         #    Football Data from Transfermarkt
│   │   ├── fifa_worldcup_dataset/ #    FIFA World Cup Dataset (train.csv, test.csv)
│   │   ├── elo_ratings/           #    Elo rating snapshots
│   │   ├── fifa_rankings/         #    FIFA Men's Rankings
│   │   ├── player_aggregates/     #    FIFA game player attribute aggregates
│   │   ├── teams_form/            #    Pre-computed team form metrics
│   │   └── teams_match_features/  #    Pre-merged match-level features
│   └── archive/                   #    Deprecated/duplicate datasets (worldcup-master/)
│
├── mappings/                      # 🗺️ Entity resolution & harmonization
│   ├── team_name_mapping.csv      #    Cross-source team name standardization
│   ├── country_code_mapping.csv   #    ISO codes ↔ team names ↔ FIFA codes
│   ├── player_id_mapping.csv      #    Cross-source player ID resolution
│   ├── tournament_id_mapping.csv  #    Tournament identifiers across sources
│   └── former_names.csv           #    Historical country name changes
│
├── processed/                     # 🧹 Cleaned, validated, standardized datasets
│   ├── matches_clean.parquet      #    Unified match dataset (all sources merged)
│   ├── teams_clean.parquet        #    Master team reference table
│   ├── players_clean.parquet      #    Unified player dataset
│   ├── tournaments_clean.parquet  #    Clean tournament metadata
│   └── README.md                  #    Processing changelog and lineage
│
├── feature_store/                 # 🧪 Engineered features ready for modelling
│   ├── team_features.parquet      #    Team-level features (rolling stats, Elo, rankings)
│   ├── match_features.parquet     #    Match-level pairwise features
│   ├── player_features.parquet    #    Aggregated player features per team
│   ├── temporal_features.parquet  #    Time-series features (form, momentum)
│   ├── feature_registry.yaml     #    Feature definitions, types, sources, versions
│   └── README.md                  #    Feature documentation
│
├── models/                        # 🤖 Trained models and artifacts
│   ├── ml/                        #    Traditional ML models (XGBoost, LightGBM, RF)
│   │   ├── xgboost_v1/
│   │   ├── lightgbm_v1/
│   │   └── logistic_v1/
│   ├── dl/                        #    Deep learning models (PyTorch/TF)
│   │   └── lstm_form_v1/
│   ├── ensemble/                  #    Ensemble configurations and weights
│   │   └── stacking_v1/
│   ├── calibration/               #    Probability calibration artifacts
│   └── model_registry.yaml       #    Model metadata, versions, metrics
│
├── reports/                       # 📊 Generated reports and visualizations
│   ├── eda/                       #    Exploratory data analysis outputs
│   ├── model_evaluation/          #    Model performance reports
│   ├── predictions/               #    Prediction outputs per round
│   │   ├── round_of_32/
│   │   ├── round_of_16/
│   │   ├── quarterfinals/
│   │   ├── semifinals/
│   │   └── final/
│   ├── explainability/            #    SHAP plots, feature importance, counterfactuals
│   └── phase_reports/             #    Phase completion reports
│
├── docs/                          # 📖 Project documentation
│   ├── README_PROJECT.md          #    Master project overview
│   ├── README_DATASETS.md         #    Dataset catalogue
│   ├── README_DATA_AUDIT.md       #    Full data audit report
│   ├── README_ARCHITECTURE.md     #    This file
│   ├── README_PHASE1.md           #    Phase 1 approval report
│   ├── CHANGELOG.md               #    Version history
│   └── decisions/                 #    Architecture Decision Records (ADRs)
│
├── notebooks/                     # 📓 Jupyter notebooks for exploration
│   ├── 01_eda_overview.ipynb
│   ├── 02_feature_exploration.ipynb
│   ├── 03_model_prototyping.ipynb
│   ├── 04_ensemble_analysis.ipynb
│   └── 05_explainability.ipynb
│
├── configs/                       # ⚙️ Configuration files
│   ├── data_config.yaml           #    Data source paths and parameters
│   ├── feature_config.yaml        #    Feature engineering parameters
│   ├── model_config.yaml          #    Model hyperparameters
│   ├── ensemble_config.yaml       #    Ensemble weights and strategy
│   ├── dashboard_config.yaml      #    Streamlit dashboard settings
│   └── logging_config.yaml        #    Logging configuration
│
├── logs/                          # 📋 Runtime logs
│   ├── data_pipeline.log
│   ├── training.log
│   ├── prediction.log
│   └── dashboard.log
│
├── dashboard/                     # 🖥️ Streamlit dashboard
│   ├── app.py                     #    Main Streamlit entry point
│   ├── pages/
│   │   ├── 01_overview.py
│   │   ├── 02_predictions.py
│   │   ├── 03_team_analysis.py
│   │   ├── 04_head_to_head.py
│   │   ├── 05_explainability.py
│   │   └── 06_bracket.py
│   ├── components/                #    Reusable UI components
│   ├── assets/                    #    Images, CSS, static files
│   └── utils/                     #    Dashboard helper functions
│
├── tests/                         # ✅ Test suite
│   ├── test_data_pipeline.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_predictions.py
│   ├── test_leakage.py            #    Dedicated leakage detection tests
│   └── conftest.py
│
├── src/                           # 🐍 Source code (Python package)
│   ├── __init__.py
│   ├── data/                      #    Data loading and cleaning modules
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── cleaner.py
│   │   ├── validator.py
│   │   └── harmonizer.py          #    Team/player name harmonization
│   ├── features/                  #    Feature engineering modules
│   │   ├── __init__.py
│   │   ├── team_features.py
│   │   ├── match_features.py
│   │   ├── player_features.py
│   │   ├── temporal_features.py
│   │   └── registry.py
│   ├── models/                    #    Model training and evaluation
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── ml_models.py
│   │   ├── dl_models.py
│   │   ├── ensemble.py
│   │   └── evaluator.py
│   ├── explainability/            #    XAI modules
│   │   ├── __init__.py
│   │   ├── shap_analysis.py
│   │   └── counterfactuals.py
│   ├── prediction/                #    Prediction pipeline
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   └── bracket.py
│   └── utils/                     #    Shared utilities
│       ├── __init__.py
│       ├── config.py
│       ├── logger.py
│       └── constants.py
│
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata
├── Makefile                       # Build/run shortcuts
└── .gitignore                     # Git ignore rules
```

---

## Folder Purpose Reference

| Folder | Purpose | Phase |
|--------|---------|-------|
| `data/raw/` | Store original, immutable datasets exactly as downloaded. Never modify files here. | Phase 1 |
| `data/archive/` | Deprecated or duplicate datasets. Kept for reference, excluded from pipelines. | Phase 1 |
| `mappings/` | Entity resolution tables for harmonizing team/player/tournament names across data sources. | Phase 2 |
| `processed/` | Cleaned, validated, type-corrected, deduplicated datasets. Parquet format for efficiency. | Phase 2 |
| `feature_store/` | Engineered features with full lineage tracking. Each feature has a defined source, computation, and version. | Phase 3 |
| `models/` | Trained model artifacts (weights, configs, metrics). Versioned by model type and training run. | Phase 4–5 |
| `reports/` | All generated outputs: EDA plots, model comparisons, predictions, SHAP analyses, phase reports. | Phase 3–7 |
| `docs/` | All documentation: README files, audit reports, architecture docs, decision records. | All phases |
| `notebooks/` | Jupyter notebooks for interactive exploration. Numbered by workflow order. Not part of production pipeline. | Phase 3–6 |
| `configs/` | YAML configuration files that drive pipeline behaviour. No hardcoded parameters in source code. | Phase 2+ |
| `logs/` | Timestamped runtime logs for debugging and audit trails. | Phase 2+ |
| `dashboard/` | Streamlit application with multi-page layout, reusable components, and assets. | Phase 7 |
| `tests/` | Pytest test suite covering data integrity, feature correctness, model behaviour, and leakage detection. | Phase 2+ |
| `src/` | Production Python package with clean module separation by domain. | Phase 2+ |

---

## Design Principles

1. **Immutability of raw data** — `data/raw/` is read-only. All transformations produce new files in `processed/` or `feature_store/`.

2. **Config-driven pipelines** — All parameters (rolling windows, model hyperparameters, file paths) live in `configs/*.yaml`, not hardcoded.

3. **Explicit lineage** — Every processed dataset and feature records its source files, transformation logic, and creation timestamp.

4. **Leakage-aware architecture** — Temporal guards are enforced at the pipeline level. Dedicated `test_leakage.py` validates no future data leaks into training.

5. **Parquet over CSV** — Processed data uses Parquet for type preservation, compression, and read performance.

6. **Modular source code** — `src/` is organized by concern (data, features, models, explainability, prediction) with clean interfaces.

7. **Versioned models** — Every model run is stored with its config, metrics, and artifacts. `model_registry.yaml` tracks the lineage.

---

*Architecture designed for WorldCupAI Phase 1 — 2026-06-28*
