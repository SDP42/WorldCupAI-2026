# 🛠️ WorldCupAI — Phase 2: Data Engineering Pipeline

> Complete production-grade data engineering pipeline that cleans, harmonizes, and merges all raw datasets into one validated master dataset.

---

## 1. Overview

Phase 2 implements the ETL (Extract, Transform, Load) pipeline for the WorldCupAI prediction platform. The pipeline takes raw, disparate datasets (including international match results, Elo ratings, FIFA rankings, and pre-computed features) and combines them into a single, cohesive, leakage-safe master dataset (`master_dataset.parquet` / `master_dataset.csv`).

All operations are fully reproducible, configuration-driven, and modularized.

---

## 2. Pipeline Architecture

The pipeline is split into independent, reusable modules located in `src/data/`:

| Module | Purpose |
|---|---|
| [`loader.py`](src/data/loader.py) | Loads raw CSV files and supports chunked loading for large datasets. |
| [`cleaner.py`](src/data/cleaner.py) | Trims whitespace, standardizes dates, removes duplicates, and handles missing/invalid values. |
| [`harmonizer.py`](src/data/harmonizer.py) | Standardizes team and country names across datasets using a master mapping. |
| [`validator.py`](src/data/validator.py) | Performs strict chronological and temporal leakage validation. |
| [`merger.py`](src/data/merger.py) | Merges datasets step-by-step using precise keys and temporal ASOF joins. |
| [`exporter.py`](src/data/exporter.py) | Exports to Parquet/CSV and generates data quality reports. |

---

## 3. How to Run the Pipeline

To execute the entire pipeline and run the verification tests, run the following commands:

```bash
# Execute the ETL pipeline
python3 run_pipeline.py

# Run the automated verification tests
python3 tests/test_pipeline.py
```

---

## 4. Key Outputs & Reports

The pipeline automatically generates the following files:

- **Master Dataset**:
  - [`processed/master_dataset.parquet`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/processed/master_dataset.parquet) (Primary binary storage)
  - [`processed/master_dataset.csv`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/processed/master_dataset.csv) (Compatibility storage)
- **Reports**:
  - [`PIPELINE_SUMMARY.md`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/PIPELINE_SUMMARY.md) — High-level summary of processed rows.
  - [`README_DATA_QUALITY.md`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/README_DATA_QUALITY.md) — Missing values, duplicates, and unique entity counts.
  - [`reports/README_DATA_PIPELINE.md`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/reports/README_DATA_PIPELINE.md) — Details on the merge steps and join coverage.
  - [`reports/README_TEMPORAL_VALIDATION.md`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/reports/README_TEMPORAL_VALIDATION.md) — Strict leakage check results.
  - [`mappings/team_name_mapping.csv`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/mappings/team_name_mapping.csv) — Entity resolution mapping table.
