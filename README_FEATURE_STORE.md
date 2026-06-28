# 📦 WorldCupAI — Feature Store Documentation

This document describes the structure, versioning, and usage of the WorldCupAI Feature Store.

## 1. Feature Store Files

The feature store is exported to both Parquet and CSV formats in the `processed/` directory.

### Paths
- **Canonical Parquet**: [`processed/feature_store.parquet`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/processed/feature_store.parquet) (Primary binary storage for ML/DL)
- **Canonical CSV**: [`processed/feature_store.csv`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/processed/feature_store.csv) (Human-readable compatibility storage)
- **Versioned Parquet**: [`processed/feature_store_v1.0.parquet`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/processed/feature_store_v1.0.parquet)
- **Versioned CSV**: [`processed/feature_store_v1.0.csv`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/processed/feature_store_v1.0.csv)

---

## 2. Feature Store Properties

- **Row Count**: 41,491 matches
- **Column Count**: 80 columns (8 base metadata/label columns + 72 engineered feature columns)
- **Primary Key**: `date` + `home_team` + `away_team`

---

## 3. How to Load and Use the Feature Store

### Python (pandas)
```python
import pandas as pd

# Load the canonical feature store
df = pd.read_parquet("processed/feature_store.parquet")

# Separate features and target labels
base_cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament', 'city', 'country']
features = df.drop(columns=base_cols)
targets = df[['home_score', 'away_score']]

print(f"Features shape: {features.shape}")
print(f"Targets shape: {targets.shape}")
```

### Reproducibility & Versioning
Every run of the feature pipeline generates a versioned feature store (e.g. `v1.0`, `v2.0`). This allows us to track feature lineage and verify that model improvements are due to algorithmic changes rather than data drift.
