# 📋 WorldCupAI — Pipeline Summary Report

Generated on: 2026-06-28 16:22:32

## Input Datasets & Row Counts
| Dataset | Input Rows |
|---|---|
| `results.csv` | 49,477 |
| `teams_match_features.csv` | 43,364 |
| `fifa_mens_rank.csv` | 13,130 |


## Pipeline Processing Metrics

| Metric | Count |
|---|---|
| **Cleaned & Filtered Base Matches (since 1970)** | 41,491 |
| **Final Master Dataset Rows** | 41,491 |
| **Final Master Dataset Columns** | 43 |
| **Parquet Output Path** | `processed/master_dataset.parquet` |
| **CSV Output Path** | `processed/master_dataset.csv` |
| **Join Coverage (Match Features)** | 100.0% (Left Joined) |

## Remaining Issues & Recommendations
1. **FIFA Rankings pre-1993**: Rankings columns (`home_rank`, `away_rank`) are naturally null for matches before August 1993. Downstream models should handle these missing values (e.g. via imputation or by using Elo rating as the primary strength feature).
2. **Team Name Mapping**: The entity resolution mapping (`mappings/team_name_mapping.csv`) should be continuously updated as new team names or historical variations are discovered.
