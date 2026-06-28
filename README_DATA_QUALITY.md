# 📈 WorldCupAI — Data Quality Report

Generated on: 2026-06-28 16:22:32

## High-Level Quality Metrics

| Metric | Value |
|---|---|
| **Total Rows** | 41,491 |
| **Total Columns** | 43 |
| **Duplicate Rows** | 0 |
| **Unique Teams** | 330 |
| **Unique Tournaments** | 181 |
| **Unique Years** | 57 |
| **Memory Usage** | 15.35 MB |

## Missing Values Analysis

| Column | Missing Count | Missing Percentage |
|---|---|---|
| `home_elo` | 11,709 | 28.22% |
| `away_elo` | 11,709 | 28.22% |
| `elo_diff` | 11,709 | 28.22% |
| `home_avg_overall` | 11,709 | 28.22% |
| `home_max_overall` | 11,709 | 28.22% |
| `home_avg_attack` | 11,709 | 28.22% |
| `home_avg_defense` | 11,709 | 28.22% |
| `home_avg_pace` | 12,226 | 29.47% |
| `home_avg_shooting` | 12,226 | 29.47% |
| `home_avg_passing` | 12,226 | 29.47% |
| `away_avg_overall` | 11,709 | 28.22% |
| `away_max_overall` | 11,709 | 28.22% |
| `away_avg_attack` | 11,709 | 28.22% |
| `away_avg_defense` | 11,709 | 28.22% |
| `away_avg_pace` | 12,244 | 29.51% |
| `away_avg_shooting` | 12,244 | 29.51% |
| `away_avg_passing` | 12,244 | 29.51% |
| `overall_diff` | 11,709 | 28.22% |
| `attack_diff` | 11,709 | 28.22% |
| `defense_diff` | 11,709 | 28.22% |
| `home_form_scored` | 11,709 | 28.22% |
| `home_form_conceded` | 11,709 | 28.22% |
| `home_form_win_rate` | 11,709 | 28.22% |
| `away_form_scored` | 11,709 | 28.22% |
| `away_form_conceded` | 11,709 | 28.22% |
| `away_form_win_rate` | 11,709 | 28.22% |
| `is_neutral` | 11,709 | 28.22% |
| `is_world_cup` | 11,709 | 28.22% |
| `is_continental` | 11,709 | 28.22% |
| `home_rank` | 2,233 | 5.38% |
| `home_fifa_points` | 2,233 | 5.38% |
| `away_rank` | 2,369 | 5.71% |
| `away_fifa_points` | 2,369 | 5.71% |
| `rank_diff` | 3,595 | 8.66% |


## Data Quality Insights
- **FIFA Rankings Missingness**: Rankings columns (`home_rank`, `away_rank`) have missing values for matches played before the introduction of the official FIFA rankings in August 1993. This is normal and expected.
- **Match Features Missingness**: Missing values in ELO or EA FC attribute columns indicate matches that could not be matched with the pre-computed feature database (often due to friendly matches or minor tournaments not covered).
