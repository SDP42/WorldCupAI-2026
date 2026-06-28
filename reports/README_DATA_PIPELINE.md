# 🔗 Data Pipeline Merge Report

Generated on: 2026-06-28 16:22:32

## Merge Steps & Join Coverage

| Step Name | Rows Before | Rows After | Row Change | Coverage | Join Keys | Total Missing Cells |
|---|---|---|---|---|---|---|
| **Match Features Join** | 41,491 | 41,491 | 0 | 100.0% | `['date', 'home_team', 'away_team']` | 342,717 |
| **FIFA Rankings Temporal Join** | 41,491 | 41,491 | 0 | 100.0% | `['date', 'home_team/away_team']` | 355,516 |

## Technical Analysis & Recommendations
- The **Match Features Join** maps historical results to Elo and EA FC player attributes. Any unmatched matches will have NaN values for these features.
- The **FIFA Rankings Temporal Join** uses a backward `merge_asof` to lookup rankings. Matches played before the first official FIFA ranking date (August 1993) will naturally have NaN rankings. This is expected behavior and must be handled by downstream models.
