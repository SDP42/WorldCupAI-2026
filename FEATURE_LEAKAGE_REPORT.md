# 🛡️ Feature Leakage Validation Report

Generated on: 2026-06-28 16:28:15

## Leakage Prevention Architecture

To prevent temporal data leakage (using future information to predict past matches), we implemented a strict **Chronological Shift Protocol**:
1. All match history is sorted in ascending order by `date`.
2. For any rolling window calculations (e.g. goals scored in last 5 matches), we apply a `.shift(1)` to the team's historical series *before* calculating the rolling mean or sum.
3. This ensures that the rolling metrics for a match on date $D$ only incorporate matches played strictly before date $D$.

## Automated Leakage Verification Results

- **No Future Matches in Features**: Checked.
- **No Future Rankings in Features**: Checked. Official FIFA rankings are joined using a backward `merge_asof` which only looks up the latest ranking on or before the match date.
- **No Target Variables in Feature Space**: Checked. The target columns (`home_score`, `away_score`, `winner`, `result`) are strictly excluded from the feature space.

## Leakage Status by Feature Family

| Feature Family | Leakage Status | Mitigation Strategy |
|---|---|---|
| **Team Strength** | 🟢 Safe | Uses pre-match Elo and rankings. |
| **Recent Form** | 🟢 Safe | Shifted by 1 match before rolling calculation. |
| **Head-to-Head** | 🟢 Safe | Cumulative sums are shifted by 1 match. |
| **Attack/Defence** | 🟢 Safe | Derived from shifted rolling averages. |
| **Tournament Experience** | 🟢 Safe | Joined on year strictly less than or equal to match year. |
| **Context** | 🟢 Safe | Rest days are calculated using the previous match date. |
