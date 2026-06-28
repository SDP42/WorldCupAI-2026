# 🔍 WorldCupAI — Feature Selection Report

Generated on: 2026-06-28 16:28:15

## Summary
- **Total Features Analyzed**: 72
- **Features Retained (Keep)**: 35
- **Features Flagged for Review**: 13
- **Features Discarded (Collinear)**: 24

## Collinear Feature Pairs (>0.9 Correlation)
| Feature 1 | Feature 2 | Correlation |
|---|---|---|
| `elo_diff` | `elo_ratio` | 0.9880 |
| `overall_diff` | `overall_ratio` | 0.9930 |
| `overall_diff` | `attack_diff` | 0.9587 |
| `overall_ratio` | `attack_diff` | 0.9530 |
| `overall_diff` | `defense_diff` | 0.9625 |
| `overall_ratio` | `defense_diff` | 0.9549 |
| `home_form_win_rate_5` | `home_form_avg_points_5` | 0.9662 |
| `home_form_win_rate_10` | `home_form_avg_points_10` | 0.9708 |
| `away_form_win_rate_5` | `away_form_avg_points_5` | 0.9644 |
| `away_form_win_rate_10` | `away_form_avg_points_10` | 0.9703 |
| `home_form_avg_goals_scored_10` | `home_attack_rating` | 1.0000 |
| `away_form_avg_goals_scored_10` | `away_attack_rating` | 1.0000 |
| `home_form_clean_sheet_rate_10` | `home_clean_sheet_rate` | 1.0000 |
| `away_form_clean_sheet_rate_10` | `away_clean_sheet_rate` | 1.0000 |
| `home_world_cup_participations_before` | `home_groups_passed_before` | 0.9492 |
| `away_world_cup_participations_before` | `away_groups_passed_before` | 0.9518 |
| `home_world_cup_participations_before` | `home_round16_before` | 0.9538 |
| `home_groups_passed_before` | `home_round16_before` | 0.9573 |
| `away_world_cup_participations_before` | `away_round16_before` | 0.9500 |
| `away_groups_passed_before` | `away_round16_before` | 0.9529 |
| `home_world_cup_titles_before` | `home_quarterfinals_before` | 0.9094 |
| `home_groups_passed_before` | `home_quarterfinals_before` | 0.9651 |
| `away_world_cup_titles_before` | `away_quarterfinals_before` | 0.9018 |
| `away_groups_passed_before` | `away_quarterfinals_before` | 0.9648 |
| `home_world_cup_titles_before` | `home_semifinals_before` | 0.9357 |
| `home_groups_passed_before` | `home_semifinals_before` | 0.9011 |
| `home_quarterfinals_before` | `home_semifinals_before` | 0.9587 |
| `away_world_cup_titles_before` | `away_semifinals_before` | 0.9277 |
| `away_groups_passed_before` | `away_semifinals_before` | 0.9042 |
| `away_quarterfinals_before` | `away_semifinals_before` | 0.9589 |
| `home_world_cup_titles_before` | `home_finals_before` | 0.9462 |
| `home_quarterfinals_before` | `home_finals_before` | 0.9247 |
| `home_semifinals_before` | `home_finals_before` | 0.9595 |
| `away_world_cup_titles_before` | `away_finals_before` | 0.9391 |
| `away_quarterfinals_before` | `away_finals_before` | 0.9191 |
| `away_semifinals_before` | `away_finals_before` | 0.9538 |


## Feature Recommendations & Rankings
| Feature Name | Mutual Information | Tree Importance | Recommendation | Rationale |
|---|---|---|---|---|
| `elo_ratio` | 0.0932 | 0.1820 | Keep 🟢 | Strong predictive power and low collinearity. |
| `elo_diff` | 0.0933 | 0.1633 | Discard ❌ | High collinearity with another feature. |
| `rank_diff` | 0.0847 | 0.1258 | Keep 🟢 | Strong predictive power and low collinearity. |
| `rank_ratio` | 0.0990 | 0.0773 | Keep 🟢 | Strong predictive power and low collinearity. |
| `overall_ratio` | 0.0890 | 0.0690 | Keep 🟢 | Strong predictive power and low collinearity. |
| `attack_diff` | 0.0721 | 0.0474 | Discard ❌ | High collinearity with another feature. |
| `h2h_gd` | 0.0459 | 0.0450 | Keep 🟢 | Strong predictive power and low collinearity. |
| `overall_diff` | 0.0744 | 0.0338 | Discard ❌ | High collinearity with another feature. |
| `defense_diff` | 0.0770 | 0.0338 | Discard ❌ | High collinearity with another feature. |
| `h2h_away_wins` | 0.0195 | 0.0258 | Keep 🟢 | Strong predictive power and low collinearity. |
| `is_neutral` | 0.0004 | 0.0219 | Keep 🟢 | Strong predictive power and low collinearity. |
| `away_world_cup_participations_before` | 0.0154 | 0.0141 | Keep 🟢 | Strong predictive power and low collinearity. |
| `h2h_home_wins` | 0.0160 | 0.0091 | Keep 🟢 | Strong predictive power and low collinearity. |
| `away_groups_passed_before` | 0.0124 | 0.0081 | Discard ❌ | High collinearity with another feature. |
| `away_form_avg_goals_conceded_10` | 0.0089 | 0.0070 | Keep 🟢 | Strong predictive power and low collinearity. |
| `away_round16_before` | 0.0091 | 0.0069 | Discard ❌ | High collinearity with another feature. |
| `h2h_meetings` | 0.0008 | 0.0062 | Keep 🟢 | Strong predictive power and low collinearity. |
| `h2h_draws` | 0.0000 | 0.0059 | Keep 🟢 | Strong predictive power and low collinearity. |
| `away_defence_rating` | 0.0084 | 0.0048 | Keep 🟢 | Strong predictive power and low collinearity. |
| `away_form_goal_diff_5` | 0.0018 | 0.0048 | Keep 🟢 | Strong predictive power and low collinearity. |
| `away_form_goal_diff_10` | 0.0067 | 0.0047 | Keep 🟢 | Strong predictive power and low collinearity. |
| `attack_ratio` | 0.0000 | 0.0045 | Keep 🟢 | Strong predictive power and low collinearity. |
| `home_round16_before` | 0.0078 | 0.0041 | Keep 🟢 | Strong predictive power and low collinearity. |
| `away_form_avg_goals_conceded_5` | 0.0060 | 0.0039 | Keep 🟢 | Strong predictive power and low collinearity. |
| `is_friendly` | 0.0007 | 0.0036 | Keep 🟢 | Strong predictive power and low collinearity. |
| `away_semifinals_before` | 0.0090 | 0.0035 | Discard ❌ | High collinearity with another feature. |
| `home_form_goal_diff_10` | 0.0018 | 0.0034 | Keep 🟢 | Strong predictive power and low collinearity. |
| `home_world_cup_participations_before` | 0.0057 | 0.0033 | Discard ❌ | High collinearity with another feature. |
| `home_groups_passed_before` | 0.0059 | 0.0033 | Discard ❌ | High collinearity with another feature. |
| `away_quarterfinals_before` | 0.0123 | 0.0032 | Discard ❌ | High collinearity with another feature. |
| `home_form_avg_goals_conceded_10` | 0.0000 | 0.0030 | Keep 🟢 | Strong predictive power and low collinearity. |
| `home_defence_rating` | 0.0032 | 0.0029 | Keep 🟢 | Strong predictive power and low collinearity. |
| `home_attack_rating` | 0.0036 | 0.0027 | Keep 🟢 | Strong predictive power and low collinearity. |
| `home_form_goal_diff_5` | 0.0021 | 0.0027 | Keep 🟢 | Strong predictive power and low collinearity. |
| `rest_difference` | 0.0003 | 0.0026 | Keep 🟢 | Strong predictive power and low collinearity. |
| `home_form_avg_goals_scored_10` | 0.0000 | 0.0025 | Discard ❌ | High collinearity with another feature. |
| `away_form_loss_rate_10` | 0.0018 | 0.0025 | Keep 🟢 | Strong predictive power and low collinearity. |
| `away_form_avg_goals_scored_10` | 0.0040 | 0.0024 | Keep 🟢 | Strong predictive power and low collinearity. |
| `away_rest_days` | 0.0000 | 0.0024 | Keep 🟢 | Strong predictive power and low collinearity. |
| `away_form_avg_goals_scored_5` | 0.0068 | 0.0024 | Keep 🟢 | Strong predictive power and low collinearity. |
| `away_form_loss_rate_5` | 0.0065 | 0.0024 | Keep 🟢 | Strong predictive power and low collinearity. |
| `away_form_avg_points_10` | 0.0056 | 0.0023 | Keep 🟢 | Strong predictive power and low collinearity. |
| `home_rest_days` | 0.0007 | 0.0023 | Keep 🟢 | Strong predictive power and low collinearity. |
| `home_form_avg_points_10` | 0.0025 | 0.0023 | Keep 🟢 | Strong predictive power and low collinearity. |
| `home_form_avg_goals_scored_5` | 0.0036 | 0.0022 | Keep 🟢 | Strong predictive power and low collinearity. |
| `away_attack_rating` | 0.0017 | 0.0021 | Discard ❌ | High collinearity with another feature. |
| `away_form_avg_points_5` | 0.0000 | 0.0021 | Keep 🟢 | Strong predictive power and low collinearity. |
| `home_form_avg_points_5` | 0.0020 | 0.0019 | Review ⚠️ | Very low predictive importance in baseline tree. |
| `away_form_win_rate_10` | 0.0103 | 0.0019 | Discard ❌ | High collinearity with another feature. |
| `home_form_avg_goals_conceded_5` | 0.0020 | 0.0019 | Review ⚠️ | Very low predictive importance in baseline tree. |
| `home_form_loss_rate_10` | 0.0006 | 0.0016 | Review ⚠️ | Very low predictive importance in baseline tree. |
| `away_clean_sheet_rate` | 0.0050 | 0.0015 | Review ⚠️ | Very low predictive importance in baseline tree. |
| `home_form_draw_rate_10` | 0.0000 | 0.0015 | Review ⚠️ | Very low predictive importance in baseline tree. |
| `away_finals_before` | 0.0057 | 0.0014 | Discard ❌ | High collinearity with another feature. |
| `home_form_win_rate_10` | 0.0007 | 0.0014 | Discard ❌ | High collinearity with another feature. |
| `home_clean_sheet_rate` | 0.0038 | 0.0013 | Review ⚠️ | Very low predictive importance in baseline tree. |
| `home_form_clean_sheet_rate_10` | 0.0000 | 0.0013 | Discard ❌ | High collinearity with another feature. |
| `home_quarterfinals_before` | 0.0003 | 0.0013 | Discard ❌ | High collinearity with another feature. |
| `away_form_draw_rate_10` | 0.0000 | 0.0012 | Review ⚠️ | Very low predictive importance in baseline tree. |
| `away_form_clean_sheet_rate_5` | 0.0043 | 0.0011 | Review ⚠️ | Very low predictive importance in baseline tree. |
| `home_form_win_rate_5` | 0.0017 | 0.0011 | Discard ❌ | High collinearity with another feature. |
| `away_form_clean_sheet_rate_10` | 0.0012 | 0.0010 | Discard ❌ | High collinearity with another feature. |
| `away_form_draw_rate_5` | 0.0015 | 0.0010 | Review ⚠️ | Very low predictive importance in baseline tree. |
| `is_world_cup` | 0.0000 | 0.0010 | Review ⚠️ | Very low predictive importance in baseline tree. |
| `home_form_clean_sheet_rate_5` | 0.0024 | 0.0008 | Review ⚠️ | Very low predictive importance in baseline tree. |
| `away_form_win_rate_5` | 0.0034 | 0.0008 | Discard ❌ | High collinearity with another feature. |
| `home_form_draw_rate_5` | 0.0074 | 0.0008 | Review ⚠️ | Very low predictive importance in baseline tree. |
| `home_form_loss_rate_5` | 0.0052 | 0.0008 | Review ⚠️ | Very low predictive importance in baseline tree. |
| `home_semifinals_before` | 0.0000 | 0.0007 | Discard ❌ | High collinearity with another feature. |
| `away_world_cup_titles_before` | 0.0065 | 0.0007 | Discard ❌ | High collinearity with another feature. |
| `home_finals_before` | 0.0004 | 0.0005 | Discard ❌ | High collinearity with another feature. |
| `home_world_cup_titles_before` | 0.0012 | 0.0003 | Discard ❌ | High collinearity with another feature. |

