# 📊 WorldCupAI — Feature Validation Report

Generated on: 2026-06-28 16:27:54

## High-Level Summary
- **Total Features Validated**: 72
- **Constant Features**: 0
- **High Missing Value Features (>10%)**: 6

## Detailed Feature Metrics
| Feature Name | Type | Missing % | Unique Values | Variance | Outliers | Constant? |
|---|---|---|---|---|---|---|
| `elo_diff` | float64 | 28.22% | 29,781 | 38036.5548 | 597 | No |
| `elo_ratio` | float64 | 28.22% | 29,781 | 0.0168 | 815 | No |
| `rank_diff` | float64 | 8.66% | 395 | 3220.3211 | 878 | No |
| `rank_ratio` | float64 | 8.66% | 7,176 | 25.883 | 4,073 | No |
| `overall_diff` | float64 | 28.22% | 3,723 | 69.7713 | 308 | No |
| `overall_ratio` | float64 | 28.22% | 10,785 | 0.0151 | 497 | No |
| `attack_diff` | float64 | 28.22% | 4,213 | 77.8307 | 493 | No |
| `defense_diff` | float64 | 28.22% | 5,100 | 70.8097 | 264 | No |
| `home_form_win_rate_5` | float64 | 0.0% | 11 | 0.063 | 0 | No |
| `home_form_loss_rate_5` | float64 | 0.0% | 11 | 0.0678 | 0 | No |
| `home_form_draw_rate_5` | float64 | 0.0% | 11 | 0.038 | 0 | No |
| `home_form_avg_goals_scored_5` | float64 | 0.0% | 74 | 0.6857 | 1,168 | No |
| `home_form_avg_goals_conceded_5` | float64 | 0.0% | 101 | 0.8541 | 1,366 | No |
| `home_form_goal_diff_5` | float64 | 0.0% | 160 | 1.9686 | 2,015 | No |
| `home_form_clean_sheet_rate_5` | float64 | 0.0% | 11 | 0.055 | 2,799 | No |
| `home_form_avg_points_5` | float64 | 0.0% | 28 | 0.5054 | 0 | No |
| `home_form_win_rate_10` | float64 | 0.0% | 27 | 0.039 | 62 | No |
| `home_form_loss_rate_10` | float64 | 0.0% | 30 | 0.0447 | 337 | No |
| `home_form_draw_rate_10` | float64 | 0.0% | 25 | 0.0202 | 154 | No |
| `home_form_avg_goals_scored_10` | float64 | 0.0% | 118 | 0.4243 | 975 | No |
| `home_form_avg_goals_conceded_10` | float64 | 0.0% | 196 | 0.6208 | 2,172 | No |
| `home_form_goal_diff_10` | float64 | 0.0% | 265 | 1.3724 | 1,764 | No |
| `home_form_clean_sheet_rate_10` | float64 | 0.0% | 25 | 0.0325 | 10 | No |
| `home_form_avg_points_10` | float64 | 0.0% | 73 | 0.328 | 0 | No |
| `away_form_win_rate_5` | float64 | 0.0% | 11 | 0.0617 | 0 | No |
| `away_form_loss_rate_5` | float64 | 0.0% | 11 | 0.0687 | 0 | No |
| `away_form_draw_rate_5` | float64 | 0.0% | 11 | 0.0391 | 0 | No |
| `away_form_avg_goals_scored_5` | float64 | 0.0% | 68 | 0.6573 | 1,027 | No |
| `away_form_avg_goals_conceded_5` | float64 | 0.0% | 104 | 0.9407 | 1,600 | No |
| `away_form_goal_diff_5` | float64 | 0.0% | 155 | 2.0306 | 1,396 | No |
| `away_form_clean_sheet_rate_5` | float64 | 0.0% | 11 | 0.0541 | 2,535 | No |
| `away_form_avg_points_5` | float64 | 0.0% | 28 | 0.4983 | 0 | No |
| `away_form_win_rate_10` | float64 | 0.0% | 27 | 0.0387 | 44 | No |
| `away_form_loss_rate_10` | float64 | 0.0% | 29 | 0.0454 | 381 | No |
| `away_form_draw_rate_10` | float64 | 0.0% | 24 | 0.0203 | 155 | No |
| `away_form_avg_goals_scored_10` | float64 | 0.0% | 122 | 0.415 | 914 | No |
| `away_form_avg_goals_conceded_10` | float64 | 0.0% | 193 | 0.6757 | 1,865 | No |
| `away_form_goal_diff_10` | float64 | 0.0% | 260 | 1.4289 | 1,695 | No |
| `away_form_clean_sheet_rate_10` | float64 | 0.0% | 24 | 0.0322 | 433 | No |
| `away_form_avg_points_10` | float64 | 0.0% | 71 | 0.3277 | 0 | No |
| `h2h_meetings` | int64 | 0.0% | 79 | 81.2671 | 3,069 | No |
| `h2h_draws` | float64 | 0.0% | 26 | 8.0081 | 3,660 | No |
| `h2h_home_wins` | int64 | 0.0% | 38 | 15.4458 | 2,228 | No |
| `h2h_away_wins` | int64 | 0.0% | 37 | 14.7146 | 3,858 | No |
| `h2h_gd` | float64 | 0.0% | 171 | 116.5151 | 6,469 | No |
| `home_attack_rating` | float64 | 0.0% | 118 | 0.2165 | 975 | No |
| `away_attack_rating` | float64 | 0.0% | 122 | 0.2117 | 914 | No |
| `attack_ratio` | float64 | 0.0% | 1,929 | 0.1443 | 1,114 | No |
| `home_defence_rating` | float64 | 0.0% | 195 | 0.9181 | 2,866 | No |
| `away_defence_rating` | float64 | 0.0% | 192 | 0.8098 | 2,531 | No |
| `home_clean_sheet_rate` | float64 | 0.0% | 25 | 0.0325 | 11 | No |
| `away_clean_sheet_rate` | float64 | 0.0% | 24 | 0.0322 | 433 | No |
| `home_world_cup_titles_before` | float64 | 0.0% | 6 | 0.3827 | 2,253 | No |
| `away_world_cup_titles_before` | float64 | 0.0% | 6 | 0.3148 | 1,938 | No |
| `home_world_cup_participations_before` | float64 | 0.0% | 23 | 13.9673 | 8,493 | No |
| `away_world_cup_participations_before` | float64 | 0.0% | 22 | 11.5744 | 7,610 | No |
| `home_groups_passed_before` | float64 | 0.0% | 20 | 7.8993 | 6,890 | No |
| `away_groups_passed_before` | float64 | 0.0% | 19 | 6.5506 | 6,132 | No |
| `home_round16_before` | float64 | 0.0% | 13 | 2.522 | 6,654 | No |
| `away_round16_before` | float64 | 0.0% | 12 | 2.0022 | 5,889 | No |
| `home_quarterfinals_before` | float64 | 0.0% | 18 | 4.2063 | 5,569 | No |
| `away_quarterfinals_before` | float64 | 0.0% | 17 | 3.4217 | 4,907 | No |
| `home_semifinals_before` | float64 | 0.0% | 13 | 2.266 | 4,087 | No |
| `away_semifinals_before` | float64 | 0.0% | 13 | 1.8408 | 3,576 | No |
| `home_finals_before` | float64 | 0.0% | 9 | 0.9568 | 2,613 | No |
| `away_finals_before` | float64 | 0.0% | 9 | 0.7628 | 2,314 | No |
| `is_neutral` | int64 | 0.0% | 2 | 0.2009 | 0 | No |
| `is_world_cup` | int64 | 0.0% | 2 | 0.1696 | 8,982 | No |
| `is_friendly` | int64 | 0.0% | 2 | 0.2257 | 0 | No |
| `home_rest_days` | float64 | 0.0% | 31 | 151.8371 | 0 | No |
| `away_rest_days` | float64 | 0.0% | 31 | 153.3468 | 0 | No |
| `rest_difference` | float64 | 0.0% | 61 | 98.3072 | 17,689 | No |


## Quality Analysis
- **Rankings Missingness**: FIFA rank features (`home_rank`, `away_rank`) naturally contain missing values for historical matches before 1993. This is expected.
- **Outliers**: Outliers are present in rolling rest days and goal differences. These represent extreme congestion or high-scoring matches and should be handled by robust scaling in the ML phase.
