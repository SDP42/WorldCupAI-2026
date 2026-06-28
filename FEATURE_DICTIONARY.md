# 📖 WorldCupAI — Feature Dictionary

This document serves as the authoritative registry for all engineered features in the WorldCupAI platform.

## Feature Registry Table
| Feature Name | Description | Formula / Logic | Source | Family | Type | Temporal Safe? | Expected Value |
|---|---|---|---|---|---|---|---|
| `elo_diff` | Difference in Elo ratings between home and away teams. | `home_elo - away_elo` | elo_ratings_wc2026.csv | Team Strength | float64 | Yes | Very High |
| `elo_ratio` | Ratio of home team Elo rating to away team Elo rating. | `home_elo / away_elo` | elo_ratings_wc2026.csv | Team Strength | float64 | Yes | High |
| `rank_diff` | Difference in official FIFA rank between home and away teams. | `home_rank - away_rank` | fifa_mens_rank.csv | Team Strength | float64 | Yes | Very High |
| `rank_ratio` | Ratio of home team FIFA rank to away team FIFA rank. | `home_rank / away_rank` | fifa_mens_rank.csv | Team Strength | float64 | Yes | High |
| `overall_diff` | Difference in average EA FC overall rating between home and away squads. | `home_avg_overall - away_avg_overall` | player_aggregates.csv | Team Strength | float64 | Yes | High |
| `overall_ratio` | Ratio of home team average EA FC overall rating to away team. | `home_avg_overall / away_avg_overall` | player_aggregates.csv | Team Strength | float64 | Yes | Medium |
| `attack_diff` | Difference in average EA FC attack rating between home and away squads. | `home_avg_attack - away_avg_attack` | player_aggregates.csv | Team Strength | float64 | Yes | Medium |
| `defense_diff` | Difference in average EA FC defense rating between home and away squads. | `home_avg_defense - away_avg_defense` | player_aggregates.csv | Team Strength | float64 | Yes | Medium |
| `home_form_win_rate_5` | Home team win rate over the last 5 matches prior to kickoff. | `wins_5 / 5` | results.csv (shifted) | Recent Form | float64 | Yes | High |
| `home_form_loss_rate_5` | Engineered feature home_form_loss_rate_5. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_form_draw_rate_5` | Engineered feature home_form_draw_rate_5. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_form_avg_goals_scored_5` | Home team average goals scored over the last 5 matches prior to kickoff. | `goals_scored_5 / 5` | results.csv (shifted) | Recent Form | float64 | Yes | High |
| `home_form_avg_goals_conceded_5` | Home team average goals conceded over the last 5 matches prior to kickoff. | `goals_conceded_5 / 5` | results.csv (shifted) | Recent Form | float64 | Yes | High |
| `home_form_goal_diff_5` | Engineered feature home_form_goal_diff_5. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_form_clean_sheet_rate_5` | Home team clean sheet rate over the last 5 matches prior to kickoff. | `clean_sheets_5 / 5` | results.csv (shifted) | Recent Form | float64 | Yes | Medium |
| `home_form_avg_points_5` | Engineered feature home_form_avg_points_5. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_form_win_rate_10` | Engineered feature home_form_win_rate_10. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_form_loss_rate_10` | Engineered feature home_form_loss_rate_10. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_form_draw_rate_10` | Engineered feature home_form_draw_rate_10. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_form_avg_goals_scored_10` | Engineered feature home_form_avg_goals_scored_10. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_form_avg_goals_conceded_10` | Engineered feature home_form_avg_goals_conceded_10. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_form_goal_diff_10` | Engineered feature home_form_goal_diff_10. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_form_clean_sheet_rate_10` | Engineered feature home_form_clean_sheet_rate_10. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_form_avg_points_10` | Engineered feature home_form_avg_points_10. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_form_win_rate_5` | Away team win rate over the last 5 matches prior to kickoff. | `wins_5 / 5` | results.csv (shifted) | Recent Form | float64 | Yes | High |
| `away_form_loss_rate_5` | Engineered feature away_form_loss_rate_5. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_form_draw_rate_5` | Engineered feature away_form_draw_rate_5. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_form_avg_goals_scored_5` | Away team average goals scored over the last 5 matches prior to kickoff. | `goals_scored_5 / 5` | results.csv (shifted) | Recent Form | float64 | Yes | High |
| `away_form_avg_goals_conceded_5` | Away team average goals conceded over the last 5 matches prior to kickoff. | `goals_conceded_5 / 5` | results.csv (shifted) | Recent Form | float64 | Yes | High |
| `away_form_goal_diff_5` | Engineered feature away_form_goal_diff_5. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_form_clean_sheet_rate_5` | Away team clean sheet rate over the last 5 matches prior to kickoff. | `clean_sheets_5 / 5` | results.csv (shifted) | Recent Form | float64 | Yes | Medium |
| `away_form_avg_points_5` | Engineered feature away_form_avg_points_5. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_form_win_rate_10` | Engineered feature away_form_win_rate_10. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_form_loss_rate_10` | Engineered feature away_form_loss_rate_10. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_form_draw_rate_10` | Engineered feature away_form_draw_rate_10. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_form_avg_goals_scored_10` | Engineered feature away_form_avg_goals_scored_10. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_form_avg_goals_conceded_10` | Engineered feature away_form_avg_goals_conceded_10. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_form_goal_diff_10` | Engineered feature away_form_goal_diff_10. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_form_clean_sheet_rate_10` | Engineered feature away_form_clean_sheet_rate_10. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_form_avg_points_10` | Engineered feature away_form_avg_points_10. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `h2h_meetings` | Total number of previous meetings between the two teams prior to kickoff. | `count(meetings)` | results.csv (shifted) | Head-to-Head | int64 | Yes | Medium |
| `h2h_draws` | Total number of draws between the two teams in prior meetings. | `count(draws)` | results.csv (shifted) | Head-to-Head | int64 | Yes | Medium |
| `h2h_home_wins` | Total number of wins by the current home team against the current away team in prior meetings. | `count(home_wins)` | results.csv (shifted) | Head-to-Head | int64 | Yes | High |
| `h2h_away_wins` | Total number of wins by the current away team against the current home team in prior meetings. | `count(away_wins)` | results.csv (shifted) | Head-to-Head | int64 | Yes | High |
| `h2h_gd` | Goal difference from the perspective of the current home team against the current away team in prior meetings. | `home_goals - away_goals` | results.csv (shifted) | Head-to-Head | float64 | Yes | High |
| `home_attack_rating` | Home team's relative attacking rating (based on rolling 10-match goals divided by global average). | `home_avg_goals_scored_10 / 1.4` | Derived | Attack | float64 | Yes | High |
| `away_attack_rating` | Away team's relative attacking rating. | `away_avg_goals_scored_10 / 1.4` | Derived | Attack | float64 | Yes | High |
| `attack_ratio` | Engineered feature attack_ratio. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_defence_rating` | Home team's relative defensive rating (global average divided by rolling 10-match goals conceded). | `1.4 / home_avg_goals_conceded_10` | Derived | Defence | float64 | Yes | High |
| `away_defence_rating` | Away team's relative defensive rating. | `1.4 / away_avg_goals_conceded_10` | Derived | Defence | float64 | Yes | High |
| `home_clean_sheet_rate` | Engineered feature home_clean_sheet_rate. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_clean_sheet_rate` | Engineered feature away_clean_sheet_rate. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_world_cup_titles_before` | Number of World Cup titles won by the home team before the match year. | `count(titles)` | train_dataset.csv / test_dataset.csv | Tournament Experience | float64 | Yes | Medium |
| `away_world_cup_titles_before` | Number of World Cup titles won by the away team before the match year. | `count(titles)` | train_dataset.csv / test_dataset.csv | Tournament Experience | float64 | Yes | Medium |
| `home_world_cup_participations_before` | Engineered feature home_world_cup_participations_before. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_world_cup_participations_before` | Engineered feature away_world_cup_participations_before. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_groups_passed_before` | Engineered feature home_groups_passed_before. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_groups_passed_before` | Engineered feature away_groups_passed_before. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_round16_before` | Engineered feature home_round16_before. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_round16_before` | Engineered feature away_round16_before. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_quarterfinals_before` | Engineered feature home_quarterfinals_before. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_quarterfinals_before` | Engineered feature away_quarterfinals_before. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_semifinals_before` | Engineered feature home_semifinals_before. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_semifinals_before` | Engineered feature away_semifinals_before. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_finals_before` | Engineered feature home_finals_before. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `away_finals_before` | Engineered feature away_finals_before. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `is_neutral` | Flag indicating if the match was played at a neutral venue. | `neutral == True` | results.csv | Context | int64 | Yes | High |
| `is_world_cup` | Flag indicating if the match is a FIFA World Cup match. | `tournament contains 'World Cup'` | results.csv | Context | int64 | Yes | Medium |
| `is_friendly` | Engineered feature is_friendly. | `N/A` | Derived | Other | float64 | Yes | Medium |
| `home_rest_days` | Days since the home team's last match (clipped to a max of 30). | `match_date - previous_match_date` | results.csv | Context | float64 | Yes | Medium |
| `away_rest_days` | Days since the away team's last match (clipped to a max of 30). | `match_date - previous_match_date` | results.csv | Context | float64 | Yes | Medium |
| `rest_difference` | Rest day difference between home and away teams. | `home_rest_days - away_rest_days` | Derived | Context | float64 | Yes | Medium |

