# 📊 WorldCupAI — Dataset Catalogue

> Complete catalogue of every dataset in the project workspace with purpose, strengths, weaknesses, dependencies, expected contribution, and classification.

---

## Classification Legend

| Classification | Meaning |
|---------------|---------|
| **Canonical Source** | Primary authoritative dataset for its domain — used directly in pipelines |
| **Supporting Source** | Enrichment data — merged into canonical sources during feature engineering |
| **Validation Source** | Used to cross-check or validate other datasets; not a primary training input |
| **Ignore** | Duplicate, raw precursor, or irrelevant — excluded from pipelines |

---

## Source 1 — FIFA World Cup Dataset

### `train.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Dataset/` |
| **Rows × Cols** | 192 × 24 |
| **Purpose** | Pre-engineered team-level features per World Cup edition (2006–2022). Each row = one team in one tournament. Includes rolling stats, FIFA rankings, market values, and historical performance. Target columns: `winner`, `finalist`, `semi_finalist`, `quarter_finalist`. |
| **Strengths** | Ready-to-use ML features; clean structure; combines multiple data dimensions into one row per team-tournament |
| **Weaknesses** | Only 192 rows (limited for DL); aggregates may hide match-level signal; no opponent-aware features |
| **Dependencies** | None (self-contained) |
| **Expected Contribution** | ⭐⭐⭐⭐⭐ **Critical** — Primary training dataset for team-level tournament outcome prediction |
| **Classification** | **Canonical Source** |

### `test.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Dataset/` |
| **Rows × Cols** | 48 × 24 |
| **Purpose** | Same schema as train.csv but for the 2026 World Cup. Target columns are empty (NaN) — these are the teams we predict for. |
| **Strengths** | Identical schema to training data; pre-populated with 2026 features |
| **Weaknesses** | Target columns intentionally blank; 48 teams (expanded format for 2026) |
| **Dependencies** | Schema depends on `train.csv` |
| **Expected Contribution** | ⭐⭐⭐⭐⭐ **Critical** — The prediction target file |
| **Classification** | **Canonical Source** |

---

## Source 2 — Fjelstul World Cup Historical Database

> 27 interconnected tables covering every World Cup from 1930–2022. Originally structured as an R package (`worldcup`).

### `tournaments.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 30 × 18 |
| **Purpose** | Master list of all 30 World Cup tournaments (22 men's + 8 women's). Includes host, winner, dates, count of teams, groups, matches. |
| **Strengths** | Comprehensive tournament metadata; contains winner field for validation |
| **Weaknesses** | `winner` column is a leakage risk if used as feature; small N |
| **Dependencies** | `tournament_id` links to all other tables |
| **Expected Contribution** | ⭐⭐⭐⭐ High — Tournament-level context and validation |
| **Classification** | **Canonical Source** |

### `matches.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 1,248 × 37 |
| **Purpose** | Every World Cup match ever played. Includes teams, scores, stage, venue, result, extra time, penalties. |
| **Strengths** | Match-level granularity; score breakdowns; stage context; links to teams, stadiums, tournaments |
| **Weaknesses** | Score columns are leakage if used for same-match prediction; limited to WC only |
| **Dependencies** | `tournament_id`, `match_id`, `home_team_id`, `away_team_id`, `stadium_id` |
| **Expected Contribution** | ⭐⭐⭐⭐⭐ **Critical** — Core training data for match-level models |
| **Classification** | **Canonical Source** |

### `team_appearances.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 2,496 × 36 |
| **Purpose** | One row per team per match. Includes goals scored/conceded, result, extra time, penalty info. |
| **Strengths** | Team-perspective match data; rich result breakdowns; links to tournament stages |
| **Weaknesses** | `goals_for`, `goals_against`, `result`, `win`, `draw` are leakage for same-match prediction |
| **Dependencies** | `match_id`, `team_id`, `tournament_id` |
| **Expected Contribution** | ⭐⭐⭐⭐⭐ **Critical** — Foundation for rolling team-performance features |
| **Classification** | **Canonical Source** |

### `group_standings.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 626 × 19 |
| **Purpose** | Final group stage standings for every team in every tournament. W/D/L, GF/GA, GD, points, position. |
| **Strengths** | Group-level aggregated performance; useful for "past group performance" features |
| **Weaknesses** | All columns are post-match outcomes; high leakage risk if not temporally filtered |
| **Dependencies** | `tournament_id`, `group_id`, `team_id` |
| **Expected Contribution** | ⭐⭐⭐⭐ High — Historical group-stage performance features (prior tournaments only) |
| **Classification** | **Canonical Source** |

### `player_appearances.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 27,432 × 21 |
| **Purpose** | Every player appearance in every World Cup match. Includes goals, assists, position, starter/sub. |
| **Strengths** | Player-level granularity; links players to matches and teams |
| **Weaknesses** | Individual-level — needs aggregation to team level for match prediction |
| **Dependencies** | `match_id`, `player_id`, `team_id` |
| **Expected Contribution** | ⭐⭐⭐⭐ High — Aggregated player experience features per team |
| **Classification** | **Canonical Source** |

### `squads.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 13,843 × 12 |
| **Purpose** | Full squad lists for every team in every tournament. Includes player position, shirt number. |
| **Strengths** | Links players to teams/tournaments; positional data |
| **Weaknesses** | No performance stats; just roster data |
| **Dependencies** | `tournament_id`, `team_id`, `player_id` |
| **Expected Contribution** | ⭐⭐⭐ Medium — Squad composition features (avg age, experience) |
| **Classification** | **Supporting Source** |

### `players.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 10,401 × 13 |
| **Purpose** | Master player list. Name, birth date, count of tournaments, position flags. |
| **Strengths** | Clean player master; birth dates enable age calculations |
| **Weaknesses** | No current team info (player-team mapping via squads/appearances) |
| **Dependencies** | `player_id` |
| **Expected Contribution** | ⭐⭐⭐ Medium — Player demographic features |
| **Classification** | **Supporting Source** |

### `goals.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 3,637 × 27 |
| **Purpose** | Every goal scored in World Cup history. Includes minute, scorer, penalty/own-goal flags. |
| **Strengths** | Minute-level goal data; penalty/own-goal distinctions |
| **Weaknesses** | Leakage: goal data from the match being predicted |
| **Dependencies** | `match_id`, `player_id`, `team_id` |
| **Expected Contribution** | ⭐⭐⭐ Medium — Historical scoring pattern features (prior matches only) |
| **Classification** | **Supporting Source** |

### `bookings.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 3,178 × 26 |
| **Purpose** | All yellow/red cards in World Cup history. |
| **Strengths** | Discipline patterns; minute-level data |
| **Weaknesses** | Post-match data; niche signal for prediction |
| **Dependencies** | `match_id`, `player_id` |
| **Expected Contribution** | ⭐⭐ Low — Discipline history features (marginal predictive value) |
| **Classification** | **Supporting Source** |

### `penalty_kicks.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 396 × 19 |
| **Purpose** | All penalty shootout kicks. Includes conversion success. |
| **Strengths** | Penalty shootout history per team; conversion rates |
| **Weaknesses** | Very small N; highly random; leakage for same-match |
| **Dependencies** | `match_id`, `player_id` |
| **Expected Contribution** | ⭐⭐ Low — Historical penalty conversion rates (marginal) |
| **Classification** | **Supporting Source** |

### `substitutions.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 10,222 × 24 |
| **Purpose** | All substitutions made in World Cup matches. |
| **Strengths** | Tactical change patterns |
| **Weaknesses** | Post-match data; niche signal |
| **Dependencies** | `match_id`, `player_id` |
| **Expected Contribution** | ⭐ Very Low — Limited predictive value for match outcome |
| **Classification** | **Ignore** (for prediction pipeline) |

### `qualified_teams.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 625 × 8 |
| **Purpose** | Which teams qualified for each tournament and their final performance tier. |
| **Strengths** | Historical qualification and performance data |
| **Weaknesses** | `performance` column is post-tournament leakage |
| **Dependencies** | `tournament_id`, `team_id` |
| **Expected Contribution** | ⭐⭐⭐ Medium — Qualification frequency features (prior tournaments) |
| **Classification** | **Supporting Source** |

### `teams.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 88 × 14 |
| **Purpose** | Master team list. ISO codes, federation, confederation, Wikipedia links. |
| **Strengths** | Definitive team ID mapping; ISO codes for joining |
| **Weaknesses** | Static metadata only |
| **Dependencies** | `team_id`, `team_code` |
| **Expected Contribution** | ⭐⭐⭐⭐ High — Critical entity mapping / joining table |
| **Classification** | **Canonical Source** |

### `confederations.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 6 × 5 |
| **Purpose** | FIFA confederation lookup (AFC, CAF, CONCACAF, CONMEBOL, OFC, UEFA). |
| **Strengths** | Clean reference table |
| **Weaknesses** | Only 6 rows; minimal feature value |
| **Dependencies** | `confederation_id` |
| **Expected Contribution** | ⭐⭐ Low — Joining/reference only |
| **Classification** | **Supporting Source** |

### `stadiums.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 240 × 8 |
| **Purpose** | All World Cup stadiums. City, country, capacity. |
| **Strengths** | Venue information for venue-aware features |
| **Weaknesses** | Limited predictive value for match outcomes |
| **Dependencies** | `stadium_id` |
| **Expected Contribution** | ⭐ Very Low — Venue features (marginal) |
| **Classification** | **Ignore** (for prediction pipeline) |

### `host_countries.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 31 × 7 |
| **Purpose** | Host countries per tournament with their performance level. |
| **Strengths** | Host advantage analysis |
| **Weaknesses** | `performance` is post-tournament leakage; tiny N |
| **Dependencies** | `tournament_id` |
| **Expected Contribution** | ⭐⭐ Low — `is_host` feature derivation (already in train/test.csv) |
| **Classification** | **Validation Source** |

### `tournament_standings.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 120 × 7 |
| **Purpose** | Final standings (1st–4th) for each tournament. |
| **Strengths** | Clear outcome reference |
| **Weaknesses** | `position` is pure leakage; only top-4 teams |
| **Dependencies** | `tournament_id`, `team_id` |
| **Expected Contribution** | ⭐⭐⭐ Medium — Historical success features (prior tournaments only) |
| **Classification** | **Validation Source** |

### `tournament_stages.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 155 × 16 |
| **Purpose** | Tournament format metadata — stages, dates, match counts per stage. |
| **Strengths** | Structural context for tournament formats |
| **Weaknesses** | Metadata only; limited predictive value |
| **Dependencies** | `tournament_id` |
| **Expected Contribution** | ⭐ Very Low |
| **Classification** | **Ignore** |

### `groups.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `FIFA World Cup Historical Dataset/data-csv/` |
| **Rows × Cols** | 159 × 7 |
| **Purpose** | Group assignments per tournament. |
| **Strengths** | Group draw context |
| **Weaknesses** | Structural only |
| **Dependencies** | `tournament_id`, `group_id` |
| **Expected Contribution** | ⭐⭐ Low |
| **Classification** | **Supporting Source** |

### `managers.csv`, `manager_appearances.csv`, `manager_appointments.csv`
| Field | Detail |
|-------|--------|
| **Rows** | 475 / 2,538 / 637 |
| **Purpose** | Manager information and their match/tournament mappings. |
| **Expected Contribution** | ⭐⭐ Low — Manager tenure/experience features (marginal) |
| **Classification** | **Supporting Source** |

### `referees.csv`, `referee_appearances.csv`, `referee_appointments.csv`
| Field | Detail |
|-------|--------|
| **Rows** | 493 / 1,248 / 668 |
| **Purpose** | Referee information and match assignments. |
| **Expected Contribution** | ⭐ Very Low — Not predictive for match outcomes |
| **Classification** | **Ignore** |

### `award_winners.csv`, `awards.csv`
| Field | Detail |
|-------|--------|
| **Rows** | 200 / 8 |
| **Purpose** | Individual player awards (Golden Ball, Golden Boot, etc.). |
| **Expected Contribution** | ⭐ Very Low — Post-tournament awards; pure leakage |
| **Classification** | **Ignore** |

---

## Source 3 — International Football Results (1872–2026)

### `results.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `International football results from 1872 to 2026/` |
| **Rows × Cols** | 49,477 × 9 |
| **Purpose** | Every international football match since 1872. Home/away teams, scores, tournament, city, country, neutral venue flag. |
| **Strengths** | Massive coverage (150+ years); all tournaments (friendlies, qualifiers, continental cups, WC); neutral venue flag |
| **Weaknesses** | Scores are leakage for same-match prediction; team names need harmonization |
| **Dependencies** | None (self-contained) |
| **Expected Contribution** | ⭐⭐⭐⭐⭐ **Critical** — Foundation for head-to-head features, overall win rates, form calculations |
| **Classification** | **Canonical Source** |

### `goalscorers.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `International football results from 1872 to 2026/` |
| **Rows × Cols** | 47,747 × 8 |
| **Purpose** | Individual goal events across all international matches. Scorer, minute, own-goal, penalty flags. |
| **Strengths** | Goal-level granularity; penalty/own-goal distinction |
| **Weaknesses** | Needs aggregation; same-match goals are leakage |
| **Dependencies** | Links to `results.csv` via `date + home_team + away_team` |
| **Expected Contribution** | ⭐⭐⭐ Medium — Scorer availability/form features |
| **Classification** | **Supporting Source** |

### `shootouts.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `International football results from 1872 to 2026/` |
| **Rows × Cols** | 678 × 5 |
| **Purpose** | Penalty shootout outcomes in international matches. Winner and first-shooter. |
| **Strengths** | Historical penalty shootout win rates |
| **Weaknesses** | Small N; `winner` is post-match; highly stochastic |
| **Dependencies** | Links to `results.csv` via `date + home_team + away_team` |
| **Expected Contribution** | ⭐⭐ Low — Penalty shootout history (very marginal) |
| **Classification** | **Supporting Source** |

### `former_names.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `International football results from 1872 to 2026/` |
| **Rows × Cols** | 36 × 4 |
| **Purpose** | Maps former country names to current names (e.g., Dahomey → Benin). |
| **Strengths** | Critical for entity resolution; enables correct historical joins |
| **Weaknesses** | Only 36 entries — may be incomplete |
| **Dependencies** | None |
| **Expected Contribution** | ⭐⭐⭐⭐ High — Essential for entity harmonization |
| **Classification** | **Canonical Source** (mapping table) |

---

## Source 4 — Football Data from Transfermarkt

### `players.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `Football Data from Transfermarkt/` |
| **Rows × Cols** | 48,351 × 26 |
| **Purpose** | Player profiles: name, DOB, nationality, position, market value, club, height, foot. |
| **Strengths** | Rich player attributes; market valuations; position breakdowns |
| **Weaknesses** | 186K+ missing values; club-level (needs national team mapping) |
| **Dependencies** | `player_id`, `current_club_id`, `country_of_citizenship` |
| **Expected Contribution** | ⭐⭐⭐⭐ High — Squad quality features (market value aggregates, age profiles) |
| **Classification** | **Supporting Source** |

### `player_valuations.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `Football Data from Transfermarkt/` |
| **Rows × Cols** | 662,998 × 6 |
| **Purpose** | Time-series of player market valuations. Date, player_id, value in EUR, club. |
| **Strengths** | Temporal valuation data; enables "squad value at tournament time" calculations |
| **Weaknesses** | 98K missing values; needs player→national team mapping |
| **Dependencies** | `player_id` |
| **Expected Contribution** | ⭐⭐⭐⭐ High — Temporal squad market value features |
| **Classification** | **Supporting Source** |

### `appearances.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `Football Data from Transfermarkt/` |
| **Rows × Cols** | 1,888,125 × 13 |
| **Purpose** | Every player appearance in tracked competitions. Goals, assists, minutes, yellow/red cards. |
| **Strengths** | Massive volume; per-appearance stats; goals/assists |
| **Weaknesses** | 141 MB — requires chunked processing; club-level (not national team) |
| **Dependencies** | `player_id`, `game_id`, `competition_id` |
| **Expected Contribution** | ⭐⭐⭐ Medium — Club-level form proxy for national team players |
| **Classification** | **Supporting Source** |

### `games.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `Football Data from Transfermarkt/` |
| **Rows × Cols** | 88,872 × 23 |
| **Purpose** | Match-level data for tracked competitions. Teams, scores, attendance, referee. |
| **Strengths** | Broad match coverage; attendance data |
| **Weaknesses** | 81K missing values; primarily club matches |
| **Dependencies** | `game_id`, `competition_id`, `home_club_id`, `away_club_id` |
| **Expected Contribution** | ⭐⭐ Low — Club competition context |
| **Classification** | **Supporting Source** |

### `clubs.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `Football Data from Transfermarkt/` |
| **Rows × Cols** | 796 × 17 |
| **Purpose** | Club profiles: name, league, market value, squad size, coach. |
| **Strengths** | Club metadata for player-club-nation mappings |
| **Weaknesses** | Club-level only |
| **Dependencies** | `club_id` |
| **Expected Contribution** | ⭐⭐ Low — Reference table for club context |
| **Classification** | **Supporting Source** |

### `competitions.csv`
| Field | Detail |
|-------|--------|
| **Folder** | `Football Data from Transfermarkt/` |
| **Rows × Cols** | 65 × 11 |
| **Purpose** | Competition metadata (leagues, cups, international comps). |
| **Expected Contribution** | ⭐ Very Low — Reference only |
| **Classification** | **Ignore** |

### `countries.csv` / `national_teams.csv`
| Field | Detail |
|-------|--------|
| **Rows** | 124 / 124 |
| **Purpose** | Country and national team reference. Country codes, confederation, market value. |
| **Expected Contribution** | ⭐⭐⭐ Medium — Country code mapping; national team market values |
| **Classification** | **Supporting Source** |

### `game_events.csv` / `game_lineups.csv` / `club_games.csv` / `transfers.csv`
| Field | Detail |
|-------|--------|
| **Rows** | 1.27M / 3.18M / 177K / 178K |
| **Purpose** | Granular club-level data: match events, lineups, club results, player transfers. |
| **Expected Contribution** | ⭐ Very Low — Too granular and club-focused for WC prediction |
| **Classification** | **Ignore** (for prediction pipeline; may revisit for player-level DL features) |

---

## Source 5 — Root-Level Engineered Datasets

### `elo_ratings_wc2026.csv`
| Field | Detail |
|-------|--------|
| **Folder** | Root |
| **Rows × Cols** | 4,683 × 23 |
| **Purpose** | Historical Elo ratings for national teams. Includes rating snapshots, rank, W/D/L/GF/GA, confederation, host flag. |
| **Strengths** | Elo is a proven predictor; temporal snapshots; rich auxiliary stats |
| **Weaknesses** | Rating methodology assumptions; may not match FIFA ranking periods exactly |
| **Dependencies** | `country_code` for joining |
| **Expected Contribution** | ⭐⭐⭐⭐⭐ **Critical** — Elo rating is one of the strongest single predictors |
| **Classification** | **Canonical Source** |

### `fifa_mens_rank.csv`
| Field | Detail |
|-------|--------|
| **Folder** | Root |
| **Rows × Cols** | 13,130 × 8 |
| **Purpose** | Official FIFA rankings. Rank, team, points, previous points, point differential. |
| **Strengths** | Official ranking system; point differentials |
| **Weaknesses** | FIFA ranking methodology has known limitations |
| **Dependencies** | `team` (name-based) |
| **Expected Contribution** | ⭐⭐⭐⭐⭐ **Critical** — FIFA ranking features |
| **Classification** | **Canonical Source** |

### `player_aggregates.csv`
| Field | Detail |
|-------|--------|
| **Folder** | Root |
| **Rows × Cols** | 1,599 × 13 |
| **Purpose** | Country-level aggregated FIFA video game ratings (EA FC). Average overall, pace, shooting, passing, dribbling, defending, physic per country per FIFA version. |
| **Strengths** | Squad quality proxy; multi-dimensional player attributes |
| **Weaknesses** | Video game ratings are subjective; 96 missing values |
| **Dependencies** | `country` for joining |
| **Expected Contribution** | ⭐⭐⭐⭐ High — Squad quality proxy features |
| **Classification** | **Canonical Source** |

### `teams_form.csv`
| Field | Detail |
|-------|--------|
| **Folder** | Root |
| **Rows × Cols** | 102,094 × 5 |
| **Purpose** | Rolling team form: avg goals scored, avg goals conceded, win rate per match date. |
| **Strengths** | Pre-computed rolling form metrics; large coverage |
| **Weaknesses** | Need to verify rolling window definition; potential leakage if window includes target match |
| **Dependencies** | `team` + `match_date` for joining |
| **Expected Contribution** | ⭐⭐⭐⭐⭐ **Critical** — Direct form features |
| **Classification** | **Canonical Source** |

### `teams_match_features.csv`
| Field | Detail |
|-------|--------|
| **Folder** | Root |
| **Rows × Cols** | 43,364 × 35 |
| **Purpose** | Pre-merged match-level features: Elo ratings, player attributes, form metrics, tournament flags, and goals (home/away). |
| **Strengths** | Most complete match-level feature set; ready for modelling; includes both home/away perspectives |
| **Weaknesses** | `home_goals` / `away_goals` are the targets → must be used as labels, not features; 4,494 missing values |
| **Dependencies** | Derived from other sources |
| **Expected Contribution** | ⭐⭐⭐⭐⭐ **Critical** — Primary match-level training dataset |
| **Classification** | **Canonical Source** |

---

## Source 6 — Duplicate Folder: `worldcup-master/`

> **Status: DUPLICATE — Recommend IGNORE**

The entire `worldcup-master/` directory is a byte-for-byte duplicate of `FIFA World Cup Historical Dataset/`. All 43 CSV files match by MD5 hash. This is the same R package repository cloned or copied twice.

**Recommendation:** Use `FIFA World Cup Historical Dataset/data-csv/` as canonical. Archive or remove `worldcup-master/`.

---

## Summary: Canonical Sources for Pipeline

| Dataset | Domain | Critical? |
|---------|--------|-----------|
| `train.csv` / `test.csv` | Team-tournament features | ✅ Yes |
| `matches.csv` (Historical) | WC match results | ✅ Yes |
| `team_appearances.csv` | Team match outcomes | ✅ Yes |
| `results.csv` (International) | All intl. match results | ✅ Yes |
| `elo_ratings_wc2026.csv` | Elo strength ratings | ✅ Yes |
| `fifa_mens_rank.csv` | FIFA rankings | ✅ Yes |
| `teams_match_features.csv` | Pre-merged match features | ✅ Yes |
| `teams_form.csv` | Rolling team form | ✅ Yes |
| `player_aggregates.csv` | Squad quality proxy | ✅ Yes |
| `teams.csv` (Historical) | Entity mapping | ✅ Yes |
| `former_names.csv` | Entity harmonization | ✅ Yes |
