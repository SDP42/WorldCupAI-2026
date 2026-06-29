import os
import pandas as pd
from typing import Dict, Any, List
from src.utils.logger import setup_logger

logger = setup_logger("feature_dictionary")

class FeatureDictionary:
    # Metadata for all engineered features
    METADATA = {
        # Team Strength
        "elo_diff": {
            "description": "Difference in Elo ratings between home and away teams.",
            "formula": "home_elo - away_elo",
            "source": "elo_ratings_wc2026.csv",
            "family": "Team Strength",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "Very High"
        },
        "elo_ratio": {
            "description": "Ratio of home team Elo rating to away team Elo rating.",
            "formula": "home_elo / away_elo",
            "source": "elo_ratings_wc2026.csv",
            "family": "Team Strength",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "High"
        },
        "rank_diff": {
            "description": "Difference in official FIFA rank between home and away teams.",
            "formula": "home_rank - away_rank",
            "source": "fifa_mens_rank.csv",
            "family": "Team Strength",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "Very High"
        },
        "rank_ratio": {
            "description": "Ratio of home team FIFA rank to away team FIFA rank.",
            "formula": "home_rank / away_rank",
            "source": "fifa_mens_rank.csv",
            "family": "Team Strength",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "High"
        },
        "overall_diff": {
            "description": "Difference in average EA FC overall rating between home and away squads.",
            "formula": "home_avg_overall - away_avg_overall",
            "source": "player_aggregates.csv",
            "family": "Team Strength",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "High"
        },
        "overall_ratio": {
            "description": "Ratio of home team average EA FC overall rating to away team.",
            "formula": "home_avg_overall / away_avg_overall",
            "source": "player_aggregates.csv",
            "family": "Team Strength",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "Medium"
        },
        "attack_diff": {
            "description": "Difference in average EA FC attack rating between home and away squads.",
            "formula": "home_avg_attack - away_avg_attack",
            "source": "player_aggregates.csv",
            "family": "Team Strength",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "Medium"
        },
        "defense_diff": {
            "description": "Difference in average EA FC defense rating between home and away squads.",
            "formula": "home_avg_defense - away_avg_defense",
            "source": "player_aggregates.csv",
            "family": "Team Strength",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "Medium"
        },
        
        # Recent Form
        "home_form_win_rate_5": {
            "description": "Home team win rate over the last 5 matches prior to kickoff.",
            "formula": "wins_5 / 5",
            "source": "results.csv (shifted)",
            "family": "Recent Form",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "High"
        },
        "away_form_win_rate_5": {
            "description": "Away team win rate over the last 5 matches prior to kickoff.",
            "formula": "wins_5 / 5",
            "source": "results.csv (shifted)",
            "family": "Recent Form",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "High"
        },
        "home_form_avg_goals_scored_5": {
            "description": "Home team average goals scored over the last 5 matches prior to kickoff.",
            "formula": "goals_scored_5 / 5",
            "source": "results.csv (shifted)",
            "family": "Recent Form",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "High"
        },
        "away_form_avg_goals_scored_5": {
            "description": "Away team average goals scored over the last 5 matches prior to kickoff.",
            "formula": "goals_scored_5 / 5",
            "source": "results.csv (shifted)",
            "family": "Recent Form",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "High"
        },
        "home_form_avg_goals_conceded_5": {
            "description": "Home team average goals conceded over the last 5 matches prior to kickoff.",
            "formula": "goals_conceded_5 / 5",
            "source": "results.csv (shifted)",
            "family": "Recent Form",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "High"
        },
        "away_form_avg_goals_conceded_5": {
            "description": "Away team average goals conceded over the last 5 matches prior to kickoff.",
            "formula": "goals_conceded_5 / 5",
            "source": "results.csv (shifted)",
            "family": "Recent Form",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "High"
        },
        "home_form_clean_sheet_rate_5": {
            "description": "Home team clean sheet rate over the last 5 matches prior to kickoff.",
            "formula": "clean_sheets_5 / 5",
            "source": "results.csv (shifted)",
            "family": "Recent Form",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "Medium"
        },
        "away_form_clean_sheet_rate_5": {
            "description": "Away team clean sheet rate over the last 5 matches prior to kickoff.",
            "formula": "clean_sheets_5 / 5",
            "source": "results.csv (shifted)",
            "family": "Recent Form",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "Medium"
        },
        
        # Head-to-Head
        "h2h_meetings": {
            "description": "Total number of previous meetings between the two teams prior to kickoff.",
            "formula": "count(meetings)",
            "source": "results.csv (shifted)",
            "family": "Head-to-Head",
            "type": "int64",
            "temporal_safe": "Yes",
            "expected_value": "Medium"
        },
        "h2h_home_wins": {
            "description": "Total number of wins by the current Team 1 (first team in bracket/positional home team) against Team 2 in prior meetings (Neutral Venue context).",
            "formula": "count(home_wins)",
            "source": "results.csv (shifted)",
            "family": "Head-to-Head",
            "type": "int64",
            "temporal_safe": "Yes",
            "expected_value": "High"
        },
        "h2h_away_wins": {
            "description": "Total number of wins by the current Team 2 (second team in bracket/positional away team) against Team 1 in prior meetings (Neutral Venue context).",
            "formula": "count(away_wins)",
            "source": "results.csv (shifted)",
            "family": "Head-to-Head",
            "type": "int64",
            "temporal_safe": "Yes",
            "expected_value": "High"
        },
        "h2h_draws": {
            "description": "Total number of draws between the two teams in prior meetings.",
            "formula": "count(draws)",
            "source": "results.csv (shifted)",
            "family": "Head-to-Head",
            "type": "int64",
            "temporal_safe": "Yes",
            "expected_value": "Medium"
        },
        "h2h_gd": {
            "description": "Goal difference from the perspective of the current home team against the current away team in prior meetings.",
            "formula": "home_goals - away_goals",
            "source": "results.csv (shifted)",
            "family": "Head-to-Head",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "High"
        },
        
        # Attack & Defence Ratings
        "home_attack_rating": {
            "description": "Home team's relative attacking rating (based on rolling 10-match goals divided by global average).",
            "formula": "home_avg_goals_scored_10 / 1.4",
            "source": "Derived",
            "family": "Attack",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "High"
        },
        "away_attack_rating": {
            "description": "Away team's relative attacking rating.",
            "formula": "away_avg_goals_scored_10 / 1.4",
            "source": "Derived",
            "family": "Attack",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "High"
        },
        "home_defence_rating": {
            "description": "Home team's relative defensive rating (global average divided by rolling 10-match goals conceded).",
            "formula": "1.4 / home_avg_goals_conceded_10",
            "source": "Derived",
            "family": "Defence",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "High"
        },
        "away_defence_rating": {
            "description": "Away team's relative defensive rating.",
            "formula": "1.4 / away_avg_goals_conceded_10",
            "source": "Derived",
            "family": "Defence",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "High"
        },
        
        # Tournament Experience
        "home_world_cup_titles_before": {
            "description": "Number of World Cup titles won by the home team before the match year.",
            "formula": "count(titles)",
            "source": "train_dataset.csv / test_dataset.csv",
            "family": "Tournament Experience",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "Medium"
        },
        "away_world_cup_titles_before": {
            "description": "Number of World Cup titles won by the away team before the match year.",
            "formula": "count(titles)",
            "source": "train_dataset.csv / test_dataset.csv",
            "family": "Tournament Experience",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "Medium"
        },
        
        # Context
        "is_neutral": {
            "description": "Flag indicating if the match was played at a neutral venue.",
            "formula": "neutral == True",
            "source": "results.csv",
            "family": "Context",
            "type": "int64",
            "temporal_safe": "Yes",
            "expected_value": "High"
        },
        "is_world_cup": {
            "description": "Flag indicating if the match is a FIFA World Cup match.",
            "formula": "tournament contains 'World Cup'",
            "source": "results.csv",
            "family": "Context",
            "type": "int64",
            "temporal_safe": "Yes",
            "expected_value": "Medium"
        },
        "home_rest_days": {
            "description": "Days since the home team's last match (clipped to a max of 30).",
            "formula": "match_date - previous_match_date",
            "source": "results.csv",
            "family": "Context",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "Medium"
        },
        "away_rest_days": {
            "description": "Days since the away team's last match (clipped to a max of 30).",
            "formula": "match_date - previous_match_date",
            "source": "results.csv",
            "family": "Context",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "Medium"
        },
        "rest_difference": {
            "description": "Rest day difference between home and away teams.",
            "formula": "home_rest_days - away_rest_days",
            "source": "Derived",
            "family": "Context",
            "type": "float64",
            "temporal_safe": "Yes",
            "expected_value": "Medium"
        }
    }

    def __init__(self):
        pass

    def generate_dictionary(self, feature_cols: List[str], output_md: str, output_csv: str):
        """Generates the FEATURE_DICTIONARY.md and feature_dictionary.csv files."""
        records = []
        
        table = "| Feature Name | Description | Formula / Logic | Source | Family | Type | Temporal Safe? | Expected Value |\n"
        table += "|---|---|---|---|---|---|---|---|\n"
        
        for col in feature_cols:
            # Fallback if metadata is not explicitly defined for a specific rolling variant
            base_col = col
            # Remove home_/away_ and _5/_10 suffixes to find base metadata
            for suffix in ["_5", "_10", "_5_home", "_5_away", "_10_home", "_10_away"]:
                if col.endswith(suffix):
                    base_col = col[:-len(suffix)]
            if col.startswith("home_form_"):
                base_col = "home_form_" + col[10:]
                # Remove _5/_10
                for suffix in ["_5", "_10"]:
                    if base_col.endswith(suffix):
                        base_col = base_col[:-len(suffix)]
            if col.startswith("away_form_"):
                base_col = "away_form_" + col[10:]
                for suffix in ["_5", "_10"]:
                    if base_col.endswith(suffix):
                        base_col = base_col[:-len(suffix)]
            if col.startswith("home_"):
                base_col_alt = col[5:]
                if base_col_alt in self.METADATA:
                    base_col = base_col_alt
            if col.startswith("away_"):
                base_col_alt = col[5:]
                if base_col_alt in self.METADATA:
                    base_col = base_col_alt
                    
            meta = self.METADATA.get(col, self.METADATA.get(base_col, {
                "description": f"Engineered feature {col}.",
                "formula": "N/A",
                "source": "Derived",
                "family": "Other",
                "type": "float64",
                "temporal_safe": "Yes",
                "expected_value": "Medium"
            }))
            
            records.append({
                "feature_name": col,
                "description": meta["description"],
                "formula": meta["formula"],
                "source": meta["source"],
                "family": meta["family"],
                "type": meta["type"],
                "temporal_safe": meta["temporal_safe"],
                "expected_value": meta["expected_value"]
            })
            
            table += f"| `{col}` | {meta['description']} | `{meta['formula']}` | {meta['source']} | {meta['family']} | {meta['type']} | {meta['temporal_safe']} | {meta['expected_value']} |\n"
            
        # Write CSV
        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved feature dictionary CSV to {output_csv}")
        
        # Write Markdown
        report_content = f"""# 📖 WorldCupAI — Feature Dictionary

This document serves as the authoritative registry for all engineered features in the WorldCupAI platform.

## Feature Registry Table
{table}
"""
        with open(output_md, "w") as f:
            f.write(report_content)
        logger.info(f"Saved feature dictionary Markdown to {output_md}")
