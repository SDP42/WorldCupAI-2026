import pandas as pd
import numpy as np
from src.features.base_features import BaseFeatureGenerator
from src.utils.logger import setup_logger

logger = setup_logger("recent_form")

class RecentFormFeatures(BaseFeatureGenerator):
    def __init__(self, windows=[5, 10]):
        self.windows = windows

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates rolling form features for both home and away teams.
        Guarantees temporal safety by shifting the match history by 1 before rolling window computation.
        """
        df = df.copy()
        df = df.sort_values('date')
        
        # 1. Create a long dataframe of team match histories
        home_appearances = df[['date', 'home_team', 'away_team', 'home_score', 'away_score']].rename(columns={
            'home_team': 'team',
            'away_team': 'opponent',
            'home_score': 'goals_scored',
            'away_score': 'goals_conceded'
        })
        home_appearances['was_home'] = True
        
        away_appearances = df[['date', 'away_team', 'home_team', 'away_score', 'home_score']].rename(columns={
            'away_team': 'team',
            'home_team': 'opponent',
            'away_score': 'goals_scored',
            'home_score': 'goals_conceded'
        })
        away_appearances['was_home'] = False
        
        team_history = pd.concat([home_appearances, away_appearances], ignore_index=True)
        team_history = team_history.sort_values(['team', 'date']).reset_index(drop=True)
        
        # Define match outcomes
        team_history['is_win'] = (team_history['goals_scored'] > team_history['goals_conceded']).astype(int)
        team_history['is_loss'] = (team_history['goals_scored'] < team_history['goals_conceded']).astype(int)
        team_history['is_draw'] = (team_history['goals_scored'] == team_history['goals_conceded']).astype(int)
        team_history['is_clean_sheet'] = (team_history['goals_conceded'] == 0).astype(int)
        team_history['points'] = team_history['is_win'] * 3 + team_history['is_draw'] * 1
        
        # 2. Compute rolling features grouped by team, shifted by 1 to prevent leakage
        rolling_features_dict = {}
        
        # Group by team
        grouped = team_history.groupby('team')
        
        for w in self.windows:
            # Shift by 1 first so the current match on date D is NOT included
            shifted_goals_scored = grouped['goals_scored'].shift(1)
            shifted_goals_conceded = grouped['goals_conceded'].shift(1)
            shifted_wins = grouped['is_win'].shift(1)
            shifted_losses = grouped['is_loss'].shift(1)
            shifted_draws = grouped['is_draw'].shift(1)
            shifted_cs = grouped['is_clean_sheet'].shift(1)
            shifted_points = grouped['points'].shift(1)
            
            # Now compute rolling metrics
            rolling_features_dict[f'win_rate_{w}'] = grouped['is_win'].shift(1).rolling(w, min_periods=1).mean()
            rolling_features_dict[f'loss_rate_{w}'] = grouped['is_loss'].shift(1).rolling(w, min_periods=1).mean()
            rolling_features_dict[f'draw_rate_{w}'] = grouped['is_draw'].shift(1).rolling(w, min_periods=1).mean()
            rolling_features_dict[f'avg_goals_scored_{w}'] = grouped['goals_scored'].shift(1).rolling(w, min_periods=1).mean()
            rolling_features_dict[f'avg_goals_conceded_{w}'] = grouped['goals_conceded'].shift(1).rolling(w, min_periods=1).mean()
            rolling_features_dict[f'goal_diff_{w}'] = (grouped['goals_scored'].shift(1) - grouped['goals_conceded'].shift(1)).rolling(w, min_periods=1).mean()
            rolling_features_dict[f'clean_sheet_rate_{w}'] = grouped['is_clean_sheet'].shift(1).rolling(w, min_periods=1).mean()
            rolling_features_dict[f'avg_points_{w}'] = grouped['points'].shift(1).rolling(w, min_periods=1).mean()
            
        # Add rolling features back to team_history
        for col_name, series in rolling_features_dict.items():
            team_history[col_name] = series
            
        # 3. Join these features back to the original master_dataset
        # Split team_history back into home and away perspectives
        home_features = team_history[team_history['was_home'] == True].copy()
        away_features = team_history[team_history['was_home'] == False].copy()
        
        # Rename columns for joining
        home_rename = {col: f'home_form_{col}' for col in rolling_features_dict.keys()}
        home_features = home_features.rename(columns=home_rename)
        
        away_rename = {col: f'away_form_{col}' for col in rolling_features_dict.keys()}
        away_features = away_features.rename(columns=away_rename)
        
        # Merge back to original df
        merged = pd.merge(
            df,
            home_features[['date', 'team', 'opponent'] + list(home_rename.values())],
            left_on=['date', 'home_team', 'away_team'],
            right_on=['date', 'team', 'opponent'],
            how='left'
        ).drop(columns=['team', 'opponent'], errors='ignore')
        
        merged = pd.merge(
            merged,
            away_features[['date', 'team', 'opponent'] + list(away_rename.values())],
            left_on=['date', 'away_team', 'home_team'],
            right_on=['date', 'team', 'opponent'],
            how='left'
        ).drop(columns=['team', 'opponent'], errors='ignore')
        
        # Return only the newly engineered features (keeping the original index)
        new_cols = list(home_rename.values()) + list(away_rename.values())
        return merged[new_cols].reindex(df.index)
