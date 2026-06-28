import pandas as pd
import numpy as np
from src.features.base_features import BaseFeatureGenerator
from src.utils.logger import setup_logger

logger = setup_logger("context_features")

class ContextFeatures(BaseFeatureGenerator):
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes context features:
        - Neutral venue flag.
        - World Cup match flag.
        - Rest days (days since last match) for home and away teams.
        """
        df = df.copy()
        df = df.sort_values('date')
        features = pd.DataFrame(index=df.index)
        
        # 1. Basic flags
        if 'neutral' in df.columns:
            features['is_neutral'] = df['neutral'].astype(int)
        else:
            features['is_neutral'] = 0
            
        if 'tournament' in df.columns:
            features['is_world_cup'] = df['tournament'].str.contains('FIFA World Cup', case=False, na=False).astype(int)
            features['is_friendly'] = df['tournament'].str.contains('Friendly', case=False, na=False).astype(int)
        else:
            features['is_world_cup'] = 0
            features['is_friendly'] = 0
            
        # 2. Compute rest days (days since last match)
        # Create a long dataframe of team match dates
        home_dates = df[['date', 'home_team']].rename(columns={'home_team': 'team'})
        home_dates['was_home'] = True
        home_dates['match_idx'] = df.index
        
        away_dates = df[['date', 'away_team']].rename(columns={'away_team': 'team'})
        away_dates['was_home'] = False
        away_dates['match_idx'] = df.index
        
        team_dates = pd.concat([home_dates, away_dates], ignore_index=True)
        team_dates = team_dates.sort_values(['team', 'date']).reset_index(drop=True)
        
        # Calculate difference in days between consecutive matches for each team
        grouped = team_dates.groupby('team')
        team_dates['rest_days'] = grouped['date'].diff().dt.days
        
        # For the first match of a team, set a default rest day value (e.g. 30 days)
        team_dates['rest_days'] = team_dates['rest_days'].fillna(30.0)
        
        # Clip rest days to a maximum of 30 to avoid extreme outliers (e.g. years between matches)
        team_dates['rest_days'] = team_dates['rest_days'].clip(upper=30.0)
        
        # Split back to home and away
        home_rest = team_dates[team_dates['was_home'] == True].set_index('match_idx')
        away_rest = team_dates[team_dates['was_home'] == False].set_index('match_idx')
        
        # Align with original index
        features['home_rest_days'] = home_rest['rest_days'].reindex(df.index)
        features['away_rest_days'] = away_rest['rest_days'].reindex(df.index)
        
        # Fill any remaining NaNs with default
        features['home_rest_days'] = features['home_rest_days'].fillna(30.0)
        features['away_rest_days'] = features['away_rest_days'].fillna(30.0)
        
        features['rest_difference'] = features['home_rest_days'] - features['away_rest_days']
        
        return features
