import pandas as pd
import numpy as np
from typing import List, Optional
from src.utils.logger import setup_logger

logger = setup_logger("cleaner")

class DataCleaner:
    def __init__(self):
        pass

    def clean_text_columns(self, df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
        """Trims whitespace and standardizes casing for text columns."""
        df = df.copy()
        for col in text_cols:
            if col in df.columns:
                # Convert to string, strip, and handle NaNs safely
                df[col] = df[col].astype(str).str.strip()
                # Replace string 'nan' back to actual NaN
                df[col] = df[col].replace({'nan': np.nan, 'None': np.nan})
        return df

    def standardize_dates(self, df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
        """Converts date columns to datetime64[ns]."""
        df = df.copy()
        for col in date_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.info(f"Standardized date column '{col}' to datetime. NaT count: {df[col].isna().sum()}")
                except Exception as e:
                    logger.warning(f"Could not convert column '{col}' to datetime: {e}")
        return df

    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Removes exact duplicate rows."""
        before = len(df)
        df = df.drop_duplicates(subset=subset)
        after = len(df)
        if before - after > 0:
            logger.info(f"Removed {before - after:,} duplicate rows.")
        return df

    def clean_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Specific cleaning for results.csv."""
        df = df.copy()
        df = self.remove_duplicates(df)
        df = self.clean_text_columns(df, ['home_team', 'away_team', 'tournament', 'city', 'country'])
        df = self.standardize_dates(df, ['date'])
        
        # Drop rows with missing crucial fields
        before = len(df)
        df = df.dropna(subset=['date', 'home_team', 'away_team', 'home_score', 'away_score'])
        if before - len(df) > 0:
            logger.info(f"Dropped {before - len(df):,} rows with missing crucial fields in results.")
            
        # Ensure scores are numeric and non-negative
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
        
        # Remove impossible score records
        invalid_scores = (df['home_score'] < 0) | (df['away_score'] < 0) | df['home_score'].isna() | df['away_score'].isna()
        if invalid_scores.sum() > 0:
            logger.warning(f"Removing {invalid_scores.sum()} records with invalid scores.")
            df = df[~invalid_scores]
            
        # Fill missing neutral values with False
        if 'neutral' in df.columns:
            df['neutral'] = df['neutral'].fillna(False).astype(bool)
            
        # Ensure unique primary keys (date, home_team, away_team)
        before_pk = len(df)
        df = df.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='first')
        if before_pk - len(df) > 0:
            logger.info(f"Removed {before_pk - len(df)} rows with duplicate primary keys (date, home_team, away_team).")
            
        return df

    def clean_elo_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Specific cleaning for elo_ratings_wc2026.csv."""
        df = df.copy()
        df = self.remove_duplicates(df)
        df = self.clean_text_columns(df, ['country', 'country_code', 'confederation'])
        df = self.standardize_dates(df, ['snapshot_date'])
        
        # Fill missing values if any
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
        
        return df

    def clean_fifa_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Specific cleaning for fifa_mens_rank.csv."""
        df = df.copy()
        df = self.remove_duplicates(df)
        df = self.clean_text_columns(df, ['team', 'acronym'])
        df = self.standardize_dates(df, ['date'])
        
        # Point columns to float
        df['total.points'] = pd.to_numeric(df['total.points'], errors='coerce')
        df['rank'] = pd.to_numeric(df['rank'], errors='coerce').astype(int)
        
        return df

    def clean_player_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Specific cleaning for player_aggregates.csv."""
        df = df.copy()
        df = self.remove_duplicates(df)
        df = self.clean_text_columns(df, ['country'])
        
        # Average ratings columns to float
        numeric_cols = [c for c in df.columns if c not in ['country', 'fifa_version']]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df

    def clean_teams_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """Specific cleaning for teams_form.csv."""
        df = df.copy()
        df = self.remove_duplicates(df)
        df = self.clean_text_columns(df, ['team'])
        df = self.standardize_dates(df, ['match_date'])
        
        df['win_rate'] = pd.to_numeric(df['win_rate'], errors='coerce')
        df['avg_goals_scored'] = pd.to_numeric(df['avg_goals_scored'], errors='coerce')
        df['avg_goals_conceded'] = pd.to_numeric(df['avg_goals_conceded'], errors='coerce')
        
        return df
