import pandas as pd
from src.features.base_features import BaseFeatureGenerator

class DefenceFeatures(BaseFeatureGenerator):
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        features = pd.DataFrame(index=df.index)
        
        # Global average goals conceded is also around 1.4.
        global_avg_goals = 1.4
        
        if 'home_form_avg_goals_conceded_10' in df.columns:
            # Lower conceded goals is better, so rating is inverse or we just keep it as is (lower = better defence)
            # A standard defensive rating is: global_avg_goals / team_avg_goals_conceded (higher = better defence)
            features['home_defence_rating'] = global_avg_goals / df['home_form_avg_goals_conceded_10'].replace(0, 0.1)
        if 'away_form_avg_goals_conceded_10' in df.columns:
            features['away_defence_rating'] = global_avg_goals / df['away_form_avg_goals_conceded_10'].replace(0, 0.1)
            
        if 'home_form_clean_sheet_rate_10' in df.columns:
            features['home_clean_sheet_rate'] = df['home_form_clean_sheet_rate_10']
        if 'away_form_clean_sheet_rate_10' in df.columns:
            features['away_clean_sheet_rate'] = df['away_form_clean_sheet_rate_10']
            
        return features.fillna(1.0)
