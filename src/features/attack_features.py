import pandas as pd
from src.features.base_features import BaseFeatureGenerator

class AttackFeatures(BaseFeatureGenerator):
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        features = pd.DataFrame(index=df.index)
        
        # We will use the rolling form averages if they are in the dataframe, 
        # or we will compute them here. Since they are computed in RecentFormFeatures,
        # we can assume they are available in the merged dataframe or we can compute them relative to global averages.
        # Let's compute relative attacking ratings.
        
        # Global average goals scored in the modern era is around 1.3 to 1.5 per team per match.
        # We will compute the rolling average goals scored and divide by a baseline or the actual rolling global average.
        global_avg_goals = 1.4
        
        # We will look for 'home_form_avg_goals_scored_10' in df or we can compute it.
        # Since we merge them in the feature store, we can calculate the relative rating here:
        if 'home_form_avg_goals_scored_10' in df.columns:
            features['home_attack_rating'] = df['home_form_avg_goals_scored_10'] / global_avg_goals
        if 'away_form_avg_goals_scored_10' in df.columns:
            features['away_attack_rating'] = df['away_form_avg_goals_scored_10'] / global_avg_goals
            
        if 'home_form_win_rate_10' in df.columns and 'away_form_win_rate_10' in df.columns:
            features['attack_ratio'] = (df['home_form_avg_goals_scored_10'] + 1) / (df['away_form_avg_goals_scored_10'] + 1)
            
        return features.fillna(1.0)
