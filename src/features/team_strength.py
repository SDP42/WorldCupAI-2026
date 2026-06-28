import pandas as pd
from src.features.base_features import BaseFeatureGenerator

class TeamStrengthFeatures(BaseFeatureGenerator):
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        features = pd.DataFrame(index=df.index)
        
        # 1. Elo differences and ratios
        if 'home_elo' in df.columns and 'away_elo' in df.columns:
            features['elo_diff'] = df['home_elo'] - df['away_elo']
            # Avoid division by zero
            features['elo_ratio'] = df['home_elo'] / df['away_elo'].replace(0, 1)
            
        # 2. FIFA rank differences and ratios
        if 'home_rank' in df.columns and 'away_rank' in df.columns:
            features['rank_diff'] = df['home_rank'] - df['away_rank']
            features['rank_ratio'] = df['home_rank'] / df['away_rank'].replace(0, 1)
            
        # 3. EA FC Overall differences and ratios
        if 'home_avg_overall' in df.columns and 'away_avg_overall' in df.columns:
            features['overall_diff'] = df['home_avg_overall'] - df['away_avg_overall']
            features['overall_ratio'] = df['home_avg_overall'] / df['away_avg_overall'].replace(0, 1)
            
        # 4. EA FC Attack/Defense differences
        if 'home_avg_attack' in df.columns and 'away_avg_attack' in df.columns:
            features['attack_diff'] = df['home_avg_attack'] - df['away_avg_attack']
        if 'home_avg_defense' in df.columns and 'away_avg_defense' in df.columns:
            features['defense_diff'] = df['home_avg_defense'] - df['away_avg_defense']
            
        return features
