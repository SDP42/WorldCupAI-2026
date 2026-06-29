import pandas as pd
import numpy as np
from src.features.base_features import BaseFeatureGenerator
from src.utils.logger import setup_logger

logger = setup_logger("head_to_head")

class HeadToHeadFeatures(BaseFeatureGenerator):
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes historical head-to-head (H2H) features between home and away teams.
        Vectorized using sorted team-pair groupings and shifted cumulative sums to prevent leakage.

        IMPORTANT — Neutral Venue Note:
          In the context of the FIFA World Cup (neutral venue), the labels 'h2h_home_wins'
          and 'h2h_away_wins' are **positional** — they refer to the team listed as
          'home_team' (Team 1) and 'away_team' (Team 2) in the bracket, respectively.
          They do NOT imply any actual home-ground advantage. The feature names are
          kept as-is because the trained models depend on these exact column names.
        """
        df = df.copy()
        df = df.sort_values('date')
        
        # 1. Create a sorted team-pair key to group matches between the same two teams
        df['team_min'] = df.apply(lambda r: min(r['home_team'], r['away_team']), axis=1)
        df['team_max'] = df.apply(lambda r: max(r['home_team'], r['away_team']), axis=1)
        
        # Calculate outcome from the perspective of team_min
        df['is_home_min'] = (df['home_team'] == df['team_min'])
        df['goals_min'] = np.where(df['is_home_min'], df['home_score'], df['away_score'])
        df['goals_max'] = np.where(df['is_home_min'], df['away_score'], df['home_score'])
        
        df['gd_min'] = df['goals_min'] - df['goals_max']
        df['win_min'] = (df['gd_min'] > 0).astype(int)
        df['loss_min'] = (df['gd_min'] < 0).astype(int)
        df['draw_min'] = (df['gd_min'] == 0).astype(int)
        
        # 2. Group by team-pair and compute shifted cumulative sums
        grouped = df.groupby(['team_min', 'team_max'])
        
        # Shifted outcomes (excluding the current match)
        shifted_win_min = grouped['win_min'].shift(1).fillna(0)
        shifted_loss_min = grouped['loss_min'].shift(1).fillna(0)
        shifted_draw_min = grouped['draw_min'].shift(1).fillna(0)
        shifted_gd_min = grouped['gd_min'].shift(1).fillna(0)
        shifted_count = grouped['date'].shift(1).notna().astype(int) # 1 if there was a prior match, else 0
        
        # Cumulative sums
        df['cum_meetings'] = grouped[tuple(list(grouped.groups.keys())[0])].cumcount() if len(grouped) == 0 else grouped['date'].cumcount()
        # Wait, cumcount doesn't need to be shifted because it counts elements before the current index!
        # Actually, let's just use cumsum of shifted_count to be absolutely safe and clear.
        df['h2h_meetings'] = grouped['win_min'].transform(lambda x: x.shift(1).fillna(0).cumsum()).astype(int)
        # Wait, the above is wrong. Let's do it simpler:
        df['h2h_meetings'] = grouped['is_home_min'].transform(lambda x: x.shift(1).notna().astype(int).cumsum()).astype(int)
        
        df['h2h_wins_min'] = grouped['win_min'].transform(lambda x: x.shift(1).fillna(0).cumsum())
        df['h2h_losses_min'] = grouped['loss_min'].transform(lambda x: x.shift(1).fillna(0).cumsum())
        df['h2h_draws'] = grouped['draw_min'].transform(lambda x: x.shift(1).fillna(0).cumsum())
        df['h2h_gd_min'] = grouped['gd_min'].transform(lambda x: x.shift(1).fillna(0).cumsum())
        
        # 3. Map min-perspective features back to home/away perspective
        features = pd.DataFrame(index=df.index)
        
        # If the current home team is team_min:
        # - home_wins = wins_min
        # - away_wins = losses_min
        # - gd = gd_min
        # If the current home team is team_max:
        # - home_wins = losses_min
        # - away_wins = wins_min
        # - gd = -gd_min
        
        features['h2h_meetings'] = df['h2h_meetings']
        features['h2h_draws'] = df['h2h_draws']
        
        features['h2h_home_wins'] = np.where(df['is_home_min'], df['h2h_wins_min'], df['h2h_losses_min']).astype(int)
        features['h2h_away_wins'] = np.where(df['is_home_min'], df['h2h_losses_min'], df['h2h_wins_min']).astype(int)
        features['h2h_gd'] = np.where(df['is_home_min'], df['h2h_gd_min'], -df['h2h_gd_min'])
        
        # Fill NaNs with 0 for matches with no prior H2H history
        features = features.fillna(0)
        
        return features
