import os
import pandas as pd
from src.features.base_features import BaseFeatureGenerator
from src.data.loader import DataLoader
from src.utils.logger import setup_logger

logger = setup_logger("tournament_features")

class TournamentFeatures(BaseFeatureGenerator):
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        self.loader = DataLoader(config_path=config_path)
        
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges historical tournament experience features from train.csv and test.csv
        based on the match year and team name.
        """
        df = df.copy()
        features = pd.DataFrame(index=df.index)
        
        # Load train and test datasets from the FIFA World Cup Dataset
        try:
            train_df = self.loader.load_csv("train_dataset")
            test_df = self.loader.load_csv("test_dataset")
            
            # Combine them to have a complete mapping of team-years to tournament stats
            comb_df = pd.concat([train_df, test_df], ignore_index=True)
            
            # Standardize team names in the tournament dataset using the harmonizer mapping
            # (We will do a simple mapping here or rely on the caller. Let's do a quick standardization).
            # To avoid circular dependency, we just use a simple mapping or assume they match.
            # USA -> United States, Korea Republic -> South Korea, etc.
            name_map = {
                "USA": "United States",
                "Korea Republic": "South Korea",
                "IR Iran": "Iran",
                "Côte d'Ivoire": "Cote d'Ivoire",
                "Congo DR": "Democratic Republic of the Congo"
            }
            comb_df['team'] = comb_df['team'].replace(name_map)
            
            # Keep only the experience features
            exp_cols = [
                'version', 'team', 'world_cup_titles_before', 'world_cup_participations_before',
                'groups_passed_before', 'round16_before', 'quarterfinals_before', 
                'semifinals_before', 'finals_before'
            ]
            comb_df = comb_df[exp_cols].drop_duplicates()
            
            # Extract year from match date
            df['match_year'] = df['date'].dt.year
            
            # For each match, we want to find the tournament stats of the home and away teams.
            # Since matches happen throughout the year, we will match:
            # - If match_year matches a tournament version (e.g. 2006, 2010, 2014, 2018, 2022, 2026), we use that version.
            # - Otherwise, we find the latest tournament version that is strictly less than match_year.
            # To do this cleanly and without leakage, we can do a merge_asof or a simple loop over tournament years.
            tournament_years = sorted(comb_df['version'].unique())
            
            # Map each match_year to the latest tournament version
            year_to_version = {}
            for yr in df['match_year'].unique():
                # Find latest tournament year <= yr
                past_versions = [v for v in tournament_years if v <= yr]
                if past_versions:
                    year_to_version[yr] = max(past_versions)
                else:
                    year_to_version[yr] = min(tournament_years) # default to earliest
                    
            df['target_version'] = df['match_year'].map(year_to_version)
            
            # Home team experience
            home_exp = comb_df.rename(columns={col: f'home_{col}' for col in exp_cols if col != 'team'})
            merged_home = pd.merge(
                df,
                home_exp,
                left_on=['target_version', 'home_team'],
                right_on=['home_version', 'team'],
                how='left'
            )
            
            # Away team experience
            away_exp = comb_df.rename(columns={col: f'away_{col}' for col in exp_cols if col != 'team'})
            merged_away = pd.merge(
                df,
                away_exp,
                left_on=['target_version', 'away_team'],
                right_on=['away_version', 'team'],
                how='left'
            )
            
            # Add to features
            cols_to_add = [
                'world_cup_titles_before', 'world_cup_participations_before',
                'groups_passed_before', 'round16_before', 'quarterfinals_before', 
                'semifinals_before', 'finals_before'
            ]
            for col in cols_to_add:
                features[f'home_{col}'] = merged_home[f'home_{col}']
                features[f'away_{col}'] = merged_away[f'away_{col}']
                
            # Fill NaNs with 0 (indicates no prior World Cup experience)
            features = features.fillna(0)
            
        except Exception as e:
            logger.error(f"Failed to generate tournament experience features: {e}", exc_info=True)
            
        return features
