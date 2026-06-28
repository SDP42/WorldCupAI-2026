import os
import pandas as pd
from typing import Dict, Set
from src.utils.logger import setup_logger

logger = setup_logger("harmonizer")

class TeamHarmonizer:
    # Baseline dictionary mapping various team/country name variants to canonical names.
    # The canonical names are based on the naming convention in results.csv.
    BASE_MAPPING = {
        # USA
        "USA": "United States",
        "United States America": "United States",
        "US America": "United States",
        
        # Korea
        "Korea Republic": "South Korea",
        "Korea, Republic of": "South Korea",
        "Korea DPR": "North Korea",
        "Korea, Democratic People's Republic of": "North Korea",
        
        # Iran
        "IR Iran": "Iran",
        "Iran, Islamic Republic of": "Iran",
        
        # Ivory Coast
        "Côte d'Ivoire": "Cote d'Ivoire",
        "Ivory Coast": "Cote d'Ivoire",
        
        # Congo
        "DR Congo": "Democratic Republic of the Congo",
        "Congo DR": "Democratic Republic of the Congo",
        "Congo-Kinshasa": "Democratic Republic of the Congo",
        "Zaire": "Democratic Republic of the Congo",
        "Congo-Brazzaville": "Congo",
        
        # Ireland
        "Republic of Ireland": "Ireland",
        
        # Cape Verde
        "Cabo Verde": "Cape Verde",
        
        # Curacao
        "Curaçao": "Curacao",
        
        # Czechia
        "Czechia": "Czech Republic",
        
        # Kyrgyzstan
        "Kyrgyz Republic": "Kyrgyzstan",
        
        # St. Vincent/Grenadines
        "Saint Vincent and the Grenadines": "St. Vincent and the Grenadines",
        "St. Vincent/Grenadines": "St. Vincent and the Grenadines",
        
        # St. Kitts/Nevis
        "Saint Kitts and Nevis": "St. Kitts and Nevis",
        "St. Kitts/Nevis": "St. Kitts and Nevis",
        
        # East Germany / West Germany
        "German DR": "East Germany",
        "Germany DR": "East Germany",
        "Germany FR": "West Germany",
        "Federal Republic of Germany": "West Germany",
        
        # USSR / Russia
        "Soviet Union": "Russia",
        "USSR": "Russia",
        
        # Yugoslavia
        "Yugoslavia": "Serbia",
        "Serbia and Montenegro": "Serbia",
        
        # Czechoslovakia
        "Czechoslovakia": "Czech Republic",
    }

    def __init__(self, mapping_path: str = "mappings/team_name_mapping.csv"):
        self.mapping_path = mapping_path
        self.mapping = self.BASE_MAPPING.copy()
        
        if os.path.exists(self.mapping_path):
            logger.info(f"Loading existing team name mapping from {self.mapping_path}")
            self.load_mapping_from_file()
        else:
            logger.info(f"No mapping file found at {self.mapping_path}. Creating a new one.")
            self.save_mapping_to_file()

    def load_mapping_from_file(self):
        try:
            df = pd.read_csv(self.mapping_path)
            for _, row in df.iterrows():
                self.mapping[str(row['variant'])] = str(row['canonical'])
        except Exception as e:
            logger.error(f"Error loading mapping file: {e}")

    def save_mapping_to_file(self):
        try:
            os.makedirs(os.path.dirname(self.mapping_path), exist_ok=True)
            records = [{"variant": k, "canonical": v} for k, v in self.mapping.items()]
            df = pd.DataFrame(records)
            df.to_csv(self.mapping_path, index=False)
            logger.info(f"Saved {len(df)} team name mappings to {self.mapping_path}")
        except Exception as e:
            logger.error(f"Error saving mapping file: {e}")

    def harmonize_name(self, name: str) -> str:
        """Standardizes a single team/country name."""
        if pd.isna(name) or not isinstance(name, str):
            return name
        
        trimmed = name.strip()
        # Direct lookup in mapping
        if trimmed in self.mapping:
            return self.mapping[trimmed]
            
        return trimmed

    def harmonize_dataframe(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Standardizes team names in specified columns of a DataFrame."""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(self.harmonize_name)
        return df

    def verify_coverage(self, df: pd.DataFrame, team_col: str, canonical_teams: Set[str], df_name: str):
        """Checks if any team names in the dataframe are not present in the canonical set."""
        if team_col not in df.columns:
            return
        
        unique_teams = set(df[team_col].dropna().unique())
        unmatched = unique_teams - canonical_teams
        
        if unmatched:
            logger.warning(f"[{df_name}] {len(unmatched)} team names could not be matched with canonical results team list.")
            logger.warning(f"Sample unmatched: {list(unmatched)[:10]}")
            # Automatically add self-mapping for logging/future resolution if desired
            for team in unmatched:
                if team not in self.mapping:
                    logger.info(f"Implicitly mapping '{team}' to itself.")
        else:
            logger.info(f"[{df_name}] 100% team name match coverage against canonical list.")
