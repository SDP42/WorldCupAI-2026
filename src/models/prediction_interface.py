import pandas as pd
import numpy as np
from typing import Tuple, List
from src.models.model_loader import load_model_and_pipeline

class PredictionInterface:
    def __init__(self, model_dir: str):
        self.model, self.pipeline = load_model_and_pipeline(model_dir)

    def predict_match(self, home_team: str, away_team: str, pre_match_features: pd.DataFrame) -> Tuple[str, List[float]]:
        """
        Generates probability predictions for a single match.
        Returns:
            predicted_outcome: One of ["Away Win", "Draw", "Home Win"]
            probabilities: Probability distributions corresponding to [Away Win, Draw, Home Win]
        """
        # Preprocess features
        X_processed = self.pipeline.transform(pre_match_features)
        
        # Predict
        probs = self.model.predict_proba(X_processed)[0] # First row
        pred_idx = np.argmax(probs)
        
        class_names = ["Away Win", "Draw", "Home Win"]
        return class_names[pred_idx], list(probs)
