import os
import time
import pickle
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils.logger import setup_logger

logger = setup_logger("trainer")

class PreprocessingPipeline:
    def __init__(self, feature_cols: List[str], scale: bool = False):
        self.feature_cols = feature_cols
        self.scale = scale
        self.pipeline = None

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Pipeline]:
        """Fits and transforms the dataset based on feature columns and scaling preference."""
        df = df[self.feature_cols].copy()
        
        # Identify numeric and categorical columns
        numeric_cols = [c for c in self.feature_cols if np.issubdtype(df[c].dtype, np.number)]
        categorical_cols = [c for c in self.feature_cols if c not in numeric_cols]
        
        # Define transformers
        numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
        if self.scale:
            numeric_steps.append(('scaler', StandardScaler()))
        numeric_transformer = Pipeline(steps=numeric_steps)
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        transformed = self.pipeline.fit_transform(df)
        
        logger.info(f"Fitted preprocessing pipeline. Scale={self.scale}. Output shape: {transformed.shape}")
        return transformed, self.pipeline

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError("Pipeline has not been fitted yet!")
        return self.pipeline.transform(df[self.feature_cols])

def save_preprocessing_pipeline(pipeline: Pipeline, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Preprocessing pipeline saved to {path}")

def prepare_targets(df: pd.DataFrame) -> pd.Series:
    """Prepares the 3-class target: 2 for Home Win, 1 for Draw, 0 for Away Win."""
    conditions = [
        df['home_score'] > df['away_score'],
        df['home_score'] == df['away_score'],
        df['home_score'] < df['away_score']
    ]
    choices = [2, 1, 0] # 2: Home, 1: Draw, 0: Away
    return pd.Series(np.select(conditions, choices, default=1), index=df.index)
