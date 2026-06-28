import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Tuple

class BaseModel(ABC):
    """Abstract base class for all machine learning models in WorldCupAI."""
    
    def __init__(self, model_name: str, model: Any):
        self.model_name = model_name
        self.model = model
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Trains the underlying model."""
        self.model.fit(X, y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates hard predictions (classes)."""
        return self.model.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generates probability predictions for all classes."""
        return self.model.predict_proba(X)
