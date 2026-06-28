import pandas as pd
from abc import ABC, abstractmethod

class BaseFeatureGenerator(ABC):
    """Abstract base class for all feature generators."""
    
    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a dataframe (e.g. master_dataset) and returns a dataframe 
        with the engineered features. The returned dataframe must have 
        the same index or length as the input.
        """
        pass
