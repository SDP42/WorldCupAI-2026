import pandas as pd
from typing import Tuple
from src.utils.logger import setup_logger

logger = setup_logger("cross_validation")

def time_aware_split(df: pd.DataFrame, 
                     train_end: str = "2018-12-31", 
                     val_end: str = "2022-12-31") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a strict time-aware chronological split of the dataset.
    This prevents temporal data leakage by ensuring that validation and test data
    occur strictly after training data.
    """
    df = df.copy()
    # Sort chronologically by date
    df = df.sort_values('date').reset_index(drop=True)
    
    train_df = df[df['date'] <= pd.to_datetime(train_end)].copy()
    val_df = df[(df['date'] > pd.to_datetime(train_end)) & (df['date'] <= pd.to_datetime(val_end))].copy()
    test_df = df[df['date'] > pd.to_datetime(val_end)].copy()
    
    logger.info(f"Chronological split completed:")
    logger.info(f"  Training Set:   {len(train_df):,} matches (up to {train_end})")
    logger.info(f"  Validation Set: {len(val_df):,} matches ({train_end} to {val_end})")
    logger.info(f"  Test Set:       {len(test_df):,} matches (after {val_end})")
    
    return train_df, val_df, test_df
