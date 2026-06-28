import os
import pickle
from typing import Tuple, Any
from src.utils.logger import setup_logger

logger = setup_logger("model_loader")

def load_model_and_pipeline(model_dir: str) -> Tuple[Any, Any]:
    """Loads a trained model and its preprocessing pipeline from a directory."""
    model_path = os.path.join(model_dir, "model.pkl")
    prep_path = os.path.join(model_dir, "preprocessing.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(prep_path):
        raise FileNotFoundError(f"Preprocessing pipeline file not found: {prep_path}")
        
    logger.info(f"Loading model and pipeline from {model_dir}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(prep_path, "rb") as f:
        pipeline = pickle.load(f)
        
    return model, pipeline
