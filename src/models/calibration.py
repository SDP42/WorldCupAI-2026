import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from src.utils.logger import setup_logger

logger = setup_logger("calibration")

def plot_multiclass_calibration(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, save_path: str):
    """Plots and saves calibration curves for a 3-class target."""
    n_classes = y_prob.shape[1]
    class_names = ["Away Win", "Draw", "Home Win"]
    
    plt.figure(figsize=(8, 6))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    
    for i in range(n_classes):
        # Convert multiclass to binary indicators for class i
        y_true_binary = (y_true == i).astype(int)
        prob_i = y_prob[:, i]
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true_binary, prob_i, n_bins=10, strategy='uniform')
        
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{class_names[i]}")
        
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted probability")
    plt.title(f"Calibration Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Calibration plot saved to {save_path}")

def recommend_calibration_method(y_true: np.ndarray, y_prob: np.ndarray) -> str:
    """Recommends Platt Scaling or Isotonic Regression based on dataset size."""
    # Isotonic Regression is non-parametric and works best with larger calibration sets (>1000 samples).
    # Platt Scaling (logistic calibration) works better for smaller sets (<1000 samples).
    # Let's check size of y_true.
    if len(y_true) > 1000:
        return "Isotonic Regression (due to larger validation sample size of >1000)"
    else:
        return "Platt Scaling (due to smaller validation sample size of <1000)"
