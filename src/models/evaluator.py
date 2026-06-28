import os
import time
import json
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from src.models.metrics import compute_classification_metrics
from src.utils.logger import setup_logger

logger = setup_logger("evaluator")

class ModelEvaluator:
    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series, training_time: float) -> dict:
        """Evaluates model performance and plots/saves key evaluation artifacts."""
        logger.info(f"Evaluating model '{self.model_name}'...")
        
        # Measure prediction time and memory usage
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        
        start_time = time.time()
        y_prob = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_time
        
        mem_after = process.memory_info().rss
        mem_usage_mb = (mem_after - mem_before) / (1024 * 1024)
        
        # Compute core metrics (accept both pandas Series and numpy arrays)
        metrics = compute_classification_metrics(np.asarray(y_test), y_pred, y_prob)
        metrics["training_time_sec"] = training_time
        metrics["prediction_time_sec"] = pred_time
        metrics["memory_usage_mb"] = max(0.0, mem_usage_mb)
        
        # Save metrics.json
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
            
        # Save classification report
        class_report = classification_report(y_test, y_pred, target_names=["Away Win", "Draw", "Home Win"], zero_division=0)
        report_path = os.path.join(self.output_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(class_report)
            
        # Save Confusion Matrix plot
        self._plot_confusion_matrix(np.asarray(y_test), y_pred)
        
        # Save ROC Curve plot
        self._plot_roc_curve(np.asarray(y_test), y_prob)
        
        logger.info(f"Successfully evaluated '{self.model_name}'. Accuracy: {metrics['accuracy']:.4f}")
        return metrics

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        class_names = ["Away Win", "Draw", "Home Win"]
        
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {self.model_name}")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add labels
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
                         
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def _plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray):
        n_classes = y_prob.shape[1]
        class_names = ["Away Win", "Draw", "Home Win"]
        
        plt.figure(figsize=(8, 6))
        
        for i in range(n_classes):
            # One-vs-rest ROC
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f"ROC curve of class {class_names[i]} (area = {roc_auc:.2f})")
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        path = os.path.join(self.output_dir, "roc_curve.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
