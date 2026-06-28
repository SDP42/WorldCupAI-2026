"""WorldCupAI — Phase 6: Thin metrics wrapper.

Delegates entirely to src.models.metrics.compute_classification_metrics
so Phase 6 uses identical metric computation to Phases 4 and 5.
No duplication.
"""
from src.models.metrics import compute_classification_metrics  # reuse Phase 4/5
from src.models.calibrator import compute_ece, compute_mce      # reuse Phase 5

__all__ = ["compute_classification_metrics", "compute_ece", "compute_mce"]
