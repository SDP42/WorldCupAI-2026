"""WorldCupAI — Phase 7: Unified Ensemble Pipeline.

Wraps all base models, preprocessing pipelines, sequence builders,
optimized weights, and meta-learners in a single loadable class.
This makes final tournament predictions in Phase 8 completely turnkey.
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from src.deep_learning.prediction_interface import DLPredictionInterface
from src.deep_learning.ann_model import ANNModel
from src.deep_learning.lstm_model import LSTMModel
from src.deep_learning.dataset import SequenceBuilder
from src.ensemble.voting import WeightedSoftVotingClassifier, SoftVotingClassifier, HardVotingClassifier
from src.ensemble.stacking import StackingClassifier, BlendingClassifier
from src.utils.logger import setup_logger

logger = setup_logger("ensemble_pipeline")


class EnsemblePipeline:
    """Production ensemble pipeline wrapping all inference modules.

    Args:
        model_dirs: Dict mapping model name to directory path.
                    e.g. {'XGBoost': 'models/xgboost_optimized', 'ANN': 'models/ann'}
        optimized_weights: Dict mapping model name to soft voting weights.
        stacking_model:    Fitted StackingClassifier instance.
        blending_model:    Fitted BlendingClassifier instance.
        best_method:       Name of the recommended ensemble method (e.g. 'Stacking').
    """

    def __init__(
        self,
        model_dirs:        Dict[str, str],
        optimized_weights: Optional[Dict[str, float]] = None,
        stacking_model:    Optional[StackingClassifier] = None,
        blending_model:    Optional[BlendingClassifier] = None,
        best_method:       str = "Weighted Soft Voting",
    ):
        self.model_dirs        = model_dirs
        self.optimized_weights = optimized_weights or {}
        self.stacking_model    = stacking_model
        self.blending_model    = blending_model
        self.best_method       = best_method

        self.ml_models: Dict[str, Tuple[Any, Any]] = {}
        self.ann_interface: Optional[DLPredictionInterface] = None
        self.lstm_interface: Optional[DLPredictionInterface] = None
        self.seq_builder: Optional[SequenceBuilder] = None

    def load_base_models(self):
        """Loads all base ML models, DL prediction interfaces, and sequence configs."""
        logger.info("EnsemblePipeline: Loading all base models...")

        for name, m_dir in self.model_dirs.items():
            if name == "ANN":
                self.ann_interface = DLPredictionInterface(m_dir, ANNModel)
            elif name == "LSTM":
                self.lstm_interface = DLPredictionInterface(m_dir, LSTMModel)
                # Load sequence config
                cfg_path = os.path.join(m_dir, "sequence_config.json")
                if os.path.exists(cfg_path):
                    import json
                    with open(cfg_path) as f:
                        seq_cfg = json.load(f)
                    self.seq_builder = SequenceBuilder(seq_len=seq_cfg["seq_len"])
                else:
                    self.seq_builder = SequenceBuilder(seq_len=5)
            else:
                # Load ML model
                prep_path = os.path.join(m_dir, "preprocessing.pkl")
                cal_path  = os.path.join(m_dir, "calibrated_model.pkl")
                if not os.path.exists(prep_path) or not os.path.exists(cal_path):
                    logger.warning(f"ML Model '{name}' artifacts not found in {m_dir} — skipping.")
                    continue

                with open(prep_path, "rb") as f:
                    pipeline = pickle.load(f)
                with open(cal_path, "rb") as f:
                    cal_model = pickle.load(f)

                self.ml_models[name] = (pipeline, cal_model)

        logger.info("EnsemblePipeline: Base models successfully loaded.")

    def get_base_predictions(
        self,
        test_df:       pd.DataFrame,
        df_full:       Optional[pd.DataFrame] = None,
        feature_cols:  Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Generates aligned predictions for all loaded base models.

        Args:
            test_df:      DataFrame containing matches to predict.
            df_full:      Full chronological DataFrame (required for LSTM sequence window lookback).
            feature_cols: List of features to use.

        Returns:
            Dict matching layout:
            {
               model_name: { "y_prob": (N, 3), "y_pred": (N,) }
            }
        """
        predictions = {}
        n_samples   = len(test_df)

        # 1. Generate ML predictions
        for name, (pipeline, model) in self.ml_models.items():
            X_test_proc = pipeline.transform(test_df)
            y_prob = model.predict_proba(X_test_proc)
            y_pred = model.predict(X_test_proc)
            predictions[name] = {"y_prob": y_prob, "y_pred": y_pred}

        # 2. Generate ANN predictions
        if self.ann_interface:
            ann_prep_path = os.path.join(self.model_dirs["ANN"], "preprocessing.pkl")
            with open(ann_prep_path, "rb") as f:
                ann_prep = pickle.load(f)
            X_ann = ann_prep.transform(test_df)
            y_prob = self.ann_interface.predict_proba(X_ann)
            y_pred = np.argmax(y_prob, axis=1)
            predictions["ANN"] = {"y_prob": y_prob, "y_pred": y_pred}

        # 3. Generate LSTM predictions
        if self.lstm_interface and self.seq_builder:
            if df_full is None:
                raise ValueError("df_full must be provided to build sequences for LSTM model!")

            # Load the LSTM's preprocessing pipeline
            lstm_prep_path = os.path.join(self.model_dirs["LSTM"], "preprocessing.pkl")
            with open(lstm_prep_path, "rb") as f:
                lstm_prep = pickle.load(f)

            # Preprocess the entire chronological df_full
            # (matches the training sequence preprocessing strategy)
            # Ensure we are using the exact same feature columns
            f_cols = feature_cols or list(self.ml_models.values())[0][0].steps[0][1].transformers[0][2]
            
            X_full_proc = lstm_prep.transform(df_full)
            y_full      = df_full["target"].values if "target" in df_full.columns else np.zeros(len(df_full))

            # Locate test_df indices inside df_full
            # In Phase 6 / 7, we track the absolute indices of the split matches
            # Let's align on match date/home_team/away_team keys to find indices
            test_indices = []
            for _, row in test_df.iterrows():
                # Locate index in df_full
                mask = (df_full["date"] == row["date"]) & \
                       (df_full["home_team"] == row["home_team"]) & \
                       (df_full["away_team"] == row["away_team"])
                matched = df_full[mask].index.values
                if len(matched) > 0:
                    test_indices.append(matched[0])
                else:
                    raise ValueError(f"Match not found in df_full: {row['date']} {row['home_team']} vs {row['away_team']}")

            # Build sequences
            X_seq_test, _ = self.seq_builder.build(X_full_proc, y_full, np.array(test_indices))
            y_prob = self.lstm_interface.predict_proba(X_seq_test)
            y_pred = np.argmax(y_prob, axis=1)
            predictions["LSTM"] = {"y_prob": y_prob, "y_pred": y_pred}

        return predictions

    def predict_proba(
        self,
        test_df: pd.DataFrame,
        df_full: Optional[pd.DataFrame] = None,
        method:  Optional[str] = None,
    ) -> np.ndarray:
        """Runs base predictions and combines them via the chosen ensemble method."""
        chosen_method = method or self.best_method
        base_preds = self.get_base_predictions(test_df, df_full)
        all_models = list(base_preds.keys())

        if chosen_method == "Soft Voting":
            voter = SoftVotingClassifier(all_models)
            return voter.predict_proba(base_preds)
            
        elif chosen_method == "Weighted Soft Voting":
            voter = WeightedSoftVotingClassifier(all_models, self.optimized_weights)
            return voter.predict_proba(base_preds)
            
        elif chosen_method == "Stacking":
            if self.stacking_model is None:
                raise ValueError("StackingClassifier model not fitted or loaded!")
            return self.stacking_model.predict_proba(base_preds)
            
        elif chosen_method == "Blending":
            if self.blending_model is None:
                raise ValueError("BlendingClassifier model not fitted or loaded!")
            return self.blending_model.predict_proba(base_preds)
            
        else:
            raise ValueError(f"Unknown ensemble method: {chosen_method}")

    def predict(
        self,
        test_df: pd.DataFrame,
        df_full: Optional[pd.DataFrame] = None,
        method:  Optional[str] = None,
    ) -> np.ndarray:
        """Returns hard class outcomes (0/1/2) for the ensemble."""
        chosen_method = method or self.best_method
        base_preds = self.get_base_predictions(test_df, df_full)
        all_models = list(base_preds.keys())

        if chosen_method == "Hard Voting":
            voter = HardVotingClassifier(all_models)
            return voter.predict(base_preds)
        else:
            probs = self.predict_proba(test_df, df_full, chosen_method)
            return np.argmax(probs, axis=1)
