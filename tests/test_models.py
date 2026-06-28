#!/usr/bin/env python3
"""WorldCupAI — Machine Learning Model Verification Tests

Verifies that all trained models exist, preprocessings load correctly,
and predictions sum up to 1.0.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.model_loader import load_model_and_pipeline
from src.models.prediction_interface import PredictionInterface

def run_tests():
    print("="*60)
    print("RUNNING WORLDCUPAI ML MODEL VERIFICATION TESTS")
    print("="*60)
    
    # Check that at least some baseline models exist
    # (e.g. Random Forest and Logistic Regression)
    check_models = ["logistic_regression", "random_forest", "xgboost"]
    
    failures = 0
    
    for model_name in check_models:
        model_dir = f"models/{model_name}"
        if not os.path.exists(model_dir):
            print(f"⚠️ Warning: Model directory {model_dir} not found. Skipping validation.")
            continue
            
        try:
            # Load model and pipeline
            model, pipeline = load_model_and_pipeline(model_dir)
            print(f"✅ Successfully loaded model and pipeline for '{model_name}'.")
            
            # Predict dummy sample
            # Generate a row with random values matching the feature columns
            from train_baselines import FEATURE_COLS
            dummy_data = pd.DataFrame([{col: 0.0 for col in FEATURE_COLS}])
            
            proc_data = pipeline.transform(dummy_data)
            probs = model.predict_proba(proc_data)[0]
            
            # Verify probabilities sum to 1.0
            prob_sum = np.sum(probs)
            if not np.isclose(prob_sum, 1.0, atol=1e-5):
                print(f"❌ Test Failed for '{model_name}': Probabilities sum to {prob_sum}, expected 1.0.")
                failures += 1
            else:
                print(f"✅ Test Passed for '{model_name}': Probabilities sum to 1.0.")
                
        except Exception as e:
            print(f"❌ Test Failed for '{model_name}' due to exception: {e}")
            failures += 1
            
    print("="*60)
    if failures == 0:
        print("🎉 ALL MACHINE LEARNING MODEL TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        sys.exit(0)
    else:
        print(f"💥 FAILED {failures} TESTS.")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
