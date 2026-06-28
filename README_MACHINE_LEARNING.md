# 🤖 WorldCupAI — Machine Learning Framework Documentation

This document describes the design and components of the WorldCupAI model training framework.

## 1. Modular Framework (`src/models/`)

The framework is structured as a modular package to support easy integration of baseline models, deep learning models, and ensembles:

- [`src/models/base_model.py`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/src/models/base_model.py): Defines the abstract wrapper class `BaseModel` to provide unified interfaces for predictions.
- [`src/models/trainer.py`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/src/models/trainer.py): Implements preprocessing pipelines (handling scaling preferences for different algorithms) and target label preparation.
- [`src/models/evaluator.py`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/src/models/evaluator.py): Computes multiclass accuracy, log loss, confusion matrices, and ROC curves.
- [`src/models/metrics.py`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/src/models/metrics.py): Centralized metrics module.
- [`src/models/cross_validation.py`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/src/models/cross_validation.py): Implements time-aware validation splits.
- [`src/models/calibration.py`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/src/models/calibration.py): Computes multiclass calibration curves and calibration recommendations.
- [`src/models/model_registry.py`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/src/models/model_registry.py): Centralized registry tracking all experiments.
- [`src/models/model_loader.py`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/src/models/model_loader.py): Standard loader for model pickle files.
- [`src/models/prediction_interface.py`](file:///Users/swayampanchal/Desktop/FIFA%20World%20Cup%20Prediction%20Project/src/models/prediction_interface.py): Production prediction interface wrapper.

---

## 2. Preprocessing & Data Preparation

Data scaling and imputations are managed via sklearn `Pipeline` and `ColumnTransformer` inside `src/models/trainer.py`:
- **Scaled Preprocessing**: Fills numeric NaNs with median, scales numeric columns using `StandardScaler`, and one-hot encodes categorical values. Used for Logistic Regression, SVM, and KNN.
- **Tree Preprocessing**: Fills NaNs and one-hot encodes categoricals. Does *not* scale values, maintaining original distributions for tree models (e.g. Random Forest, Gradient Boosting, XGBoost).
