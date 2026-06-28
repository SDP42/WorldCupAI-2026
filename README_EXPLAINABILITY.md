# 🧠 WorldCupAI — Explainability & Decision Support Guide

This document serves as a guide for users and researchers to understand the explainability (XAI) engine integrated into the WorldCupAI model stack.

## Explanation Mechanics

### 1. Global Feature Importance
Global feature importances are extracted directly from the base classifiers inside the ensemble. 
* Tree-based estimators (`XGBoost`, `Gradient Boosting`, `Random Forest`, `Extra Trees`) leverage their built-in `feature_importances_` attributes.
* Linear models (`Logistic Regression`) leverage scaled absolute coefficients.
* The Ensemble feature importance is computed using a weighted average of individual importances, utilizing the optimized ensemble weights.

### 2. Local Match Explanations (Attribution)
Since true SHAP values can be slow to compute in real-time, the engine approximates feature contributions using **signed feature attributions**:
$$Attribution_i = Sign(Feature_i) \times Importance_i$$
Positive contributions support the predicted winner, while negative contributions point towards the runner-up or a draw outcome.

### 3. Confidence Metrics
We monitor:
* **Prediction Entropy**: Level of uncertainty ($0$ is perfect certainty, $1.585$ is total uncertainty).
* **Margin**: The difference between the highest class probability and the second-highest.
* **Agreement**: The percentage of ensemble models predicting the same hard label.

### 4. Counterfactual Perturbations
We perturb key match features (e.g. +50 ELO, -20% Form) and record the resulting probability delta to determine the model's decision boundaries.
