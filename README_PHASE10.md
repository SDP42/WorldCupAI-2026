# 🏆 WorldCupAI — Phase 10: XAI & Decision Intelligence

Welcome to Phase 10 of the **WorldCupAI** project. This phase transforms the prediction pipeline into a transparent, audit-ready machine learning framework.

## Project Structure

Our explainability suite lives in `src/explainability/` and consists of the following modules:
* `feature_importance.py`: Global feature importance extractor across ML models.
* `global_explanations.py`: Importance stability, correlation, and feature ranking.
* `local_explanations.py`: Attributions and narrative explanations for every single match.
* `confidence_analysis.py`: Prediction entropy, margin, calibration, and risk categorisation.
* `counterfactuals.py`: What-if perturbation scenario testing (Elo, Form, Attack, Defence).
* `ensemble_explanations.py`: Voting weight split and model contribution breakdown.
* `tournament_explanations.py`: Bracket narrative, upsets, and round statistics.
* `visualizations.py`: Production plots stored under `outputs/plots/`.
* `report_generator.py`: Generates the complete Markdown report suite.

## Generated Reports

The following detailed reports are generated at the root of the project:
1. `README_EXPLAINABILITY.md` - High-level user guide to our XAI architecture.
2. `XAI_REPORT.md` - Main dashboard covering all aspects of explainability.
3. `FEATURE_IMPORTANCE_REPORT.md` - In-depth global feature rankings and stability.
4. `CONFIDENCE_ANALYSIS.md` - Probability margins, entropy, and risk tiers.
5. `COUNTERFACTUAL_REPORT.md` - Robustness testing and decision boundary flips.
6. `ENSEMBLE_EXPLAINABILITY.md` - Base model contributions and agreement metrics.
7. `TOURNAMENT_EXPLANATION_REPORT.md` - Champion path tracing and tournament diagnostics.
8. `MODEL_TRUST_REPORT.md` - Verification details and production validation.
