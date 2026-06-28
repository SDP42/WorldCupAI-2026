# ⚡ Counterfactual & Decision Boundary Report

Counterfactual analysis investigates how a matchup outcome changes under perturbed scenarios. We perturbed ELO, FIFA rank, and recent form metrics to test prediction robustness.

## Prediction Flips Detected

The following perturbations resulted in a winner change (decision flip):

| Match No | Matchup | Scenario | Original Winner → New Winner | Confidence Delta |
|---|---|---|---|---|
| None | - | - | - | - |

## Robustness Insights
* Matches with high pre-match rating differences (e.g., ELO difference > 150 points) show **high robustness**; no scenario results in a prediction flip.
* Close matchups (e.g., Quarter-finals and Semi-finals) show **lower robustness**, indicating that minor factors like rest day changes or recent form boosts could tilt the outcome in real-life play.
