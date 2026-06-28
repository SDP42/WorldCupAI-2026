# 📖 WorldCupAI — Dashboard User & Operational Guide

## Getting Started

1. **Prerequisites**: Confirm python dependencies (`streamlit`, `plotly`, `pyyaml`) are installed.
2. **Launch command**: `streamlit run app.py`
3. **Target Port**: Open `http://localhost:8501` in your browser.

## Key Operational Guidelines

### 1. Data Hot-Reloading
The dashboard does not reload models from disk (saving startup overhead). Instead, it loads compiled prediction records from `predictions/`. If you run new simulations or XAI updates in the pipeline, navigate to the **⚙ Settings** page and click **Refresh Data Sources** to sync the dashboard with new predictions instantly.

### 2. Explanations (XAI)
To debug a specific match prediction:
1. Select the **📈 Explainable AI** tab.
2. Click **Local Match Explanations**.
3. Select a match from the drop-down.
4. Review the horizontal bar chart showing positive (green) and negative (red) features driving the decision.

### 3. What-If Testing (Counterfactuals)
To understand ELO boundaries:
1. Navigate to **🔬 Counterfactual Analysis**.
2. Select a scenario (e.g. `Elo +50` or `Form -20%`).
3. Check the prediction changes table to inspect if a match decision flipped under the perturbation.
