# 📊 WorldCupAI — Calibration Report

**Generated**: 2026-06-28 21:05:03

---

## Step 3: Feature Scaling Verification

The preprocessing pipeline uses a fitted `ColumnTransformer` + `StandardScaler` applied with `.transform()` only during inference. No `fit_transform()` is called on knockout match data — this is enforced by `ml_predict_subprocess.py` which calls `pipeline.transform(X_test)` with the training-fitted pipeline.

## Step 7: Probability Calibration Analysis

The ensemble uses Platt-scaled calibration via `CalibratedClassifierCV` on each base model (fitted during Phase 5). No additional calibration is applied at the ensemble level. The post-fix probability ranges are:

| Match | Home Win% | Draw% | Away Win% | Predicted Winner |
|---|---|---|---|---|
| M73: South Africa vs Canada | 52.2% | 38.2% | 9.6% | **South Africa** |
| M74: Germany vs Paraguay | 75.1% | 22.1% | 2.7% | **Germany** |
| M75: Netherlands vs Morocco | 68.3% | 24.9% | 6.8% | **Netherlands** |
| M76: Brazil vs Japan | 67.8% | 26.4% | 5.7% | **Brazil** |
| M77: France vs Sweden | 80.1% | 17.2% | 2.7% | **France** |
| M78: Ivory Coast vs Norway | 65.0% | 26.3% | 8.7% | **Ivory Coast** |
| M79: Mexico vs Ecuador | 59.6% | 31.6% | 8.8% | **Mexico** |
| M80: England vs DR Congo | 87.7% | 10.3% | 2.0% | **England** |
| M81: United States vs Bosnia & Herzegovina | 83.9% | 13.7% | 2.4% | **United States** |
| M82: Belgium vs Senegal | 44.2% | 22.2% | 33.5% | **Belgium** |
| M83: Portugal vs Croatia | 61.8% | 30.5% | 7.8% | **Portugal** |
| M84: Spain vs Austria | 78.9% | 17.1% | 4.1% | **Spain** |
| M85: Switzerland vs Algeria | 80.0% | 16.5% | 3.5% | **Switzerland** |
| M86: Argentina vs Cape Verde | 85.9% | 11.3% | 2.8% | **Argentina** |
| M87: Colombia vs Ghana | 89.0% | 9.0% | 2.0% | **Colombia** |
| M88: Australia vs Egypt | 67.1% | 26.8% | 6.1% | **Australia** |
| M89: South Africa vs Netherlands | 25.7% | 42.5% | 31.8% | **Netherlands** |
| M90: Germany vs France | 55.5% | 35.3% | 9.2% | **Germany** |
| M91: Brazil vs Ivory Coast | 82.8% | 14.2% | 3.0% | **Brazil** |
| M92: Mexico vs England | 44.6% | 46.2% | 9.2% | **England** |
| M93: Portugal vs Spain | 59.6% | 31.9% | 8.5% | **Portugal** |
| M94: United States vs Belgium | 51.4% | 38.0% | 10.6% | **United States** |
| M95: Argentina vs Australia | 75.3% | 22.0% | 2.7% | **Argentina** |
| M96: Switzerland vs Colombia | 65.9% | 26.4% | 7.7% | **Switzerland** |
| M97: Netherlands vs Germany | 69.0% | 23.6% | 7.4% | **Netherlands** |
| M98: Portugal vs United States | 67.2% | 28.1% | 4.7% | **Portugal** |
| M99: Brazil vs England | 69.6% | 24.0% | 6.4% | **Brazil** |
| M100: Argentina vs Switzerland | 69.1% | 25.9% | 5.0% | **Argentina** |
| M101: Netherlands vs Portugal | 66.7% | 25.5% | 7.8% | **Netherlands** |
| M102: Brazil vs Argentina | 53.1% | 36.5% | 10.4% | **Brazil** |
| M103: Portugal vs Argentina | 57.9% | 33.1% | 9.1% | **Portugal** |
| M104: Netherlands vs Brazil | 66.8% | 26.7% | 6.4% | **Netherlands** |
