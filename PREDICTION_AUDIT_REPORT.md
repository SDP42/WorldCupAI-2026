# 🔍 WorldCupAI — Prediction Audit Report

**Generated**: 2026-06-29 19:04:27

---

## ✅ Audit Summary

| Step | Status | Details |
|---|---|---|
| Feature Validation | ✅ PASS | All 37 features validated, `is_neutral=1` for all matches |
| Neutral Venue Fix | ✅ PASS | `is_neutral` hardcoded to `1` across all FIFA WC 2026 predictions |
| Symmetric Prediction | ✅ PASS | Both (A vs B) and (B vs A) averaged to eliminate venue bias |
| Home/Away Bias Removal | ✅ PASS | `home_advantage_score` set to `0` for neutral venue context |
| Ensemble Probabilities | ✅ PASS | All probabilities sum to 1.0 |
| Simulation Results | ✅ PASS | Monte Carlo stats loaded from existing run |

## 🏆 Tournament Predictions

- **Predicted Champion**: France
- **Predicted Runner-Up**: Argentina
- **Total Matches Predicted**: 32

## 📊 Match-Level Results

| Match # | Round | Team 1 | Team 2 | Predicted Winner | Confidence |
|---|---|---|---|---|---|
| 73 | Round of 32 | South Africa | Canada | Canada | 43.2% |
| 74 | Round of 32 | Germany | Paraguay | Germany | 45.2% |
| 75 | Round of 32 | Netherlands | Morocco | Netherlands | 38.2% |
| 76 | Round of 32 | Brazil | Japan | Brazil | 38.9% |
| 77 | Round of 32 | France | Sweden | France | 47.9% |
| 78 | Round of 32 | Ivory Coast | Norway | Norway | 42.4% |
| 79 | Round of 32 | Mexico | Ecuador | Mexico | 34.8% |
| 80 | Round of 32 | England | DR Congo | England | 60.2% |
| 81 | Round of 32 | United States | Bosnia & Herzegovina | United States | 49.6% |
| 82 | Round of 32 | Belgium | Senegal | Senegal | 43.7% |
| 83 | Round of 32 | Portugal | Croatia | Portugal | 36.0% |
| 84 | Round of 32 | Spain | Austria | Spain | 44.3% |
| 85 | Round of 32 | Switzerland | Algeria | Switzerland | 44.0% |
| 86 | Round of 32 | Argentina | Cape Verde | Argentina | 75.8% |
| 87 | Round of 32 | Colombia | Ghana | Colombia | 59.9% |
| 88 | Round of 32 | Australia | Egypt | Australia | 38.1% |
| 89 | Round of 16 | Canada | Netherlands | Netherlands | 46.9% |
| 90 | Round of 16 | Germany | France | France | 36.9% |
| 91 | Round of 16 | Brazil | Norway | Brazil | 40.7% |
| 92 | Round of 16 | Mexico | England | England | 42.5% |
| 93 | Round of 16 | Portugal | Spain | Spain | 41.3% |
| 94 | Round of 16 | United States | Senegal | United States | 37.8% |
| 95 | Round of 16 | Argentina | Australia | Argentina | 43.5% |
| 96 | Round of 16 | Switzerland | Colombia | Colombia | 41.3% |
| 97 | Quarter-final | Netherlands | France | France | 36.6% |
| 98 | Quarter-final | Spain | United States | Spain | 48.5% |
| 99 | Quarter-final | Brazil | England | Brazil | 38.9% |
| 100 | Quarter-final | Argentina | Colombia | Argentina | 34.0% |
| 101 | Semi-final | France | Spain | France | 38.7% |
| 102 | Semi-final | Brazil | Argentina | Argentina | 39.1% |
| 103 | Third Place Play-off | Spain | Brazil | Spain | 38.5% |
| 104 | Final | France | Argentina | France | 35.2% |

## ⚽ Neutral Venue Verification

All matches verified with `is_neutral=1`. Home/away bias completely removed.

## 📈 Simulation Statistics


---
*Report auto-generated from existing prediction outputs. Neutral venue flag verified.*
