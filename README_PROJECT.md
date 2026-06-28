# 🏆 WorldCupAI — FIFA World Cup Prediction Platform

> A production-grade AI system for predicting FIFA World Cup knockout match outcomes using Machine Learning, Deep Learning, Ensemble Learning, and Explainable AI.

---

## Project Mission

Build an end-to-end ML platform capable of predicting FIFA World Cup 2026 knockout-stage results — starting with the **Round of 32**, then cascading through the Round of 16, Quarter-finals, Semi-finals, and the Final.

The system will combine:

| Pillar | Purpose |
|--------|---------|
| **Machine Learning** | Gradient-boosted trees, logistic regression, random forests |
| **Deep Learning** | Sequence-aware neural architectures for team form modelling |
| **Ensemble Learning** | Stacking, blending, and weighted voting across diverse models |
| **Explainable AI (XAI)** | SHAP values, feature importance, and counterfactual analysis |
| **Interactive Dashboard** | Streamlit-based UI for live exploration and prediction |

---

## Phase Roadmap

| Phase | Name | Status |
|-------|------|--------|
| **1** | Foundation, Architecture & Dataset Intelligence | ✅ Complete |
| **2** | Data Engineering, Cleaning & Feature Store | 🔲 Pending |
| **3** | Feature Engineering & EDA | 🔲 Pending |
| **4** | Model Development & Training | 🔲 Pending |
| **5** | Ensemble Construction & Evaluation | 🔲 Pending |
| **6** | Explainability & Reporting | 🔲 Pending |
| **7** | Streamlit Dashboard & Deployment | 🔲 Pending |

---

## Data Sources

| Source | Type | Records | Coverage |
|--------|------|---------|----------|
| Fjelstul World Cup Database | World Cup historical (27 tables) | ~75K+ records | 1930–2022 |
| International Football Results | All international matches | 49,477 matches | 1872–2026 |
| Football Data from Transfermarkt | Club/player/valuations | 6M+ records | Modern era |
| FIFA World Cup Dataset | Pre-engineered team features | 240 teams × tournaments | 2006–2026 |
| ELO Ratings | Historical Elo strength | 4,683 team-years | 1901–2026 |
| FIFA Men's Rankings | Official FIFA rankings | 13,130 entries | 2024 snapshots |
| Player Aggregates | FIFA game attribute averages | 1,599 country-versions | Multi-year |
| Team Form | Rolling match performance | 102,094 entries | Full history |
| Team Match Features | Merged match-level features | 43,364 matches | Full history |

---

## Repository Structure

See [README_ARCHITECTURE.md](README_ARCHITECTURE.md) for the complete production directory layout.

---

## Key Documentation

| Document | Description |
|----------|-------------|
| [README_DATASETS.md](README_DATASETS.md) | Dataset catalogue with purpose, strengths, weaknesses |
| [README_DATA_AUDIT.md](README_DATA_AUDIT.md) | Full audit: rows, columns, types, missing values, duplicates |
| [README_ARCHITECTURE.md](README_ARCHITECTURE.md) | Repository structure and folder purposes |
| [README_PHASE1.md](README_PHASE1.md) | Phase 1 approval report and readiness assessment |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## Engineering Standards

- **No data leakage** — strict temporal validation; target columns isolated
- **Reproducibility** — seeded random states, versioned configs, deterministic pipelines
- **Modularity** — each pipeline stage is independently testable
- **Documentation** — every dataset, feature, and model decision is recorded
- **Production-grade** — logging, error handling, config-driven architecture

---

## License & Attribution

This project uses datasets under their respective licenses:
- Fjelstul World Cup Database: CC-BY-SA 4.0 (Joshua C. Fjelstul, Ph.D.)
- International Football Results: Public domain (Mart Jürisoo)
- Transfermarkt Data: Educational / research use
- FIFA Rankings, Elo Ratings: Public data

---

*WorldCupAI — Engineered for accuracy. Built for production.*
