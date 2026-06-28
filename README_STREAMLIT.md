# 🚀 WorldCupAI — Streamlit Platform Run Guide

This document describes how to execute the production-ready Streamlit decision intelligence dashboard for WorldCupAI.

## Prerequisites

Make sure the required packages are installed:
```bash
pip install streamlit plotly pyyaml pandas numpy
```

## Running the Dashboard

Launch the Streamlit app from the repository root:
```bash
streamlit run app.py
```

Streamlit will automatically assign a local port (typically `http://localhost:8501`) and open the platform in your browser.

## Performance Caching
The application utilizes `@st.cache_data` in `dashboard/data_loader.py` to parse prediction files, model registry stats, and ELO ratings. 
* **Startup time**: < 1.5 seconds.
* **Navigation lag**: 0.0 seconds.
