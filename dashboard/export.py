"""WorldCupAI — Data Export Utility.

Provides data transformation and custom DataFrame compilation for Streamlit.
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any, List

def render_export_section(df: pd.DataFrame, default_filename: str = "custom_export.csv"):
    """Helper component to render a data table and offer CSV download."""
    if df.empty:
        st.warning("No data available to export.")
        return
        
    st.dataframe(df.head(20), use_container_width=True)
    
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Selection as CSV",
        data=csv_data,
        file_name=default_filename,
        mime='text/csv',
    )
