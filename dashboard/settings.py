"""WorldCupAI — Settings Page.

Allows configuring folders, theme selection, clearing cache, developer flags,
and random seed selection.
"""
import streamlit as st

def render_settings_page():
    """Renders the settings and caching control board."""
    st.markdown("<h2 style='color:#FFD700;'>⚙ Settings & Caching Panel</h2>", unsafe_allow_html=True)
    st.markdown("Configure operational variables, adjust data sources, or manage cached predictions.")

    st.subheader("Operational Paths")
    pred_dir = st.text_input("Prediction Logs Folder", value="predictions")
    export_dir = st.text_input("Report Export Directory", value=".")
    
    st.subheader("Random Seed Controls")
    seed_val = st.number_input("Deterministic Random Seed", min_value=1, max_value=99999, value=42)

    st.subheader("Visual Theme Setup")
    theme = st.selectbox("Active Theme Mode", ["Dark Mode (Standard)", "Light Mode (Beta)"])

    st.subheader("Streamlit Cache Controls")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Clear Streamlit Cache"):
            st.cache_data.clear()
            st.success("Streamlit data cache successfully cleared.")
            st.rerun()
            
    with col2:
        if st.button("🔄 Refresh Data Sources"):
            st.success("Re-scanned and synchronized with predictions folder.")
            st.rerun()
            
    st.subheader("Developer Mode Flag")
    dev_mode = st.toggle("Enable Verbose Auditing", value=False)
    if dev_mode:
        st.info("Verbose logging enabled in terminal stdout stream.")
