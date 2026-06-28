"""WorldCupAI — Sidebar Navigation Component.

Renders a premium sidebar with Royal Blue and Gold theme styling, and provides
the navigation route mapping.
"""
import streamlit as st

def render_sidebar() -> str:
    """Renders the sidebar and returns the selected page."""
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding-bottom: 20px;">
            <h1 style="color: #FFD700; font-family: 'Outfit', sans-serif; font-size: 26px; font-weight: 800; margin-bottom: 5px;">🏆 WorldCupAI</h1>
            <p style="color: #8892B0; font-size: 12px; font-weight: 500; letter-spacing: 1px; text-transform: uppercase; margin-top: 0;">FIFA 2026 Prediction Engine</p>
            <hr style="border-top: 1px solid rgba(255, 255, 255, 0.1); margin: 15px 0;">
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown(
        """
        <style>
        /* Custom sidebar menu styling */
        div[data-testid="stSidebarNav"] {
            display: none;
        }
        .stRadio > label {
            display: none;
        }
        div.row-widget.stRadio > div {
            background-color: transparent !important;
            padding: 0px !important;
        }
        div.row-widget.stRadio > div > kbd {
            display: none;
        }
        /* Style radio buttons as premium menu items */
        div[data-testid="stSidebar"] div.stRadio label {
            background-color: rgba(255, 255, 255, 0.03) !important;
            color: #C9D1D9 !important;
            border-radius: 8px !important;
            padding: 10px 15px !important;
            margin-bottom: 6px !important;
            display: block !important;
            width: 100% !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
        }
        div[data-testid="stSidebar"] div.stRadio label:hover {
            background-color: rgba(30, 144, 255, 0.1) !important;
            border: 1px solid rgba(30, 144, 255, 0.3) !important;
            color: #1E90FF !important;
            transform: translateX(4px);
        }
        div[data-testid="stSidebar"] div.stRadio label[data-checked="true"] {
            background-color: rgba(30, 144, 255, 0.2) !important;
            border: 1px solid #1E90FF !important;
            color: #FFD700 !important;
            font-weight: bold !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    pages = [
        "🏠 Home",
        "🏆 Tournament Overview",
        "🌳 Interactive Bracket",
        "⚽ Match Predictions",
        "🤖 Ensemble Analysis",
        "📈 Explainable AI",
        "📊 Model Performance",
        "📉 Confidence Analytics",
        "🔬 Counterfactual Analysis",
        "⭐ Feature Importance",
        "📑 Reports",
        "⚙ Settings",
        "ℹ About"
    ]
    
    # We will use selectbox or radio. Radio with custom CSS looks like a real navbar
    selected_page = st.sidebar.radio(
        "Navigation",
        options=pages,
        index=0,
        key="sidebar_navigation"
    )

    st.sidebar.markdown(
        """
        <div style="position: absolute; bottom: 20px; left: 20px; right: 20px; text-align: center;">
            <p style="color: #8892B0; font-size: 11px; margin: 0;">Production Build v1.0.0</p>
            <p style="color: #FFD700; font-size: 10px; font-weight: bold; margin: 0;">Royal Blue & Gold Theme</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    return selected_page
