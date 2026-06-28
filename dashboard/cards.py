"""WorldCupAI — Glassmorphism Cards Component.

Provides a set of reusable metric/KPI card displays using HTML/CSS for
premium royal blue and gold glassmorphic dashboard looks.
"""
import streamlit as st
from typing import Dict, Any

def render_kpi_grid(metrics: Dict[str, Any]):
    """Renders the top home KPI cards grid with animations and glassmorphism."""
    st.markdown(
        f"""
        <div style="
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        ">
            <!-- Champion Card -->
            <div class="kpi-card" style="border-left: 4px solid #FFD700;">
                <div class="kpi-label">👑 Champion</div>
                <div class="kpi-value" style="color: #FFD700;">{metrics['champion']}</div>
                <div class="kpi-sub">Win probability: {metrics.get('champion_prob', 18.5):.1f}%</div>
            </div>
            <!-- Runner-Up Card -->
            <div class="kpi-card" style="border-left: 4px solid #1E90FF;">
                <div class="kpi-label">🥈 Runner-Up</div>
                <div class="kpi-value" style="color: #C9D1D9;">{metrics['runner_up']}</div>
                <div class="kpi-sub">Predicted finalist</div>
            </div>
            <!-- Confidence Card -->
            <div class="kpi-card" style="border-left: 4px solid #00D2FF;">
                <div class="kpi-label">🎯 Avg Confidence</div>
                <div class="kpi-value" style="color: #00D2FF;">{metrics['mean_confidence']*100:.1f}%</div>
                <div class="kpi-sub">Entropy: {metrics['mean_entropy']:.3f}</div>
            </div>
            <!-- Accuracy Card -->
            <div class="kpi-card" style="border-left: 4px solid #2ECC71;">
                <div class="kpi-label">📊 Model Accuracy</div>
                <div class="kpi-value" style="color: #2ECC71;">{metrics['model_accuracy']*100:.1f}%</div>
                <div class="kpi-sub">ROC-AUC: {metrics['model_roc_auc']:.4f}</div>
            </div>
        </div>
        
        <style>
        .kpi-card {{
            background: rgba(22, 29, 42, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 20px;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .kpi-card:hover {{
            transform: translateY(-5px);
            border-color: rgba(30, 144, 255, 0.3);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2), 0 0 15px rgba(30, 144, 255, 0.1);
        }}
        .kpi-label {{
            font-size: 13px;
            color: #8892B0;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 8px;
            font-weight: 600;
        }}
        .kpi-value {{
            font-size: 28px;
            font-weight: 800;
            margin-bottom: 4px;
            font-family: 'Outfit', sans-serif;
        }}
        .kpi-sub {{
            font-size: 11px;
            color: #8892B0;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def render_stat_card(title: str, value: str, subtext: str, color: str = "#1E90FF"):
    """Renders a single high-quality glassmorphic stat card."""
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left: 4px solid {color}; margin-bottom: 15px;">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value" style="color: {color}; font-size: 22px;">{value}</div>
            <div class="kpi-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
