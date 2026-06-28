"""WorldCupAI — Model Performance Visualizations.

Parses model registry metrics and builds comparison Plotly charts for accuracy,
Brier score, ECE, and execution times.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

def render_performance_page(registry_data: Dict[str, Any]):
    """Renders the Model Performance page with interactive radar/bar charts."""
    st.markdown("<h2 style='color:#FFD700;'>📊 Model Performance & Leaderboard</h2>", unsafe_allow_html=True)
    st.markdown("Compare validation metrics, calibration error, and training times across all candidate classifiers.")

    if not registry_data:
        st.warning("No model registry metadata found. Train base models first.")
        return

    # Process registry data into flat df
    rows = []
    for model_name, data in registry_data.items():
        metrics = data.get("metrics", {})
        if metrics:
            rows.append({
                "Model": model_name,
                "Accuracy": metrics.get("accuracy", 0.0),
                "F1 Score (Macro)": metrics.get("f1_macro", 0.0),
                "Log Loss": metrics.get("log_loss", 0.0),
                "Brier Score": metrics.get("brier_score", 0.0),
                "Calibration (ECE)": metrics.get("ece", 0.0),
                "Training Time (s)": metrics.get("training_time_sec", 0.0),
                "Inference Time (s)": metrics.get("prediction_time_sec", 0.0),
            })
            
    df = pd.DataFrame(rows)

    # Leaderboard Table
    st.subheader("Validation Leaderboard")
    sort_metric = st.selectbox("Sort Leaderboard By", ["Accuracy", "F1 Score (Macro)", "Log Loss", "Calibration (ECE)", "Training Time (s)"])
    ascending = (sort_metric in ["Log Loss", "Calibration (ECE)", "Training Time (s)"])
    df_sorted = df.sort_values(sort_metric, ascending=ascending)
    st.dataframe(df_sorted.style.highlight_max(subset=["Accuracy", "F1 Score (Macro)"], color='#1E90FF')
                          .highlight_min(subset=["Log Loss", "Calibration (ECE)"], color='#2ECC71'),
                 use_container_width=True)

    # Plotly Visuals
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### Accuracy & F1-Score Comparison")
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=df["Model"], y=df["Accuracy"], name="Accuracy", marker_color="#1E90FF"))
        fig1.add_trace(go.Bar(x=df["Model"], y=df["F1 Score (Macro)"], name="F1 Score", marker_color="#FFD700"))
        fig1.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#C9D1D9",
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        st.markdown("#### Calibration (Expected Calibration Error) Comparison")
        # Lower is better
        fig2 = px.bar(df, x="Model", y="Calibration (ECE)", color="Calibration (ECE)",
                      color_continuous_scale="RdYlGn_r", labels={"Calibration (ECE)": "ECE (Lower is Better)"})
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#C9D1D9",
            margin=dict(l=10, r=10, t=20, b=10),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Runtime trade-off bubble chart
    st.markdown("#### Training Time vs Accuracy Trade-off")
    fig3 = px.scatter(
        df,
        x="Training Time (s)",
        y="Accuracy",
        size="Inference Time (s)",
        color="Model",
        hover_name="Model",
        size_max=40,
        text="Model",
    )
    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#C9D1D9",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig3, use_container_width=True)
