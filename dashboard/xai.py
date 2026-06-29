"""WorldCupAI — Explainable AI (XAI) Views.

Renders interactive Plotly horizontal bar charts for global importance rankings
and signed local feature attributions per match matchup.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

def render_xai_page(
    global_importance_df: pd.DataFrame,
    match_explanations: List[Dict[str, Any]],
    confidence_df: pd.DataFrame,
):
    """Renders the XAI tab page with global/local explanation dashboards."""
    st.markdown("<h2 style='color:#FFD700;'>📈 Explainable AI (XAI) Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("Inspect global feature importance rankings and explore local attributions for each matchup.")

    # Tabs
    tab_global, tab_local = st.tabs(["🌍 Global Feature Rankings", "⚽ Local Match Explanations"])

    with tab_global:
        st.subheader("Global Feature Importance (Weighted Ensemble)")
        if not global_importance_df.empty:
            # Filter Ensemble Weighted
            ens_df = global_importance_df[global_importance_df["model"] == "Ensemble (Weighted)"].copy()
            if ens_df.empty:
                ens_df = global_importance_df[global_importance_df["model"] == "Average"].copy()
            
            ens_df = ens_df.sort_values("importance", ascending=True).tail(15)

            # Horizontal Plotly Chart
            fig = px.bar(
                ens_df,
                x="importance",
                y="label",
                orientation="h",
                labels={"importance": "Importance Score", "label": "Feature"},
                color="importance",
                color_continuous_scale="Blues",
                height=500,
            )
            # Custom styling
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#C9D1D9",
                margin=dict(l=10, r=10, t=20, b=20),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show category table
            st.markdown("### Importance by Feature Category")
            cat_df = global_importance_df.groupby(["model", "category"])["importance"].sum().reset_index()
            cat_ens = cat_df[cat_df["model"] == "Ensemble (Weighted)"]
            if cat_ens.empty:
                cat_ens = cat_df[cat_df["model"] == "Average"]
            st.table(cat_ens[["category", "importance"]].sort_values("importance", ascending=False))
        else:
            st.warning("No global feature importance metrics found.")

    with tab_local:
        st.subheader("Local Match Explanation Viewer")
        if match_explanations:
            match_labels = [f"Match {m['match_no']}: {m.get('team1', m['home_team'])} vs {m.get('team2', m['away_team'])} ({m['round']})" for m in match_explanations]
            selected_match_label = st.selectbox("Select Matchup to Explain", match_labels)
            
            # Get selected match
            idx = match_labels.index(selected_match_label)
            m_exp = match_explanations[idx]
            
            # Render layout
            col_info, col_chart = st.columns([1, 1.2])
            
            with col_info:
                st.markdown(
                    f"""
                    <div style="background: rgba(22, 29, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 20px; min-height: 380px;">
                        <h3 style="color:#FFD700; margin-top:0;">{m_exp.get('team1', m_exp['home_team'])} vs {m_exp.get('team2', m_exp['away_team'])}</h3>
                        <p style="color:#8892B0; font-size:12px;">{m_exp['round']} - Match No #{m_exp['match_no']}</p>
                        <hr style="border-top: 1px solid rgba(255,255,255,0.05); margin: 10px 0;">
                        <div style="margin-bottom:15px;">
                            <span style="font-size:12px; color:#8892B0; text-transform:uppercase;">Predicted Winner:</span><br>
                            <span style="font-size:20px; font-weight:bold; color:#FFD700;">{m_exp['predicted_winner']}</span>
                        </div>
                        <div style="margin-bottom:15px;">
                            <span style="font-size:12px; color:#8892B0; text-transform:uppercase;">Confidence Score:</span><br>
                            <span style="font-size:20px; font-weight:bold; color:#1E90FF;">{m_exp['confidence']*100:.1f}% ({m_exp['confidence_tier']})</span>
                        </div>
                        <div style="margin-bottom:15px;">
                            <span style="font-size:12px; color:#8892B0; text-transform:uppercase;">Uncertainty (Entropy):</span><br>
                            <span style="font-size:14px; font-weight:bold; color:#E74C3C;">{m_exp['entropy']:.4f}</span>
                        </div>
                        <div style="margin-bottom:10px;">
                            <span style="font-size:12px; color:#8892B0; text-transform:uppercase;">Decision Narrative:</span><br>
                            <span style="font-size:12px; color:#C9D1D9; font-style:italic;">"{m_exp['narrative']}"</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            with col_chart:
                st.markdown("#### Feature Contributions to Outcome")
                contribs = m_exp.get("all_feature_contributions", [])
                
                if contribs:
                    df_contrib = pd.DataFrame(contribs).head(10)
                    df_contrib = df_contrib.sort_values("contribution", ascending=True)
                    
                    # Color bar depending on direction
                    colors = ['#E74C3C' if c < 0 else '#2ECC71' for c in df_contrib['contribution']]
                    
                    fig = go.Figure(go.Bar(
                        x=df_contrib['contribution'],
                        y=df_contrib['label'],
                        orientation='h',
                        marker_color=colors,
                    ))
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color="#C9D1D9",
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Contribution Direction (Draw Baseline)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No detailed local contributions available.")
        else:
            st.warning("No local match explanations found.")
