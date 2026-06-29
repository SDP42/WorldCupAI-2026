"""WorldCupAI — Streamlit Production Dashboard.

Entry point for the premium, dark-themed decision intelligence platform.
Coordinating sidebar routing, layout modules, and visualizations.
"""
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add current path to sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dashboard.data_loader import (
    load_predictions, load_explanations, load_confidence, load_confidence_summary,
    load_counterfactuals, load_counterfactual_examples, load_feature_importance,
    load_global_feature_importance_json, load_ensemble_summary, load_ensemble_explanations,
    load_tournament_explanations, load_simulation_stats, load_bracket, load_tree,
    load_model_registry, get_dashboard_summary_metrics
)
from dashboard.sidebar import render_sidebar
from dashboard.cards import render_kpi_grid, render_stat_card
from dashboard.bracket import render_bracket_page
from dashboard.reports import render_reports_page
from dashboard.export import render_export_section
from dashboard.xai import render_xai_page
from dashboard.performance import render_performance_page
from dashboard.settings import render_settings_page

# Configure Page Settings
st.set_page_config(
    page_title="WorldCupAI Dashboard",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Global CSS for Dark Mode & Styling
st.markdown(
    """
    <style>
    /* Main container styling */
    .stApp {
        background-color: #111827 !important;
        color: #C9D1D9 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header fonts */
    h1, h2, h3, h4, h5, h6 {
        color: #FFD700 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
    }
    
    /* Styled dividers */
    hr {
        border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Center aligning info boxes */
    .info-container {
        background: rgba(22, 29, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    # 1. Load sidebar and get selection routing
    selected_page = render_sidebar()

    # 2. Preload cached data streams
    predictions = load_predictions()
    explanations = load_explanations()
    confidence_df = load_confidence()
    confidence_sum = load_confidence_summary()
    counterfactual_df = load_counterfactuals()
    counterfactual_examples = load_counterfactual_examples()
    feature_imp_df = load_feature_importance()
    ensemble_sum = load_ensemble_summary()
    ensemble_df = load_ensemble_explanations()
    tourney_exps = load_tournament_explanations()
    simulation_df = load_simulation_stats()
    bracket_data = load_bracket()
    tree_data = load_tree()
    model_registry_data = load_model_registry()
    summary_metrics = get_dashboard_summary_metrics()

    # Add simulation champ probability to summary
    champ_prob = 18.5
    if not simulation_df.empty:
        champ_row = simulation_df[simulation_df["team"] == summary_metrics["champion"]]
        if not champ_row.empty:
            champ_prob = float(champ_row["champion_prob"].values[0]) * 100
    summary_metrics["champion_prob"] = champ_prob

    # 3. Router logic
    if selected_page == "🏠 Home":
        st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>🏆 WorldCupAI</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #8892B0 !important; font-weight: normal; margin-top: 0;'>AI-Powered FIFA World Cup 2026 Prediction Platform</h3>", unsafe_allow_html=True)
        st.markdown("<hr style='width: 30%; margin: auto; margin-bottom: 30px;'>", unsafe_allow_html=True)

        # KPI metric cards
        render_kpi_grid(summary_metrics)

        # Overview section
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div class="info-container">
                    <h3 style="margin-top:0; color:#1E90FF !important;">🧠 Neural Tournament Simulation</h3>
                    <p style="color:#8892B0; font-size:14px; line-height:1.6;">
                        WorldCupAI leverages a Unified Ensemble Pipeline that merges deep learning sequence processors (LSTM) with optimized, probability-calibrated tree models (XGBoost, Gradient Boosting).
                        Our engine simulates the entire official FIFA 2026 Knockout Bracket (from Round of 32 through to the Final) over 1,000 parallel Monte Carlo runs to determine statistical champions and probability distributions.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Display timeline completion stats
            st.subheader("Repository Technical Specifications")
            st.text("Framework Version: v1.0.0 (Production Build)")
            st.text("Active Classifiers: XGBoost, Gradient Boosting, RF, Extra Trees, LR, ANN, LSTM")
            st.text("Warmed Matchup Cache: 296 Unique Matchup Configurations")

        with col2:
            st.markdown("### Production Ensemble Configuration")
            if ensemble_sum:
                weights = ensemble_sum.get("ensemble_weights", {})
                active_weights = {k: v for k, v in weights.items() if v > 0}
                
                # Pie Chart
                fig = px.pie(
                    names=list(active_weights.keys()),
                    values=list(active_weights.values()),
                    title="Ensemble Voter Weights",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color="#C9D1D9",
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)

    elif selected_page == "🏆 Tournament Overview":
        st.markdown("<h2 style='color:#FFD700;'>🏆 Tournament Diagnostic Overview</h2>", unsafe_allow_html=True)
        st.markdown("Trace key tournament predictions, upset margins, and champion probability distributions.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            render_stat_card("🥇 World Cup Champion", summary_metrics["champion"], f"Probability: {summary_metrics['champion_prob']:.1f}%", "#FFD700")
        with col2:
            render_stat_card("🥈 Runner-Up Runner", summary_metrics["runner_up"], "Predicted finalist outcome", "#C9D1D9")
        with col3:
            render_stat_card("🥉 Third Place Winner", summary_metrics["third_place"], f"Fourth: {summary_metrics['fourth_place']}", "#CD7F32")

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("### Top Champion Probabilities (Monte Carlo)")
            if not simulation_df.empty:
                top_champs = simulation_df.sort_values("champion_prob", ascending=False).head(10)
                fig = px.bar(
                    top_champs,
                    x="champion_prob",
                    y="team",
                    orientation="h",
                    labels={"champion_prob": "Champion Probability", "team": "Team"},
                    color="champion_prob",
                    color_continuous_scale="Reds"
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color="#C9D1D9",
                    margin=dict(l=10, r=10, t=10, b=10),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run Monte Carlo simulations to see probabilities.")

        with col_right:
            st.markdown("### Predicted Tournament Upsets")
            if tourney_exps:
                upsets = tourney_exps.get("biggest_upsets", [])
                if upsets:
                    upset_df = pd.DataFrame(upsets)
                    st.dataframe(upset_df, use_container_width=True)
                else:
                    st.info("No significant upset predictions in bracket.")

    elif selected_page == "🌳 Interactive Bracket":
        render_bracket_page(bracket_data)

    elif selected_page == "⚽ Match Predictions":
        st.markdown("<h2 style='color:#FFD700;'>⚽ Match Predictions Log</h2>", unsafe_allow_html=True)
        st.markdown("Detailed searchable predictions ledger for the 32 matches in the tournament bracket.")
        st.info("🌐 **Neutral Venue Notice:** All matches are played at neutral venues (no home-ground advantage). Teams are labeled as Team 1 and Team 2 based on bracket positions.")
        
        if predictions:
            df_preds = pd.DataFrame(predictions)
            search_query = st.text_input("🔍 Search Teams (e.g. Argentina, Germany)", "")
            
            if search_query:
                mask = df_preds["home_team"].str.contains(search_query, case=False) | df_preds["away_team"].str.contains(search_query, case=False)
                df_filtered = df_preds[mask]
            else:
                df_filtered = df_preds
                
            # Formatting cols with neutral labels
            df_display = df_filtered.copy()
            df_display = df_display.rename(columns={
                "home_team": "Team 1",
                "away_team": "Team 2",
                "predicted_winner": "Predicted Winner",
                "confidence": "Confidence",
                "entropy": "Entropy",
                "shootout_played": "Shootout?",
            })
            
            # Map probabilities if they exist
            if "prob_team1_win" in df_display.columns:
                df_display["Team 1 Win %"] = (df_display["prob_team1_win"] * 100).round(1)
                df_display["Team 2 Win %"] = (df_display["prob_team2_win"] * 100).round(1)
            else:
                df_display["Team 1 Win %"] = (df_display["prob_home_win"] * 100).round(1)
                df_display["Team 2 Win %"] = (df_display["prob_away_win"] * 100).round(1)
                
            df_display["Draw %"] = (df_display["prob_draw"] * 100).round(1)
            
            cols_to_show = ["match_no", "round", "Team 1", "Team 2", "Team 1 Win %", "Draw %", "Team 2 Win %", "predicted_outcome", "Predicted Winner", "Confidence", "Shootout?"]
            st.dataframe(df_display[cols_to_show], use_container_width=True)
        else:
            st.warning("No prediction log found.")

    elif selected_page == "🤖 Ensemble Analysis":
        st.markdown("<h2 style='color:#FFD700;'>🤖 Unified Ensemble Diagnostics</h2>", unsafe_allow_html=True)
        st.markdown("Analyze soft voting weights, model consensus levels, and prediction probability distributions.")

        if ensemble_sum:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Ensemble Model Calibration Summary")
                st.write(f"Active Ensemble Method: **{ensemble_sum.get('ensemble_method', 'Weighted Soft Voting')}**")
                st.write(f"Average Base Model Agreement: **{ensemble_sum.get('model_agreement', {}).get('mean', 0.8)*100:.1f}%**")
                
                # Show weight table
                w_df = pd.DataFrame(list(ensemble_sum.get("ensemble_weights", {}).items()), columns=["Classifier", "Weight"])
                st.table(w_df.sort_values("Weight", ascending=False))
            with col2:
                # Add plot from outputs if exists
                if os.path.exists("outputs/plots/model_contribution_chart.png"):
                    st.image("outputs/plots/model_contribution_chart.png", caption="Model contributions per match", use_column_width=True)
        else:
            st.info("No ensemble summary stats available.")

    elif selected_page == "📈 Explainable AI":
        render_xai_page(feature_imp_df, explanations, confidence_df)

    elif selected_page == "📊 Model Performance":
        render_performance_page(model_registry_data)

    elif selected_page == "📉 Confidence Analytics":
        st.markdown("<h2 style='color:#FFD700;'>📉 Confidence Analytics Dashboard</h2>", unsafe_allow_html=True)
        st.markdown("Study prediction entropy and calibration distribution.")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if os.path.exists("outputs/plots/confidence_distribution.png"):
                st.image("outputs/plots/confidence_distribution.png", caption="Confidence Distribution", use_column_width=True)
        with col_c2:
            if os.path.exists("outputs/plots/prediction_margin_histogram.png"):
                st.image("outputs/plots/prediction_margin_histogram.png", caption="Prediction Margin Histogram", use_column_width=True)

    elif selected_page == "🔬 Counterfactual Analysis":
        st.markdown("<h2 style='color:#FFD700;'>🔬 What-If Perturbation Analysis</h2>", unsafe_allow_html=True)
        st.markdown("Analyze how the model decisions would change if team ratings (ELO, rank, recent form) were perturbed.")
        st.info("🌐 **Neutral Venue Notice:** Matches are played at neutral venues. Team 1 / Team 2 indicate bracket order.")
        
        if not counterfactual_df.empty:
            # Let user select scenario
            scenarios_list = counterfactual_df["scenario"].unique()
            selected_scenario = st.selectbox("Select Scenario to Filter", scenarios_list)
            
            filtered_cf = counterfactual_df[counterfactual_df["scenario"] == selected_scenario].copy()
            filtered_cf = filtered_cf.rename(columns={"home_team": "Team 1", "away_team": "Team 2"})
            st.dataframe(filtered_cf[["match_no", "round", "Team 1", "Team 2", "original_winner", "new_winner", "confidence_delta", "decision_flip"]],
                         use_container_width=True)

            if os.path.exists("outputs/plots/counterfactual_comparison.png"):
                st.image("outputs/plots/counterfactual_comparison.png", caption="Counterfactual comparison", use_column_width=True)
        else:
            st.info("No counterfactual analysis logs found.")

    elif selected_page == "⭐ Feature Importance":
        st.markdown("<h2 style='color:#FFD700;'>⭐ Feature Importance Rankings</h2>", unsafe_allow_html=True)
        st.markdown("Detailed list of features and categories sorted by contribution to final predictions.")
        
        if not feature_imp_df.empty:
            ens_df = feature_imp_df[feature_imp_df["model"] == "Ensemble (Weighted)"].copy()
            if ens_df.empty:
                ens_df = feature_imp_df[feature_imp_df["model"] == "Average"].copy()
            
            search_feat = st.text_input("🔍 Search Feature Name (e.g. Elo, Rank, form)", "")
            if search_feat:
                ens_df = ens_df[ens_df["label"].str.contains(search_feat, case=False)]
                
            render_export_section(ens_df.sort_values("importance", ascending=False)[["feature", "label", "category", "importance", "importance_pct"]], "feature_importance_export.csv")
        else:
            st.warning("No feature importances found.")

    elif selected_page == "📑 Reports":
        render_reports_page()

    elif selected_page == "⚙ Settings":
        render_settings_page()

    elif selected_page == "ℹ About":
        st.markdown("<h2 style='color:#FFD700;'>ℹ About WorldCupAI</h2>", unsafe_allow_html=True)
        
        st.markdown(
            """
            <div class="info-container">
                <h3>🏛 Architecture Overview</h3>
                <p style="color:#8892B0; font-size:14px;">
                    WorldCupAI uses a layered model architecture. Raw results and form datasets are harmonized, chronological matchups are engineered, and features are parsed. Base models (XGBoost, Gradient Boosting, RF, ET, LR, Neural Networks ANN/LSTM) are calibrated, and prediction boundaries are optimized via constrained multi-objective optimizations.
                </p>
                <h4>🚀 Technology Stack</h4>
                <ul>
                    <li>Streamlit (Production Dashboard Layout)</li>
                    <li>Plotly & Seaborn (Analytical Charts)</li>
                    <li>Scikit-Learn & XGBoost (Machine Learning Engine)</li>
                    <li>PyTorch (Deep Learning Core)</li>
                    <li>Pandas, NumPy, Parquet (Data ETL Engineering)</li>
                </ul>
                <h4>Completed Technical Phases Timeline</h4>
                <ul>
                    <li>Phase 1-2: Datasets & Foundation Setup</li>
                    <li>Phase 3-4: Feature Store & Leakage Auditing</li>
                    <li>Phase 5: Probability Calibration & ML Leaderboard</li>
                    <li>Phase 6: PyTorch Deep Learning ANN & LSTM Models</li>
                    <li>Phase 7: Hybrid Ensemble & Voting Pipeline</li>
                    <li>Phase 7.1: Multi-objective Weight Optimization</li>
                    <li>Phase 8: Subprocess-Isolated Knockout Prediction Engine</li>
                    <li>Phase 9: Parallel Monte Carlo Simulator</li>
                    <li>Phase 10: XAI explainability & attributions</li>
                    <li>Phase 11: Production Dashboard View</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
