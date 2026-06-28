"""WorldCupAI — Reports Preview and Download Center.

Loads root markdown files and predictions directories to offer preview
tabs and download actions for researchers.
"""
import os
import streamlit as st

def render_reports_page():
    """Renders the reports page with download actions and markdown previews."""
    st.markdown("<h2 style='color:#FFD700;'>📑 Reports & Exports Center</h2>", unsafe_allow_html=True)
    st.markdown("Download data files, spreadsheets, and analytical reports directly from the prediction pipeline.")
    
    st.info("💡 Files are loaded live from the production storage (`predictions/` directory and project root).")

    # Categories
    tab1, tab2 = st.tabs(["📊 Data File Downloads", "📝 Markdown Reports Preview"])

    with tab1:
        st.subheader("Select and Download Data Streams")
        
        # Grid of download cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                """
                <div style="background: rgba(22, 29, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 15px; min-height: 180px;">
                    <h4 style="color:#FFD700; margin-top:0;">⚽ Match Predictions</h4>
                    <p style="color:#8892B0; font-size:12px;">Full predictions logs, probabilities and outcome for 32 tournament bracket matchups.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Fetch files
            if os.path.exists("predictions/tournament_predictions.csv"):
                with open("predictions/tournament_predictions.csv", "rb") as f:
                    st.download_button("Download CSV", f.read(), "tournament_predictions.csv", "text/csv")
            if os.path.exists("predictions/tournament_predictions.json"):
                with open("predictions/tournament_predictions.json", "rb") as f:
                    st.download_button("Download JSON", f.read(), "tournament_predictions.json", "application/json")

        with col2:
            st.markdown(
                """
                <div style="background: rgba(22, 29, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 15px; min-height: 180px;">
                    <h4 style="color:#FFD700; margin-top:0;">🧠 Explainable AI</h4>
                    <p style="color:#8892B0; font-size:12px;">Local attributions, global importances, and signed feature attribution matrix for the ensemble.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            if os.path.exists("predictions/match_explanations.csv"):
                with open("predictions/match_explanations.csv", "rb") as f:
                    st.download_button("Download XAI CSV", f.read(), "match_explanations.csv", "text/csv")
            if os.path.exists("predictions/global_explanations.json"):
                with open("predictions/global_explanations.json", "rb") as f:
                    st.download_button("Download XAI JSON", f.read(), "global_explanations.json", "application/json")

        with col3:
            st.markdown(
                """
                <div style="background: rgba(22, 29, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 15px; min-height: 180px;">
                    <h4 style="color:#FFD700; margin-top:0;">📊 Simulation Stats</h4>
                    <p style="color:#8892B0; font-size:12px;">Advancement, champion, and runner-up probability matrices from 1,000 Monte Carlo runs.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            if os.path.exists("predictions/team_statistics.csv"):
                with open("predictions/team_statistics.csv", "rb") as f:
                    st.download_button("Download Stats CSV", f.read(), "team_statistics.csv", "text/csv")
            if os.path.exists("predictions/simulation_results.parquet"):
                with open("predictions/simulation_results.parquet", "rb") as f:
                    st.download_button("Download Parquet", f.read(), "simulation_results.parquet", "application/octet-stream")

    with tab2:
        st.subheader("Interactive Document Reader")
        reports = {
            "XAI Dashboard Overview": "XAI_REPORT.md",
            "Confidence & Risk analysis": "CONFIDENCE_ANALYSIS.md",
            "What-If Counterfactual Report": "COUNTERFACTUAL_REPORT.md",
            "Global Feature Importance": "FEATURE_IMPORTANCE_REPORT.md",
            "Model Trust Report": "MODEL_TRUST_REPORT.md",
            "Ensemble Explainability": "ENSEMBLE_EXPLAINABILITY.md"
        }
        
        sel_report = st.selectbox("Select Report to Read", list(reports.keys()))
        report_file = reports[sel_report]
        
        if os.path.exists(report_file):
            with open(report_file, "r") as f:
                content = f.read()
            st.markdown(
                f"""
                <div style="background: rgba(22, 29, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 25px; max-height: 500px; overflow-y: auto; font-family: 'Inter', sans-serif;">
                    {content}
                </div>
                """,
                unsafe_allow_html=True
            )
            # Add MD download button
            st.download_button(f"Download {sel_report} (.md)", content, report_file, "text/markdown")
        else:
            st.warning(f"Report file `{report_file}` not found at project root. Make sure to run the XAI orchestrator first.")
