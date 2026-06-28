"""WorldCupAI — Interactive Bracket Visualizer.

Loads bracket nodes directly from bracket.json / tournament_tree.json
and renders a premium, scrollable tournament tree diagram.
"""
import streamlit as st
from typing import Dict, Any, List

def render_bracket_page(bracket_data: Dict[str, Any]):
    """Renders the tournament bracket view with CSS tree grid layouts."""
    st.markdown("<h2 style='color:#FFD700;'>🌳 Interactive Knockout Bracket</h2>", unsafe_allow_html=True)
    st.markdown("Explore the predicted progression of teams from the Round of 32 down to the Final.")
    
    # Check if data loaded successfully
    if not bracket_data:
        st.warning("No bracket data found. Run predictions first.")
        return

    # Let's extract rounds
    r32 = bracket_data.get("round_32", [])
    r16 = bracket_data.get("round_16", [])
    qf = bracket_data.get("quarter_finals", [])
    sf = bracket_data.get("semi_finals", [])
    final = bracket_data.get("final", [])
    
    # Trace champion
    champ = "Argentina"
    if final:
        champ = final[0].get("predicted_winner", "Argentina")

    # Render a responsive horizontal flex bracket container
    st.markdown(
        f"""
        <style>
        .bracket-container {{
            display: flex;
            overflow-x: auto;
            gap: 20px;
            padding: 20px 10px;
            background: rgba(22, 29, 42, 0.4);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            min-height: 500px;
        }}
        .bracket-column {{
            display: flex;
            flex-direction: column;
            justify-content: space-around;
            min-width: 180px;
        }}
        .bracket-card {{
            background: rgba(22, 29, 42, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            font-size: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
            transition: all 0.2s ease;
        }}
        .bracket-card:hover {{
            border-color: #1E90FF;
            box-shadow: 0 0 10px rgba(30, 144, 255, 0.2);
            transform: scale(1.02);
        }}
        .bracket-team {{
            display: flex;
            justify-content: space-between;
            padding: 4px 2px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.02);
        }}
        .bracket-team.winner {{
            color: #FFD700;
            font-weight: bold;
        }}
        .round-title {{
            text-align: center;
            color: #8892B0;
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
            border-bottom: 2px solid #1E90FF;
            padding-bottom: 5px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Allow users to view round tabs or the full flex tree
    tab1, tab2 = st.tabs(["🌐 Full Tree Layout", "📋 Round Lists"])

    with tab1:
        # Full flex layout
        # Render a subset of rounds to keep it clean in widescreen
        st.markdown(
            f"""
            <div class="bracket-container">
                <!-- Round of 32 (Sample 4 matches to show tree flow) -->
                <div class="bracket-column">
                    <div class="round-title">Round of 32 (Selected)</div>
                    {"".join(_get_bracket_match_html(m) for m in r32[:8])}
                </div>
                <!-- Round of 16 -->
                <div class="bracket-column">
                    <div class="round-title">Round of 16</div>
                    {"".join(_get_bracket_match_html(m) for m in r16[:4])}
                </div>
                <!-- Quarter-finals -->
                <div class="bracket-column">
                    <div class="round-title">Quarter-finals</div>
                    {"".join(_get_bracket_match_html(m) for m in qf)}
                </div>
                <!-- Semi-finals -->
                <div class="bracket-column">
                    <div class="round-title">Semi-finals</div>
                    {"".join(_get_bracket_match_html(m) for m in sf)}
                </div>
                <!-- Final -->
                <div class="bracket-column">
                    <div class="round-title">Final</div>
                    {"".join(_get_bracket_match_html(m) for m in final)}
                    <div class="bracket-card" style="border: 1px solid #FFD700; background: rgba(255, 215, 0, 0.05); text-align: center;">
                        <span style="color:#FFD700; font-weight:bold; font-size:14px;">🏆 CHAMPION</span>
                        <div style="font-size:16px; font-weight:800; margin-top:5px;">{champ}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab2:
        # Detailed tabs
        rtabs = st.tabs(["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Finals"])
        
        rounds_list = [r32, r16, qf, sf, final]
        for idx, tab_obj in enumerate(rtabs):
            with tab_obj:
                round_matches = rounds_list[idx]
                col_left, col_right = st.columns(2)
                for i, m in enumerate(round_matches):
                    target_col = col_left if i % 2 == 0 else col_right
                    with target_col:
                        is_draw = (m.get("predicted_outcome") == "Draw")
                        outcome_text = f"Outcome: {m.get('predicted_outcome')} ({m.get('confidence')*100:.1f}% confidence)"
                        if is_draw:
                            outcome_text += f" - {m.get('shootout_reason', 'Shootout win')}"
                        
                        st.markdown(
                            f"""
                            <div style="background: rgba(22, 29, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                                <div style="display:flex; justify-content:space-between; font-size:11px; color:#8892B0;">
                                    <span>Match #{m.get('match_no')}</span>
                                    <span>{m.get('date')} - {m.get('city')}</span>
                                </div>
                                <div style="font-size:15px; font-weight:bold; margin: 10px 0;">
                                    <span style="color:{'#FFD700' if m.get('predicted_winner') == m.get('home_team') else '#C9D1D9'}">{m.get('home_team')}</span>
                                    <span style="color:#8892B0;"> vs </span>
                                    <span style="color:{'#FFD700' if m.get('predicted_winner') == m.get('away_team') else '#C9D1D9'}">{m.get('away_team')}</span>
                                </div>
                                <div style="font-size:12px; color:#1E90FF;">{outcome_text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

def _get_bracket_match_html(match: Dict[str, Any]) -> str:
    """Helper to compile match card HTML."""
    h = match.get("home_team", "Home")
    a = match.get("away_team", "Away")
    w = match.get("predicted_winner", "")
    
    h_class = "winner" if w == h else ""
    a_class = "winner" if w == a else ""
    
    return f"""
    <div class="bracket-card">
        <div class="bracket-team {h_class}">
            <span>{h}</span>
            <span>{match.get('prob_home_win', 0.5)*100:.0f}%</span>
        </div>
        <div style="text-align: center; color: rgba(255, 255, 255, 0.15); font-size: 10px; margin: 2px 0;">VS</div>
        <div class="bracket-team {a_class}">
            <span>{a}</span>
            <span>{match.get('prob_away_win', 0.5)*100:.0f}%</span>
        </div>
    </div>
    """
