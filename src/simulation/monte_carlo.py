"""WorldCupAI — Phase 9 Monte Carlo Bracket Simulation Engine."""
import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from src.prediction.knockout_engine import KnockoutEngine, map_team_name
from src.simulation.probability_sampler import ProbabilitySampler, PredictionCache

def run_tournament_simulation(engine: KnockoutEngine, sampler: ProbabilitySampler, rng: np.random.Generator) -> Dict[str, Any]:
    """Runs a single probabilistic knockout bracket simulation (Matches 73–104)."""
    # Clear mutable state on the engine to avoid cross-simulation leakage
    engine.team_last_match_date.clear()
    engine.matches_played.clear()
    
    match_results = {}
    
    # ── 1. Round of 32 (Matches 73–88) ────────────────────────────────────
    for fixture in engine.round_32_fixtures:
        m_no = fixture["match_no"]
        h = map_team_name(fixture["home_team"])
        a = map_team_name(fixture["away_team"])
        date = fixture.get("date", "2026-06-28")
        
        res = sampler.sample_match(h, a, m_no, date, rng)
        match_results[m_no] = res
        
        # Track last match dates in the engine instance for rest days lookup
        engine.team_last_match_date[h] = date
        engine.team_last_match_date[a] = date

    # ── 2. Round of 16 (Matches 89–96) ────────────────────────────────────
    r16_pairings = [
        (89, 73, 75, "2026-07-04"),
        (90, 74, 77, "2026-07-04"),
        (91, 76, 78, "2026-07-05"),
        (92, 79, 80, "2026-07-05"),
        (93, 83, 84, "2026-07-06"),
        (94, 81, 82, "2026-07-06"),
        (95, 86, 88, "2026-07-07"),
        (96, 85, 87, "2026-07-07"),
    ]
    for m_no, prev_a, prev_b, date in r16_pairings:
        w_a = match_results[prev_a]["winner"]
        w_b = match_results[prev_b]["winner"]
        res = sampler.sample_match(w_a, w_b, m_no, date, rng)
        match_results[m_no] = res
        engine.team_last_match_date[map_team_name(w_a)] = date
        engine.team_last_match_date[map_team_name(w_b)] = date

    # ── 3. Quarter-finals (Matches 97–100) ────────────────────────────────
    qf_pairings = [
        (97, 89, 90, "2026-07-09"),
        (98, 93, 94, "2026-07-10"),
        (99, 91, 92, "2026-07-10"),
        (100, 95, 96, "2026-07-11"),
    ]
    for m_no, prev_a, prev_b, date in qf_pairings:
        w_a = match_results[prev_a]["winner"]
        w_b = match_results[prev_b]["winner"]
        res = sampler.sample_match(w_a, w_b, m_no, date, rng)
        match_results[m_no] = res
        engine.team_last_match_date[map_team_name(w_a)] = date
        engine.team_last_match_date[map_team_name(w_b)] = date

    # ── 4. Semi-finals (Matches 101–102) ──────────────────────────────────
    sf_pairings = [
        (101, 97, 98, "2026-07-14"),
        (102, 99, 100, "2026-07-15"),
    ]
    for m_no, prev_a, prev_b, date in sf_pairings:
        w_a = match_results[prev_a]["winner"]
        w_b = match_results[prev_b]["winner"]
        res = sampler.sample_match(w_a, w_b, m_no, date, rng)
        match_results[m_no] = res
        engine.team_last_match_date[map_team_name(w_a)] = date
        engine.team_last_match_date[map_team_name(w_b)] = date

    # ── 5. Third Place Play-off (Match 103) ───────────────────────────────
    m101 = match_results[101]
    l_a = m101["home_team"] if m101["winner"] != m101["home_team"] else m101["away_team"]
    
    m102 = match_results[102]
    l_b = m102["home_team"] if m102["winner"] != m102["home_team"] else m102["away_team"]
    
    res_103 = sampler.sample_match(l_a, l_b, 103, "2026-07-18", rng)
    match_results[103] = res_103
    engine.team_last_match_date[map_team_name(l_a)] = "2026-07-18"
    engine.team_last_match_date[map_team_name(l_b)] = "2026-07-18"

    # ── 6. Final (Match 104) ──────────────────────────────────────────────
    w_a = match_results[101]["winner"]
    w_b = match_results[102]["winner"]
    res_104 = sampler.sample_match(w_a, w_b, 104, "2026-07-19", rng)
    match_results[104] = res_104
    engine.team_last_match_date[map_team_name(w_a)] = "2026-07-19"
    engine.team_last_match_date[map_team_name(w_b)] = "2026-07-19"

    return match_results

def analyze_simulation_run(match_results: Dict[int, Dict[str, Any]], teams_list: List[str]) -> Dict[str, Dict[str, Any]]:
    """Analyzes a single simulation and extracts performance info for each team."""
    team_stats = {t: {
        "reached_r16": 0, "reached_qf": 0, "reached_sf": 0, "reached_final": 0, "reached_champion": 0,
        "is_runner_up": 0, "is_third": 0, "is_fourth": 0, "matches_played": 0, "confidence_sum": 0.0,
        "entropy_sum": 0.0, "eliminated_round": "Champion", "eliminator": "None"
    } for t in teams_list}

    # Helper to track rounds reached
    # R32 is automatic for all teams in the bracket config
    
    # R16
    for m_no in range(89, 97):
        team_stats[match_results[m_no]["home_team"]]["reached_r16"] = 1
        team_stats[match_results[m_no]["away_team"]]["reached_r16"] = 1

    # QF
    for m_no in range(97, 101):
        team_stats[match_results[m_no]["home_team"]]["reached_qf"] = 1
        team_stats[match_results[m_no]["away_team"]]["reached_qf"] = 1

    # SF
    for m_no in range(101, 103):
        team_stats[match_results[m_no]["home_team"]]["reached_sf"] = 1
        team_stats[match_results[m_no]["away_team"]]["reached_sf"] = 1

    # Final
    m104 = match_results[104]
    team_stats[m104["home_team"]]["reached_final"] = 1
    team_stats[m104["away_team"]]["reached_final"] = 1
    
    # Champion / Runner-up
    champion = m104["winner"]
    runner_up = m104["home_team"] if m104["winner"] == m104["away_team"] else m104["away_team"]
    
    team_stats[champion]["reached_champion"] = 1
    team_stats[runner_up]["is_runner_up"] = 1

    # Third / Fourth Place
    m103 = match_results[103]
    third = m103["winner"]
    fourth = m103["home_team"] if m103["winner"] == m103["away_team"] else m103["away_team"]
    team_stats[third]["is_third"] = 1
    team_stats[fourth]["is_fourth"] = 1

    # Match counts, confidence, entropy, and elimination stats
    # Every team plays in the Round of 32
    for t in teams_list:
        # Trace team matches
        t_matches = [m for m in match_results.values() if m["home_team"] == t or m["away_team"] == t]
        team_stats[t]["matches_played"] = len(t_matches)
        
        conf_sum = 0.0
        ent_sum = 0.0
        for m in t_matches:
            conf_sum += m["confidence"]
            ent_sum += m["entropy"]
        team_stats[t]["confidence_sum"] = conf_sum
        team_stats[t]["entropy_sum"] = ent_sum

        # Find elimination match
        # The elimination match is the match where this team played but did NOT win
        elim_matches = [m for m in t_matches if m["winner"] != t]
        if elim_matches:
            # Sort by match number to get the final elimination
            elim_match = sorted(elim_matches, key=lambda x: x["match_no"])[-1]
            m_no = elim_match["match_no"]
            
            # Map match number to round name
            if m_no <= 88:
                r_name = "Round of 32"
            elif m_no <= 96:
                r_name = "Round of 16"
            elif m_no <= 100:
                r_name = "Quarter-final"
            elif m_no <= 102:
                r_name = "Semi-final"
            elif m_no == 103:
                # Lost 3rd Place match
                r_name = "Semi-final"  # Eliminating tournament round is Semi-final
            elif m_no == 104:
                r_name = "Final"
            else:
                r_name = "Knockout"
                
            team_stats[t]["eliminated_round"] = r_name
            team_stats[t]["eliminator"] = elim_match["winner"]

    return team_stats
