"""WorldCupAI — Phase 9 Statistical Analysis & Metrics Engine."""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from collections import Counter
from scipy.stats import sem

def get_finishing_position(team_run: Dict[str, Any]) -> int:
    """Maps team outcomes to numeric tournament finishing positions (lower is better)."""
    if team_run.get("reached_champion", 0) > 0:
        return 1
    if team_run.get("is_runner_up", 0) > 0:
        return 2
    if team_run.get("is_third", 0) > 0:
        return 3
    if team_run.get("is_fourth", 0) > 0:
        return 4
    if team_run.get("reached_qf", 0) == 0:
        if team_run.get("reached_r16", 0) == 0:
            return 32  # Lost in R32
        return 16  # Lost in R16
    return 8  # Lost in QF

def is_upset(match: Dict[str, Any]) -> bool:
    """Returns True if the match winner has a lower Elo rating than the loser."""
    # Find elo_diff from features dictionary
    features = match.get("features", {})
    elo_diff = features.get("elo_diff")
    
    if elo_diff is None:
        # Reconstruct elo_diff
        h_elo = features.get("home_elo", 1500.0)
        a_elo = features.get("away_elo", 1500.0)
        elo_diff = h_elo - a_elo
        
    winner = match["winner"]
    home = match["home_team"]
    away = match["away_team"]
    
    if elo_diff > 15.0 and winner == away:
        return True
    if elo_diff < -15.0 and winner == home:
        return True
    return False

def calculate_team_statistics(all_team_runs: List[Dict[str, Dict[str, Any]]], teams_list: List[str]) -> pd.DataFrame:
    """Computes comprehensive team-level simulation statistics across all iterations."""
    n_sims = len(all_team_runs)
    team_records = []

    for t in teams_list:
        champs = sum(run[t]["reached_champion"] for run in all_team_runs)
        runners = sum(run[t]["is_runner_up"] for run in all_team_runs)
        sfs = sum(run[t]["reached_sf"] for run in all_team_runs)
        qfs = sum(run[t]["reached_qf"] for run in all_team_runs)
        r16s = sum(run[t]["reached_r16"] for run in all_team_runs)
        
        matches = [run[t]["matches_played"] for run in all_team_runs]
        positions = [get_finishing_position(run[t]) for run in all_team_runs]
        
        confidences = []
        entropies = []
        for run in all_team_runs:
            matches_count = run[t]["matches_played"]
            if matches_count > 0:
                confidences.append(run[t]["confidence_sum"] / matches_count)
                entropies.append(run[t]["entropy_sum"] / matches_count)
            else:
                confidences.append(0.5)
                entropies.append(1.0)

        # Elimination stats
        elim_rounds = [run[t]["eliminated_round"] for run in all_team_runs if run[t]["eliminated_round"] != "Champion"]
        elim_opponents = [run[t]["eliminator"] for run in all_team_runs if run[t]["eliminator"] != "None"]
        
        common_elim_round = Counter(elim_rounds).most_common(1)[0][0] if elim_rounds else "None (Champion)"
        common_elim_opponent = Counter(elim_opponents).most_common(1)[0][0] if elim_opponents else "None (Champion)"

        team_records.append({
            "team": t,
            "champion_prob": champs / n_sims,
            "runner_up_prob": runners / n_sims,
            "semi_final_prob": sfs / n_sims,
            "quarter_final_prob": qfs / n_sims,
            "round_of_16_prob": r16s / n_sims,
            "avg_finishing_position": np.mean(positions),
            "avg_matches_played": np.mean(matches),
            "avg_confidence": np.mean(confidences),
            "avg_entropy": np.mean(entropies),
            "most_common_eliminating_round": common_elim_round,
            "most_common_eliminating_opponent": common_elim_opponent
        })

    return pd.DataFrame(team_records)

def calculate_tournament_statistics(all_match_runs: List[Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
    """Computes tournament-level indicators such as upset frequencies and common final pairings."""
    n_sims = len(all_match_runs)
    
    champions = []
    runners = []
    finalists_pairs = []
    semi_finalists = []
    quarter_finalists = []
    r16s = []
    
    upset_counts = []
    
    # Track prediction bounds
    max_prob = 0.0
    min_prob = 1.0
    max_swing = 0.0
    max_prob_match = ""
    min_prob_match = ""
    max_swing_match = ""

    for run in all_match_runs:
        # Champion and runner
        m104 = run[104]
        champ = m104["winner"]
        runner = m104["home_team"] if champ == m104["away_team"] else m104["away_team"]
        champions.append(champ)
        runners.append(runner)
        finalists_pairs.append(tuple(sorted([m104["home_team"], m104["away_team"]])))

        # Semi-finalists
        sf_teams = [run[101]["home_team"], run[101]["away_team"], run[102]["home_team"], run[102]["away_team"]]
        semi_finalists.append(tuple(sorted(sf_teams)))

        # Quarter-finalists
        qf_teams = []
        for m_no in range(97, 101):
            qf_teams.extend([run[m_no]["home_team"], run[m_no]["away_team"]])
        quarter_finalists.append(tuple(sorted(qf_teams)))

        # Round of 16
        r16_teams = []
        for m_no in range(89, 97):
            r16_teams.extend([run[m_no]["home_team"], run[m_no]["away_team"]])
        r16s.append(tuple(sorted(r16_teams)))

        # Count upsets in this simulation
        upsets = sum(1 for m in run.values() if is_upset(m))
        upset_counts.append(upsets)

        # Track probabilities
        for m in run.values():
            p_home = m["prob_home_win"]
            p_away = m["prob_away_win"]
            high = max(p_home, p_away)
            low = min(p_home, p_away)
            swing = abs(p_home - p_away)
            
            m_label = f"{m['home_team']} vs {m['away_team']}"
            if high > max_prob:
                max_prob = high
                max_prob_match = m_label
            if low < min_prob and low > 0.0:
                min_prob = low
                min_prob_match = m_label
            if swing > max_swing:
                max_swing = swing
                max_swing_match = m_label

    # Mode calculations
    most_common_champ = Counter(champions).most_common(1)[0][0]
    most_common_runner = Counter(runners).most_common(1)[0][0]
    most_common_final = " vs ".join(Counter(finalists_pairs).most_common(1)[0][0])
    most_common_sfs = Counter(semi_finalists).most_common(1)[0][0]
    most_common_qfs = Counter(quarter_finalists).most_common(1)[0][0]
    most_common_r16 = Counter(r16s).most_common(1)[0][0]

    upset_rate = np.mean(upset_counts)
    # total matches in bracket = 32
    pct_upsets = (upset_rate / 32.0) * 100.0

    return {
        "most_common_champion": most_common_champ,
        "most_common_runner_up": most_common_runner,
        "most_common_final": most_common_final,
        "most_common_semi_finalists": list(most_common_sfs),
        "most_common_quarter_finalists": list(most_common_qfs),
        "most_common_round_of_16": list(most_common_r16),
        "upset_frequency_pct": pct_upsets,
        "avg_number_of_upsets": upset_rate,
        "highest_probability_match": {
            "matchup": max_prob_match,
            "prob": round(max_prob, 4)
        },
        "lowest_probability_match": {
            "matchup": min_prob_match,
            "prob": round(min_prob, 4)
        },
        "largest_probability_swing": {
            "matchup": max_swing_match,
            "swing": round(max_swing, 4)
        }
    }

def calculate_uncertainty_analysis(all_team_runs: List[Dict[str, Dict[str, Any]]], teams_list: List[str]) -> pd.DataFrame:
    """Computes variance, standard error, entropy, confidence intervals, and stability index."""
    n_sims = len(all_team_runs)
    records = []
    
    for t in teams_list:
        champs = np.array([run[t]["reached_champion"] for run in all_team_runs], dtype=float)
        mean_p = np.mean(champs)
        
        # Standard error of the mean (SEM)
        se = sem(champs) if len(champs) > 1 else 0.0
        
        # 95% Confidence Interval for proportion: mean_p +/- 1.96 * se
        ci_lower = max(0.0, mean_p - 1.96 * se)
        ci_upper = min(1.0, mean_p + 1.96 * se)
        
        variance = np.var(champs)
        std_dev = np.std(champs)
        
        # Binary entropy: -p log2(p) - (1-p) log2(1-p)
        if mean_p > 0.0 and mean_p < 1.0:
            entropy = -mean_p * np.log2(mean_p) - (1.0 - mean_p) * np.log2(1.0 - mean_p)
        else:
            entropy = 0.0
            
        # Stability index = 1.0 - std_dev
        stability_score = 1.0 - std_dev

        records.append({
            "team": t,
            "mean_prob": mean_p,
            "variance": variance,
            "std_dev": std_dev,
            "std_error": se,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "entropy": entropy,
            "prediction_stability_score": stability_score
        })
        
    return pd.DataFrame(records)

def calculate_convergence(all_team_runs: List[Dict[str, Dict[str, Any]]], teams_list: List[str], 
                          checkpoints: List[int]) -> pd.DataFrame:
    """Extracts convergence data (champion probabilities) at different simulation lengths."""
    records = []
    
    for count in checkpoints:
        if count > len(all_team_runs):
            continue
            
        subset = all_team_runs[:count]
        for t in teams_list:
            champs = sum(run[t]["reached_champion"] for run in subset)
            prob = champs / count
            records.append({
                "simulations": count,
                "team": t,
                "champion_prob": prob
            })
            
    return pd.DataFrame(records)

def validate_probabilities(team_stats_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Runs structural checks on generated probabilities for logical consistency."""
    errors = []
    
    # 1. Champion probabilities sum to exactly 1.0
    sum_champs = team_stats_df["champion_prob"].sum()
    if not np.isclose(sum_champs, 1.0, atol=1e-3):
        errors.append(f"Champion probabilities sum to {sum_champs:.4f} instead of 1.0")

    # 2. Runner-Up probabilities sum to exactly 1.0
    sum_runners = team_stats_df["runner_up_prob"].sum()
    if not np.isclose(sum_runners, 1.0, atol=1e-3):
        errors.append(f"Runner-Up probabilities sum to {sum_runners:.4f} instead of 1.0")

    # 3. Round probabilities consistent: Champion <= Final <= Semi-final <= Quarter-final <= Round of 16
    for idx, row in team_stats_df.iterrows():
        t = row["team"]
        p_c = row["champion_prob"]
        p_f = row["runner_up_prob"] + p_c  # reached final probability
        p_sf = row["semi_final_prob"]
        p_qf = row["quarter_final_prob"]
        p_r16 = row["round_of_16_prob"]
        
        # Reached final is p_c + runner_up_prob.
        # Semi-final probability is reached SF.
        if p_c > p_f + 1e-4:
            errors.append(f"Team {t}: Champion prob ({p_c:.3f}) > Reached Final prob ({p_f:.3f})")
        if p_f > p_sf + 1e-4:
            errors.append(f"Team {t}: Reached Final prob ({p_f:.3f}) > Reached Semi prob ({p_sf:.3f})")
        if p_sf > p_qf + 1e-4:
            errors.append(f"Team {t}: Reached Semi prob ({p_sf:.3f}) > Reached Quarter prob ({p_qf:.3f})")
        if p_qf > p_r16 + 1e-4:
            errors.append(f"Team {t}: Reached Quarter prob ({p_qf:.3f}) > Reached R16 prob ({p_r16:.3f})")
            
    # 4. Check for invalid or negative probabilities
    for col in ["champion_prob", "runner_up_prob", "semi_final_prob", "quarter_final_prob", "round_of_16_prob"]:
        if (team_stats_df[col] < 0.0).any() or (team_stats_df[col] > 1.0).any():
            errors.append(f"Column {col} contains probabilities outside the valid range [0, 1]")

    return len(errors) == 0, errors
