"""WorldCupAI — Phase 9 Plotting and Analytics Visualizer."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Set modern style and font properties
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13
})

def plot_champion_probability_bar(team_stats_df: pd.DataFrame, output_dir: str):
    """Generates a bar chart of champion probabilities for top teams."""
    df_sorted = team_stats_df.sort_values("champion_prob", ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(df_sorted))
    
    sns.barplot(
        x="champion_prob",
        y="team",
        data=df_sorted,
        palette=colors,
        hue="team",
        legend=False
    )
    
    plt.title("FIFA World Cup 2026 — Champion Probability (Top 15 Teams)")
    plt.xlabel("Probability of Winning")
    plt.ylabel("Team")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "champion_probability_bar.png"), dpi=150)
    plt.close()

def plot_champion_distribution_donut(team_stats_df: pd.DataFrame, output_dir: str):
    """Generates a donut chart of champion distributions."""
    df_sorted = team_stats_df.sort_values("champion_prob", ascending=False)
    top_5 = df_sorted.head(5)
    others_prob = df_sorted.iloc[5:]["champion_prob"].sum()
    
    labels = list(top_5["team"]) + ["Others"]
    probs = list(top_5["champion_prob"]) + [others_prob]
    
    # Filter out 0 probs
    filtered = [(l, p) for l, p in zip(labels, probs) if p > 0.0]
    if not filtered:
        return
    labels, probs = zip(*filtered)

    plt.figure(figsize=(8, 8))
    colors = sns.color_palette("pastel", len(labels))
    
    plt.pie(
        probs,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        pctdistance=0.85
    )
    
    # Donut center
    centre_circle = plt.Circle((0,0), 0.70, fc="white")
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.title("FIFA World Cup 2026 — Champion Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "champion_distribution.png"), dpi=150)
    plt.close()

def plot_runner_up_distribution_donut(team_stats_df: pd.DataFrame, output_dir: str):
    """Generates a donut chart of runner-up distributions."""
    df_sorted = team_stats_df.sort_values("runner_up_prob", ascending=False)
    top_5 = df_sorted.head(5)
    others_prob = df_sorted.iloc[5:]["runner_up_prob"].sum()
    
    labels = list(top_5["team"]) + ["Others"]
    probs = list(top_5["runner_up_prob"]) + [others_prob]
    
    # Filter out 0 probs
    filtered = [(l, p) for l, p in zip(labels, probs) if p > 0.0]
    if not filtered:
        return
    labels, probs = zip(*filtered)

    plt.figure(figsize=(8, 8))
    colors = sns.color_palette("muted", len(labels))
    
    plt.pie(
        probs,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        pctdistance=0.85
    )
    
    centre_circle = plt.Circle((0,0), 0.70, fc="white")
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.title("FIFA World Cup 2026 — Runner-up Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runner_up_distribution.png"), dpi=150)
    plt.close()

def plot_advancement_heatmap(team_stats_df: pd.DataFrame, output_dir: str):
    """Generates a heatmap of advancement probability by round for top teams."""
    df_sorted = team_stats_df.sort_values("champion_prob", ascending=False).head(16)
    
    # Build heatmap matrix
    matrix_data = []
    teams = []
    
    for _, row in df_sorted.iterrows():
        teams.append(row["team"])
        # Reconstruct round advancement probabilities
        # reached_r16, reached_qf, reached_sf, reached_final (c + r_up), champion
        p_c = row["champion_prob"]
        p_f = row["runner_up_prob"] + p_c
        p_sf = row["semi_final_prob"]
        p_qf = row["quarter_final_prob"]
        p_r16 = row["round_of_16_prob"]
        
        matrix_data.append([p_r16, p_qf, p_sf, p_f, p_c])
        
    df_heatmap = pd.DataFrame(
        matrix_data,
        index=teams,
        columns=["Round of 16", "Quarter-final", "Semi-final", "Final", "Champion"]
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_heatmap, annot=True, fmt=".1%", cmap="YlOrRd", cbar_kws={"label": "Probability"})
    plt.title("FIFA World Cup 2026 — Stage Advancement Heatmap")
    plt.ylabel("Team")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "advancement_heatmap.png"), dpi=150)
    plt.close()

def plot_probability_histogram(all_match_runs: List[Dict[int, Dict[str, Any]]], output_dir: str):
    """Generates a histogram of predicted match probabilities."""
    all_probs = []
    for run in all_match_runs:
        for m in run.values():
            all_probs.extend([m["prob_home_win"], m["prob_away_win"]])
            
    plt.figure(figsize=(9, 5))
    sns.histplot(all_probs, bins=25, kde=True, color="teal")
    plt.title("Distribution of Predicted Match Win Probabilities")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "probability_histogram.png"), dpi=150)
    plt.close()

def plot_upset_distribution(all_match_runs: List[Dict[int, Dict[str, Any]]], output_dir: str):
    """Generates a histogram of the number of upsets per tournament."""
    from src.simulation.statistics import is_upset
    upset_counts = []
    
    for run in all_match_runs:
        upsets = sum(1 for m in run.values() if is_upset(m))
        upset_counts.append(upsets)
        
    plt.figure(figsize=(9, 5))
    sns.histplot(upset_counts, bins=np.arange(min(upset_counts)-0.5, max(upset_counts)+1.5, 1), color="darkorange", kde=True)
    plt.title("Distribution of Total Upsets per Tournament Simulation")
    plt.xlabel("Number of Upsets (out of 32 matches)")
    plt.ylabel("Simulations")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "upset_distribution.png"), dpi=150)
    plt.close()

def plot_convergence(convergence_df: pd.DataFrame, output_dir: str):
    """Generates a line plot showing champion probability convergence."""
    # Find the top 6 teams by final probability
    final_probs = convergence_df[convergence_df["simulations"] == convergence_df["simulations"].max()]
    top_teams = list(final_probs.sort_values("champion_prob", ascending=False).head(6)["team"])
    
    df_subset = convergence_df[convergence_df["team"].isin(top_teams)]
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x="simulations",
        y="champion_prob",
        hue="team",
        marker="o",
        data=df_subset,
        linewidth=2.0
    )
    plt.title("Monte Carlo Convergence — Champion Probability Stability")
    plt.xlabel("Number of Simulations")
    plt.ylabel("Champion Probability")
    plt.legend(title="Team")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence_plot.png"), dpi=150)
    plt.close()

def plot_confidence_intervals(uncertainty_df: pd.DataFrame, output_dir: str):
    """Generates an error bar plot of 95% confidence intervals for top teams."""
    df_sorted = uncertainty_df.sort_values("mean_prob", ascending=False).head(12)
    
    plt.figure(figsize=(10, 6))
    
    # Calculate error margins
    y_err_lower = df_sorted["mean_prob"] - df_sorted["ci_95_lower"]
    y_err_upper = df_sorted["ci_95_upper"] - df_sorted["mean_prob"]
    y_err = [y_err_lower, y_err_upper]
    
    plt.errorbar(
        x=df_sorted["mean_prob"],
        y=df_sorted["team"],
        xerr=y_err,
        fmt="o",
        color="crimson",
        ecolor="gray",
        elinewidth=2,
        capsize=4,
        markersize=6
    )
    
    plt.title("95% Confidence Intervals for Champion Probability")
    plt.xlabel("Probability")
    plt.ylabel("Team")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_intervals_plot.png"), dpi=150)
    plt.close()
