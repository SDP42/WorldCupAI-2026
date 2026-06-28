"""WorldCupAI — Phase 10: Visualization Generator.

Generates 8 production-grade analytical plots for the XAI framework:
  1. Global Feature Importance (horizontal bar chart)
  2. Feature Correlation Heatmap
  3. Confidence Distribution (histogram)
  4. Prediction Margin Histogram
  5. Model Contribution Chart (stacked bar)
  6. Probability Distribution (violin/box)
  7. Tournament Confidence Heatmap (by round)
  8. Counterfactual Comparison (before/after delta)

Writes to: outputs/plots/
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for subprocess use
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Any, Optional

from src.explainability.utils import (
    FEATURE_COLS, FEATURE_LABELS, FEATURE_CATEGORIES,
    PLOTS_DIR, PREDICTIONS_DIR,
    get_label, ensure_dir,
)
from src.utils.logger import setup_logger

logger = setup_logger("xai_visualizations")

# Style constants
PALETTE = {
    "primary":    "#1E90FF",
    "secondary":  "#FF6B35",
    "positive":   "#2ECC71",
    "negative":   "#E74C3C",
    "neutral":    "#95A5A6",
    "background": "#0D1117",
    "surface":    "#161B22",
    "text":       "#C9D1D9",
    "grid":       "#30363D",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["background"],
    "axes.facecolor":    PALETTE["surface"],
    "axes.edgecolor":    PALETTE["grid"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["text"],
    "ytick.color":       PALETTE["text"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["grid"],
    "figure.dpi":        120,
    "font.family":       "DejaVu Sans",
})

CONFIDENCE_COLORS = {
    "Very High": "#2ECC71",
    "High":      "#27AE60",
    "Medium":    "#F39C12",
    "Low":       "#E67E22",
    "Very Low":  "#E74C3C",
}


def _save(fig, filename: str, plots_dir: str = PLOTS_DIR) -> str:
    ensure_dir(plots_dir)
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, bbox_inches="tight", facecolor=PALETTE["background"])
    plt.close(fig)
    logger.info(f"Saved plot → {path}")
    return path


def plot_global_feature_importance(
    importance_csv: str = "predictions/feature_importance.csv",
    plots_dir: str = PLOTS_DIR,
    top_n: int = 15,
) -> str:
    """Horizontal bar chart of top-N ensemble feature importances."""
    df = pd.read_csv(importance_csv)
    ens = df[df["model"] == "Ensemble (Weighted)"].copy()
    if ens.empty:
        ens = df[df["model"] == "Average"].copy()
    ens = ens.sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(ens)))
    bars = ax.barh(ens["label"], ens["importance"], color=colors, edgecolor=PALETTE["grid"])

    for bar, val in zip(bars, ens["importance"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9, color=PALETTE["text"])

    ax.set_xlabel("Feature Importance (Weighted Ensemble)", fontsize=11)
    ax.set_title("🌍 WorldCupAI — Global Feature Importance\nTop 15 Features (Ensemble Weighted)", fontsize=13, pad=15)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save(fig, "global_feature_importance.png", plots_dir)


def plot_feature_correlation_heatmap(
    importance_csv: str = "predictions/feature_importance.csv",
    plots_dir: str = PLOTS_DIR,
    top_n: int = 10,
) -> str:
    """Heatmap of feature importance correlation across models."""
    df = pd.read_csv(importance_csv)
    models = [m for m in df["model"].unique() if m not in ("Ensemble (Weighted)", "Average")]

    if len(models) < 2:
        logger.warning("Not enough models for correlation heatmap — skipping.")
        # Create placeholder
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Insufficient models for correlation heatmap",
                ha="center", va="center", transform=ax.transAxes, color=PALETTE["text"])
        return _save(fig, "feature_correlation_heatmap.png", plots_dir)

    # Get top-N features from ensemble importance
    ens = df[df["model"] == "Ensemble (Weighted)"].sort_values("importance", ascending=False).head(top_n)
    top_feats = ens["feature"].tolist()

    # Build matrix
    mat_data = {}
    for model in models:
        model_df = df[df["model"] == model]
        feat_imp = dict(zip(model_df["feature"], model_df["importance"]))
        mat_data[model] = [feat_imp.get(f, 0.0) for f in top_feats]

    mat = pd.DataFrame(mat_data, index=[ens[ens["feature"] == f]["label"].values[0]
                                         if not ens[ens["feature"] == f].empty else f
                                         for f in top_feats])

    corr = mat.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.coolwarm
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)

    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.values[i, j]) > 0.5 else PALETTE["text"])

    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Model Feature Importance Correlation", fontsize=12, pad=12)
    fig.tight_layout()
    return _save(fig, "feature_correlation_heatmap.png", plots_dir)


def plot_confidence_distribution(
    confidence_csv: str = "predictions/confidence_analysis.csv",
    plots_dir: str = PLOTS_DIR,
) -> str:
    """Histogram of prediction confidence scores by tier."""
    df = pd.read_csv(confidence_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram
    ax = axes[0]
    ax.hist(df["confidence"], bins=15, color=PALETTE["primary"], edgecolor=PALETTE["grid"],
            alpha=0.8, rwidth=0.9)
    ax.axvline(df["confidence"].mean(), color=PALETTE["secondary"], linestyle="--",
               linewidth=2, label=f"Mean: {df['confidence'].mean():.3f}")
    ax.set_xlabel("Confidence Score", fontsize=11)
    ax.set_ylabel("Number of Matches", fontsize=11)
    ax.set_title("Confidence Distribution", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: donut chart by tier
    ax2 = axes[1]
    tier_counts = df["confidence_tier"].value_counts()
    tier_order  = ["Very High", "High", "Medium", "Low", "Very Low"]
    labels = [t for t in tier_order if t in tier_counts.index]
    sizes  = [tier_counts.get(t, 0) for t in labels]
    colors = [CONFIDENCE_COLORS.get(t, PALETTE["neutral"]) for t in labels]

    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels, autopct="%1.0f%%", colors=colors,
        startangle=90, pctdistance=0.75,
        wedgeprops={"linewidth": 2, "edgecolor": PALETTE["background"]},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(10)
    ax2.set_title("Confidence Tier Distribution", fontsize=12)

    fig.suptitle("🎯 WorldCupAI — Prediction Confidence Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save(fig, "confidence_distribution.png", plots_dir)


def plot_prediction_margin_histogram(
    confidence_csv: str = "predictions/confidence_analysis.csv",
    plots_dir: str = PLOTS_DIR,
) -> str:
    """Histogram of prediction margins (top prob - second prob)."""
    df = pd.read_csv(confidence_csv)
    if "margin" not in df.columns:
        logger.warning("margin column not found in confidence_analysis.csv")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No margin data", ha="center", va="center",
                transform=ax.transAxes, color=PALETTE["text"])
        return _save(fig, "prediction_margin_histogram.png", plots_dir)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn(df["margin"].values)
    ax.bar(range(len(df)), df.sort_values("margin", ascending=False)["margin"],
           color=plt.cm.Blues(np.linspace(0.3, 0.9, len(df))),
           edgecolor=PALETTE["grid"], width=0.7)

    ax.axhline(df["margin"].mean(), color=PALETTE["secondary"], linestyle="--",
               linewidth=2, label=f"Mean margin: {df['margin'].mean():.3f}")
    ax.set_xlabel("Match (sorted by margin)", fontsize=11)
    ax.set_ylabel("Prediction Margin", fontsize=11)
    ax.set_title("Prediction Margin per Match (Top Prob − 2nd Prob)", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "prediction_margin_histogram.png", plots_dir)


def plot_model_contribution_chart(
    ensemble_csv: str = "predictions/ensemble_explanations.csv",
    plots_dir: str = PLOTS_DIR,
) -> str:
    """Stacked bar chart of per-model home-win contribution per match."""
    df = pd.read_csv(ensemble_csv)

    home_cols = [c for c in df.columns if c.endswith("_prob_home")]
    weight_cols = [c for c in df.columns if c.endswith("_weight")]

    if not home_cols:
        logger.warning("No per-model prob_home columns found in ensemble_explanations.csv")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No per-model data", ha="center", va="center",
                transform=ax.transAxes, color=PALETTE["text"])
        return _save(fig, "model_contribution_chart.png", plots_dir)

    model_names = [c.replace("_prob_home", "") for c in home_cols]
    x = np.arange(len(df))
    width = 0.6

    fig, ax = plt.subplots(figsize=(max(14, len(df)), 6))
    bottom = np.zeros(len(df))
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    for i, (model, col) in enumerate(zip(model_names, home_cols)):
        w_col = f"{model}_weight"
        weights = df[w_col].values if w_col in df.columns else np.ones(len(df))
        contrib = df[col].fillna(0).values * weights
        ax.bar(x, contrib, width, bottom=bottom, label=model,
               color=colors[i], edgecolor=PALETTE["background"], alpha=0.85)
        bottom += contrib

    labels = [f"M{row['match_no']}" for _, row in df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.set_ylabel("Weighted Contribution to Home Win Prob", fontsize=10)
    ax.set_title("Model Contributions to Ensemble Prediction (per match)", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "model_contribution_chart.png", plots_dir)


def plot_probability_distribution(
    tournament_json: str = "predictions/tournament_predictions.json",
    plots_dir: str = PLOTS_DIR,
) -> str:
    """Box/violin plot of probability distributions across tournament rounds."""
    with open(tournament_json) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    rounds = df["round"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    prob_cols = ["prob_home_win", "prob_draw", "prob_away_win"]
    titles    = ["Home Win Probability", "Draw Probability", "Away Win Probability"]
    colors    = [PALETTE["primary"], PALETTE["neutral"], PALETTE["secondary"]]

    for ax, col, title, color in zip(axes, prob_cols, titles, colors):
        data_by_round = [df[df["round"] == rnd][col].dropna().values for rnd in rounds]
        bp = ax.boxplot(
            data_by_round,
            patch_artist=True,
            labels=[r.replace(" ", "\n") for r in rounds],
            medianprops={"color": "white", "linewidth": 2},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Probability Distribution by Tournament Round", fontsize=13)
    fig.tight_layout()
    return _save(fig, "probability_distribution.png", plots_dir)


def plot_tournament_confidence_heatmap(
    confidence_csv: str = "predictions/confidence_analysis.csv",
    plots_dir: str = PLOTS_DIR,
) -> str:
    """Heatmap of confidence scores arranged by round."""
    df = pd.read_csv(confidence_csv)

    round_order = [
        "Round of 32", "Round of 16", "Quarter-final",
        "Semi-final", "Third Place Play-off", "Final"
    ]
    rounds_present = [r for r in round_order if r in df["round"].values]

    max_per_round = df.groupby("round").size().max()
    matrix = np.full((len(rounds_present), max_per_round), np.nan)

    for i, rnd in enumerate(rounds_present):
        rnd_df = df[df["round"] == rnd].reset_index(drop=True)
        for j, row in rnd_df.iterrows():
            if j < max_per_round:
                matrix[i, j] = row["confidence"]

    fig, ax = plt.subplots(figsize=(max(10, max_per_round), len(rounds_present) + 1))
    cmap = plt.cm.RdYlGn
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=0.35, vmax=0.85, aspect="auto")

    ax.set_yticks(range(len(rounds_present)))
    ax.set_yticklabels(rounds_present, fontsize=10)
    ax.set_xlabel("Match Index within Round", fontsize=10)
    ax.set_title("Tournament Confidence Heatmap by Round", fontsize=12, pad=12)

    plt.colorbar(im, ax=ax, label="Confidence Score", shrink=0.8)
    fig.tight_layout()
    return _save(fig, "tournament_confidence_heatmap.png", plots_dir)


def plot_counterfactual_comparison(
    cf_csv: str = "predictions/counterfactual_report.csv",
    plots_dir: str = PLOTS_DIR,
) -> str:
    """Bar chart comparing original vs counterfactual confidence per scenario."""
    if not os.path.exists(cf_csv):
        logger.warning("counterfactual_report.csv not found — skipping plot.")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No counterfactual data", ha="center", va="center",
                transform=ax.transAxes, color=PALETTE["text"])
        return _save(fig, "counterfactual_comparison.png", plots_dir)

    df = pd.read_csv(cf_csv)
    # Focus on Final match for clarity
    final_df = df[df["round"] == "Final"]
    if final_df.empty:
        final_df = df.head(20)  # fallback

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(final_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, final_df["original_confidence"], width,
                   label="Original", color=PALETTE["primary"], alpha=0.8)
    bars2 = ax.bar(x + width/2, final_df["new_confidence"], width,
                   label="Counterfactual", color=PALETTE["secondary"], alpha=0.8)

    # Mark decision flips
    for i, (_, row) in enumerate(final_df.iterrows()):
        if row.get("decision_flip", False):
            ax.annotate("⚡", (i, max(row["original_confidence"], row["new_confidence"]) + 0.02),
                        ha="center", fontsize=12, color="yellow")

    ax.set_xticks(x)
    ax.set_xticklabels(final_df["scenario"], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Confidence Score", fontsize=11)
    ax.set_title("Counterfactual Analysis — Original vs Perturbed Prediction\n(⚡ = Decision Flip)", fontsize=11)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "counterfactual_comparison.png", plots_dir)


def generate_all_plots(plots_dir: str = PLOTS_DIR) -> Dict[str, str]:
    """Generates all 8 Phase 10 visualizations."""
    paths = {}
    logger.info("Generating all Phase 10 visualizations...")

    tasks = [
        ("global_feature_importance",     lambda: plot_global_feature_importance(plots_dir=plots_dir)),
        ("feature_correlation_heatmap",   lambda: plot_feature_correlation_heatmap(plots_dir=plots_dir)),
        ("confidence_distribution",       lambda: plot_confidence_distribution(plots_dir=plots_dir)),
        ("prediction_margin_histogram",   lambda: plot_prediction_margin_histogram(plots_dir=plots_dir)),
        ("model_contribution_chart",      lambda: plot_model_contribution_chart(plots_dir=plots_dir)),
        ("probability_distribution",      lambda: plot_probability_distribution(plots_dir=plots_dir)),
        ("tournament_confidence_heatmap", lambda: plot_tournament_confidence_heatmap(plots_dir=plots_dir)),
        ("counterfactual_comparison",     lambda: plot_counterfactual_comparison(plots_dir=plots_dir)),
    ]

    for name, fn in tasks:
        try:
            path = fn()
            paths[name] = path
        except Exception as e:
            logger.error(f"Failed to generate plot '{name}': {e}")

    logger.info(f"Generated {len(paths)}/8 plots.")
    return paths
