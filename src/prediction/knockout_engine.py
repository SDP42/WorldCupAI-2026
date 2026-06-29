"""WorldCupAI — Phase 8: Official FIFA 2026 Knockout Prediction Engine

Predicts the complete tournament bracket from the Round of 32 through the Final
using the Phase 7.1 production ensemble.
Bypasses macOS SIGSEGV conflicts using subprocess-isolated ML model loading.
"""
import os
import sys
import json
import time
import pickle
import tempfile
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

from src.utils.logger import setup_logger
from src.ensemble.pipeline import EnsemblePipeline
from src.deep_learning.prediction_interface import DLPredictionInterface
from src.deep_learning.ann_model import ANNModel
from src.deep_learning.lstm_model import LSTMModel
from src.deep_learning.dataset import SequenceBuilder

logger = setup_logger("knockout_engine")


def map_team_name(name: str) -> str:
    """Standardizes tournament team names to canonical dataset names."""
    mapping = {
        "Ivory Coast": "Cote d'Ivoire",
        "DR Congo": "Democratic Republic of the Congo",
        "Bosnia & Herzegovina": "Bosnia and Herzegovina",
        "Bosnia and Herzegovina": "Bosnia and Herzegovina"
    }
    return mapping.get(name, name)


class SubprocessEnsemblePipeline(EnsemblePipeline):
    """EnsemblePipeline subclass that isolates ML model execution in subprocesses."""

    def load_base_models(self):
        """Only loads PyTorch-based DL models in the main process."""
        logger.info("SubprocessEnsemblePipeline: Loading DL base models...")
        for name, m_dir in self.model_dirs.items():
            if name == "ANN":
                self.ann_interface = DLPredictionInterface(m_dir, ANNModel)
            elif name == "LSTM":
                self.lstm_interface = DLPredictionInterface(m_dir, LSTMModel)
                # Also load the SequenceBuilder config
                cfg_path = os.path.join(m_dir, "sequence_config.json")
                if os.path.exists(cfg_path):
                    with open(cfg_path) as f:
                        seq_cfg = json.load(f)
                    self.seq_builder = SequenceBuilder(seq_len=seq_cfg["seq_len"])
                else:
                    self.seq_builder = SequenceBuilder(seq_len=5)
        logger.info("SubprocessEnsemblePipeline: DL base models loaded.")

    def get_base_predictions(
        self,
        test_df:       pd.DataFrame,
        df_full:       Optional[pd.DataFrame] = None,
        feature_cols:  Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Spawns subprocesses for ML model prediction to prevent library conflicts."""
        # 1. Run DL models in the main process
        predictions = {}
        
        # Load ANN if present
        if self.ann_interface:
            ann_prep_path = os.path.join(self.model_dirs["ANN"], "preprocessing.pkl")
            with open(ann_prep_path, "rb") as f:
                ann_prep = pickle.load(f)
            X_ann = ann_prep.transform(test_df)
            y_prob = self.ann_interface.predict_proba(X_ann)
            y_pred = np.argmax(y_prob, axis=1)
            predictions["ANN"] = {"y_prob": y_prob, "y_pred": y_pred}

        # Load LSTM if present
        if self.lstm_interface and self.seq_builder:
            if df_full is None:
                raise ValueError("df_full must be provided to build sequences for LSTM model!")
            
            lstm_prep_path = os.path.join(self.model_dirs["LSTM"], "preprocessing.pkl")
            with open(lstm_prep_path, "rb") as f:
                lstm_prep = pickle.load(f)
            
            f_cols = feature_cols or list(self.model_dirs.keys())
            X_full_proc = lstm_prep.transform(df_full)
            y_full      = df_full["target"].values if "target" in df_full.columns else np.zeros(len(df_full))
            
            # Match dates and names to find sequence indices
            test_indices = []
            for _, row in test_df.iterrows():
                mask = (df_full["date"] == row["date"]) & \
                       (df_full["home_team"] == row["home_team"]) & \
                       (df_full["away_team"] == row["away_team"])
                matched = df_full[mask].index.values
                if len(matched) > 0:
                    test_indices.append(matched[0])
                else:
                    raise ValueError(f"Match not found in df_full: {row['date']} {row['home_team']} vs {row['away_team']}")
            
            from src.deep_learning.dataset import SequenceBuilder
            X_seq_test, _ = self.seq_builder.build(X_full_proc, y_full, np.array(test_indices))
            y_prob = self.lstm_interface.predict_proba(X_seq_test)
            y_pred = np.argmax(y_prob, axis=1)
            predictions["LSTM"] = {"y_prob": y_prob, "y_pred": y_pred}

        # 2. Run ML models in subprocesses concurrently to maximize performance
        # Save test_df to a temporary CSV
        fd, temp_csv = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        test_df.to_csv(temp_csv, index=False)

        script_path = os.path.join("src", "prediction", "ml_predict_subprocess.py")
        processes = []
        temp_files = []

        for name, m_dir in self.model_dirs.items():
            if name in ["ANN", "LSTM"]:
                continue

            fd_out, temp_json = tempfile.mkstemp(suffix=".json")
            os.close(fd_out)
            temp_files.append((name, temp_json))

            cmd = [sys.executable, script_path, name, m_dir, temp_csv, temp_json]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append((name, p))

        # Wait for all processes to finish and handle errors
        for name, p in processes:
            stdout, stderr = p.communicate()
            if p.returncode != 0:
                logger.error(f"ML subprocess prediction for '{name}' failed!\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}")
                if os.path.exists(temp_csv):
                    os.remove(temp_csv)
                raise subprocess.CalledProcessError(p.returncode, p.args, stdout, stderr)

        # Parse prediction outputs
        for name, temp_json in temp_files:
            with open(temp_json) as f:
                out_data = json.load(f)
            predictions[name] = {
                "y_prob": np.array(out_data["y_prob"], dtype=np.float32),
                "y_pred": np.array(out_data["y_pred"], dtype=np.int64)
            }
            if os.path.exists(temp_json):
                os.remove(temp_json)

        if os.path.exists(temp_csv):
            os.remove(temp_csv)

        return predictions


class KnockoutEngine:
    """Orchestrates predictions and progressions for the FIFA 2026 Knockout Stage."""

    def __init__(
        self,
        fixtures_path: str = "configs/knockout_fixtures.json",
        ensemble_pkl_path: str = "models/ensemble/ensemble.pkl",
        feature_store_path: str = "processed/feature_store.parquet",
        master_dataset_path: str = "processed/master_dataset.parquet",
        draw_resolution_strategy: str = "higher_elo",
    ):
        self.fixtures_path = fixtures_path
        self.ensemble_pkl_path = ensemble_pkl_path
        self.feature_store_path = feature_store_path
        self.master_dataset_path = master_dataset_path
        self.draw_resolution_strategy = draw_resolution_strategy

        # Load historical datasets
        logger.info("Loading feature store and master dataset...")
        self.df_store = pd.read_parquet(self.feature_store_path)
        self.df_master = pd.read_parquet(self.master_dataset_path)

        # Parse master dates
        self.df_master["date"] = pd.to_datetime(self.df_master["date"])
        self.df_store["date"] = pd.to_datetime(self.df_store["date"])

        # Load ensemble config & weights
        logger.info(f"Loading production ensemble from {ensemble_pkl_path}...")
        with open(ensemble_pkl_path, "rb") as f:
            self.ens_data = pickle.load(f)

        model_dirs = {
            "XGBoost":            "models/xgboost_optimized",
            "Gradient Boosting":  "models/gradient_boosting_optimized",
            "Random Forest":      "models/random_forest_optimized",
            "Extra Trees":        "models/extra_trees_optimized",
            "Logistic Regression":"models/logistic_regression_optimized",
            "ANN":                "models/ann",
            "LSTM":               "models/lstm",
        }

        # Override pipeline loading with SubprocessEnsemblePipeline to bypass library conflicts
        self.pipeline = SubprocessEnsemblePipeline(
            model_dirs=model_dirs,
            optimized_weights=self.ens_data.get("optimized_weights"),
            stacking_model=self.ens_data.get("stacking_model"),
            blending_model=self.ens_data.get("blending_model"),
            best_method=self.ens_data.get("best_method", "Weighted Soft Voting")
        )
        self.pipeline.load_base_models()
        logger.info(f"SubprocessEnsemblePipeline instantiated. Selected method: {self.pipeline.best_method}")

        # Round of 32 fixtures
        with open(self.fixtures_path) as f:
            self.round_32_fixtures = json.load(f)

        self.matches_played: Dict[int, Dict[str, Any]] = {}
        self.team_last_match_date: Dict[str, str] = {}

    def _elo_from_rank(self, rank: float) -> float:
        """Rank-based Elo proxy calibrated to observed Elo-rank pairs.

        Calibration points (Nov/Dec 2025 data):
          Rank 1 (Argentina)=2060, Rank 3 (Spain)=2092, Rank 5 (Brazil)=1957,
          Rank 13 (Germany)=1926, Rank 18 (Japan)=1906, Rank 29 (Sweden)=1737,
          Rank 46 (Algeria)=1783, Rank 64 (Ghana)=1587.
        """
        if rank <= 10:
            return max(1860.0, 2080.0 - rank * 12.0)
        elif rank <= 30:
            return max(1720.0, 2000.0 - rank * 9.0)
        elif rank <= 60:
            return max(1580.0, 1820.0 - rank * 4.0)
        else:
            return max(1500.0, 1720.0 - rank * 2.5)

    def get_team_latest_state(self, team_name: str) -> Dict[str, Any]:
        """Queries historical database for pre-tournament stats.

        Two-pass lookup to fix the Elo NaN bug:
        - Form features, rank, attack/defense ratings:
            from ABSOLUTE LATEST match (includes 2026 group stage).
        - Elo:
            from LATEST MATCH WITH VALID ELO DATA (non-NaN).
            2026 group stage matches have elo=NaN — using them caused every team
            to receive elo=1500 and elo_diff=0, destroying prediction quality.
        """
        canon_name = map_team_name(team_name)
        m1 = self.df_master["home_team"] == canon_name
        m2 = self.df_master["away_team"] == canon_name
        team_matches = self.df_master[m1 | m2].sort_values("date")

        if len(team_matches) == 0:
            raise ValueError(f"No historical matches found for canonical team: {canon_name}")

        # ── Pass 1: Form features, rank, attack/defence from absolute latest ──
        last_match = team_matches.iloc[-1]
        match_date = last_match["date"]
        is_home = (last_match["home_team"] == canon_name)

        store_row = self.df_store[
            (self.df_store["date"] == match_date) &
            (self.df_store["home_team"] == last_match["home_team"]) &
            (self.df_store["away_team"] == last_match["away_team"])
        ]
        if len(store_row) == 0:
            store_row = self.df_store[
                (self.df_store["home_team"] == last_match["home_team"]) &
                (self.df_store["away_team"] == last_match["away_team"])
            ].iloc[-1:]
        store_row = store_row.iloc[0]
        prefix = "home_" if is_home else "away_"
        rank = float(last_match["home_rank"] if is_home else last_match["away_rank"])

        # ── Pass 2: Elo from latest match with valid non-NaN Elo data ─────────
        matches_with_elo = team_matches[
            team_matches["home_elo"].notna() | team_matches["away_elo"].notna()
        ]
        if len(matches_with_elo) > 0:
            last_elo_match = matches_with_elo.iloc[-1]
            is_home_elo = (last_elo_match["home_team"] == canon_name)
            raw_elo = last_elo_match["home_elo"] if is_home_elo else last_elo_match["away_elo"]
            if pd.isna(raw_elo):
                raw_elo = self._elo_from_rank(rank)
        else:
            # No Elo data for this team at all — use rank-based proxy
            raw_elo = self._elo_from_rank(rank)

        form_keys = [
            "win_rate_5", "loss_rate_5", "draw_rate_5", "avg_goals_scored_5",
            "avg_goals_conceded_5", "goal_diff_5", "clean_sheet_rate_5", "avg_points_5",
            "win_rate_10", "loss_rate_10", "draw_rate_10", "avg_goals_scored_10",
            "avg_goals_conceded_10", "goal_diff_10", "clean_sheet_rate_10", "avg_points_10"
        ]

        stats = {
            "team_name": canon_name,
            "elo": float(raw_elo),
            "rank": rank,
            "attack_rating": store_row[f"{prefix}attack_rating"],
            "defence_rating": store_row[f"{prefix}defence_rating"],
            "clean_sheet_rate": store_row[f"{prefix}clean_sheet_rate"],
            "world_cup_titles_before": store_row[f"{prefix}world_cup_titles_before"],
            "world_cup_participations_before": store_row[f"{prefix}world_cup_participations_before"],
            "groups_passed_before": store_row[f"{prefix}groups_passed_before"],
            "round16_before": store_row[f"{prefix}round16_before"],
            "quarterfinals_before": store_row[f"{prefix}quarterfinals_before"],
            "semifinals_before": store_row[f"{prefix}semifinals_before"],
            "finals_before": store_row[f"{prefix}finals_before"]
        }

        for k in form_keys:
            stats[k] = store_row[f"{prefix}form_{k}"]

        return stats

    def compute_h2h(self, team_a: str, team_b: str) -> Dict[str, Any]:
        """Computes live historical H2H between A and B from master dataset.

        NOTE — Neutral Venue Context:
          Since FIFA World Cup matches are played at neutral venues, the labels
          'h2h_home_wins' and 'h2h_away_wins' here are **positional** (team_a wins
          vs team_b wins), NOT venue-based home/away wins. The feature names are
          kept as-is to match the trained model's expected feature schema.
          team_a = the team passed as 'home_team' (listed first in the bracket).
          team_b = the team passed as 'away_team' (listed second in the bracket).
        """
        canon_a = map_team_name(team_a)
        canon_b = map_team_name(team_b)

        m1 = (self.df_master["home_team"] == canon_a) & (self.df_master["away_team"] == canon_b)
        m2 = (self.df_master["home_team"] == canon_b) & (self.df_master["away_team"] == canon_a)
        h2h_matches = self.df_master[m1 | m2]

        meetings = len(h2h_matches)
        if meetings == 0:
            return {
                "h2h_meetings": 0, "h2h_draws": 0,
                "h2h_home_wins": 0, "h2h_away_wins": 0, "h2h_gd": 0.0
            }

        draws = int(np.sum(h2h_matches["home_score"] == h2h_matches["away_score"]))

        # wins_a = historical wins by team_a (positionally 'home'), regardless of venue
        wins_a = 0
        gd_a = 0.0
        for _, row in h2h_matches.iterrows():
            if row["home_team"] == canon_a:
                gd_a += (row["home_score"] - row["away_score"])
                if row["home_score"] > row["away_score"]:
                    wins_a += 1
            else:
                gd_a += (row["away_score"] - row["home_score"])
                if row["away_score"] > row["home_score"]:
                    wins_a += 1

        wins_b = meetings - draws - wins_a

        return {
            "h2h_meetings": meetings,
            "h2h_draws": draws,
            # 'h2h_home_wins' = wins by team_a (positional label, NOT venue-based)
            "h2h_home_wins": wins_a,
            # 'h2h_away_wins' = wins by team_b (positional label, NOT venue-based)
            "h2h_away_wins": wins_b,
            "h2h_gd": gd_a
        }

    def get_rest_days(self, team_name: str, current_date_str: str) -> float:
        """Calculates rest days for a team in the tournament."""
        canon_name = map_team_name(team_name)
        curr_date = pd.to_datetime(current_date_str)
        
        if canon_name in self.team_last_match_date:
            prev_date = pd.to_datetime(self.team_last_match_date[canon_name])
            rest = (curr_date - prev_date).days
            return float(np.clip(rest, 0, 30))
        
        m1 = self.df_master["home_team"] == canon_name
        m2 = self.df_master["away_team"] == canon_name
        team_matches = self.df_master[m1 | m2]
        
        if len(team_matches) > 0:
            last_date = team_matches.sort_values("date").iloc[-1]["date"]
            rest = (curr_date - last_date).days
            return float(np.clip(rest, 0, 30))

        return 5.0

    def construct_matchup_features(
        self,
        home_team: str,
        away_team: str,
        match_date_str: str
    ) -> pd.DataFrame:
        """Generates the 37-column model feature vector for a matchup."""
        home_stats = self.get_team_latest_state(home_team)
        away_stats = self.get_team_latest_state(away_team)
        h2h_stats = self.compute_h2h(home_team, away_team)

        home_rest = self.get_rest_days(home_team, match_date_str)
        away_rest = self.get_rest_days(away_team, match_date_str)

        row = {
            "elo_diff": home_stats["elo"] - away_stats["elo"],
            "elo_ratio": home_stats["elo"] / max(1.0, away_stats["elo"]),
            "rank_diff": home_stats["rank"] - away_stats["rank"],
            "rank_ratio": home_stats["rank"] / max(1.0, away_stats["rank"]),
            
            "home_form_win_rate_5": home_stats["win_rate_5"],
            "away_form_win_rate_5": away_stats["win_rate_5"],
            "home_form_avg_goals_scored_5": home_stats["avg_goals_scored_5"],
            "away_form_avg_goals_scored_5": away_stats["avg_goals_scored_5"],
            "home_form_avg_goals_conceded_5": home_stats["avg_goals_conceded_5"],
            "away_form_avg_goals_conceded_5": away_stats["avg_goals_conceded_5"],
            "home_form_clean_sheet_rate_5": home_stats["clean_sheet_rate_5"],
            "away_form_clean_sheet_rate_5": away_stats["clean_sheet_rate_5"],
            
            "home_form_win_rate_10": home_stats["win_rate_10"],
            "away_form_win_rate_10": away_stats["win_rate_10"],
            "home_form_avg_goals_scored_10": home_stats["avg_goals_scored_10"],
            "away_form_avg_goals_scored_10": away_stats["avg_goals_scored_10"],
            "home_form_avg_goals_conceded_10": home_stats["avg_goals_conceded_10"],
            "away_form_avg_goals_conceded_10": away_stats["avg_goals_conceded_10"],
            "home_form_clean_sheet_rate_10": home_stats["clean_sheet_rate_10"],
            "away_form_clean_sheet_rate_10": away_stats["clean_sheet_rate_10"],
            
            "h2h_meetings": h2h_stats["h2h_meetings"],
            "h2h_home_wins": h2h_stats["h2h_home_wins"],
            "h2h_away_wins": h2h_stats["h2h_away_wins"],
            "h2h_draws": h2h_stats["h2h_draws"],
            "h2h_gd": h2h_stats["h2h_gd"],
            
            "home_attack_rating": home_stats["attack_rating"],
            "away_attack_rating": away_stats["attack_rating"],
            "home_defence_rating": home_stats["defence_rating"],
            "away_defence_rating": away_stats["defence_rating"],
            
            "home_world_cup_titles_before": home_stats["world_cup_titles_before"],
            "away_world_cup_titles_before": away_stats["world_cup_titles_before"],
            
            "is_neutral": 1.0,
            "is_world_cup": 1.0,
            "is_friendly": 0.0,
            
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "rest_difference": home_rest - away_rest
        }

        row["home_world_cup_participations_before"] = home_stats["world_cup_participations_before"]
        row["away_world_cup_participations_before"] = away_stats["world_cup_participations_before"]
        row["home_groups_passed_before"] = home_stats["groups_passed_before"]
        row["away_groups_passed_before"] = away_stats["groups_passed_before"]
        row["home_round16_before"] = home_stats["round16_before"]
        row["away_round16_before"] = away_stats["round16_before"]
        row["home_quarterfinals_before"] = home_stats["quarterfinals_before"]
        row["away_quarterfinals_before"] = away_stats["quarterfinals_before"]
        row["home_semifinals_before"] = home_stats["semifinals_before"]
        row["away_semifinals_before"] = away_stats["semifinals_before"]
        row["home_finals_before"] = home_stats["finals_before"]
        row["away_finals_before"] = away_stats["finals_before"]
        row["home_clean_sheet_rate"] = home_stats["clean_sheet_rate"]
        row["away_clean_sheet_rate"] = away_stats["clean_sheet_rate"]

        row["date"] = pd.to_datetime(match_date_str)
        row["home_team"] = map_team_name(home_team)
        row["away_team"] = map_team_name(away_team)
        row["tournament"] = "FIFA World Cup"
        row["city"] = "TBD"
        row["country"] = "TBD"

        return pd.DataFrame([row])

    def resolve_draw_deterministically(self, home_team: str, away_team: str) -> Tuple[str, str]:
        """Resolves draws deterministically using configurable statistics (shootout/extra-time mock)."""
        home_canon = map_team_name(home_team)
        away_canon = map_team_name(away_team)
        home_stats = self.get_team_latest_state(home_canon)
        away_stats = self.get_team_latest_state(away_canon)

        if self.draw_resolution_strategy == "higher_elo":
            if home_stats["elo"] != away_stats["elo"]:
                w = home_team if home_stats["elo"] > away_stats["elo"] else away_team
                return w, f"Deterministic shootout won by higher pre-match Elo rating ({max(home_stats['elo'], away_stats['elo'])} vs {min(home_stats['elo'], away_stats['elo'])})"
            
        if home_stats["rank"] != away_stats["rank"]:
            w = home_team if home_stats["rank"] < away_stats["rank"] else away_team
            return w, f"Deterministic shootout won by better pre-match FIFA ranking ({min(home_stats['rank'], away_stats['rank'])} vs {max(home_stats['rank'], away_stats['rank'])})"

        if home_stats["attack_rating"] != away_stats["attack_rating"]:
            w = home_team if home_stats["attack_rating"] > away_stats["attack_rating"] else away_team
            return w, "Deterministic shootout won by higher attacking rating"

        w = home_team if home_team < away_team else away_team
        return w, "Shootout resolved by alphabetical tie-breaker"

    def predict_match(
        self,
        match_no: int,
        round_name: str,
        home_team: str,
        away_team: str,
        date_str: str,
        stadium: str = "TBD",
        city: str = "TBD"
    ) -> Dict[str, Any]:
        """Loads features, runs ensemble Soft/Weighted Soft voting, and determines winner."""
        logger.info(f"Predicting Match {match_no}: {home_team} vs {away_team} ({round_name})")

        # Enforce venue neutrality (symmetry) by predicting both home/away configurations
        match_df_1 = self.construct_matchup_features(home_team, away_team, date_str)
        df_full_1 = pd.concat([self.df_master, match_df_1], ignore_index=True)
        df_full_1["target"] = 0
        probs_1 = self.pipeline.predict_proba(match_df_1, df_full_1)[0]
        probs_1 = np.array(probs_1, dtype=float)

        match_df_2 = self.construct_matchup_features(away_team, home_team, date_str)
        df_full_2 = pd.concat([self.df_master, match_df_2], ignore_index=True)
        df_full_2["target"] = 0
        probs_2 = self.pipeline.predict_proba(match_df_2, df_full_2)[0]
        probs_2 = np.array(probs_2, dtype=float)

        # Average the corresponding outcomes (reversing the team perspectives for the second match)
        prob_home = (probs_1[2] + probs_2[0]) / 2.0  # Team 1 Win
        prob_draw = (probs_1[1] + probs_2[1]) / 2.0  # Draw
        prob_away = (probs_1[0] + probs_2[2]) / 2.0  # Team 2 Win

        probs = np.array([prob_away, prob_draw, prob_home], dtype=float)
        probs /= np.sum(probs)

        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

        prob_away, prob_draw, prob_home = probs[0], probs[1], probs[2]
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])

        is_draw = (pred_idx == 1)
        shootout_played = False
        shootout_reason = ""

        if is_draw:
            winner, shootout_reason = self.resolve_draw_deterministically(home_team, away_team)
            shootout_played = True
        else:
            winner = home_team if pred_idx == 2 else away_team

        # FIFA World Cup is played at neutral venues — use team names in outcome label,
        # not "Home Win" / "Away Win" which implies a venue-based advantage.
        if pred_idx == 1:
            predicted_outcome = "Draw"
        elif pred_idx == 2:
            predicted_outcome = f"{home_team} Win"
        else:
            predicted_outcome = f"{away_team} Win"

        # Save last match date to track rest days correctly
        self.team_last_match_date[map_team_name(home_team)] = date_str
        self.team_last_match_date[map_team_name(away_team)] = date_str

        # Retrieve ELO ratings for explainability report
        home_elo = self.get_team_latest_state(home_team)["elo"]
        away_elo = self.get_team_latest_state(away_team)["elo"]

        match_report = {
            "match_no": match_no,
            "round": round_name,
            # 'home_team' / 'away_team' are positional bracket labels (Team 1 / Team 2)
            # — NOT actual home/away venue designations (FIFA WC is neutral venue).
            "home_team": home_team,
            "away_team": away_team,
            # Neutral-venue aliases for display purposes
            "team1": home_team,
            "team2": away_team,
            "date": date_str,
            "stadium": stadium,
            "city": city,
            "neutral_venue": True,
            # Legacy probability fields kept for backward compatibility
            "prob_away_win": prob_away,
            "prob_draw": prob_draw,
            "prob_home_win": prob_home,
            # Neutral-venue probability aliases (team1/team2 instead of home/away)
            "prob_team1_win": prob_home,
            "prob_team2_win": prob_away,
            "predicted_outcome": predicted_outcome,
            "predicted_winner": winner,
            "confidence": confidence,
            "entropy": entropy,
            "shootout_played": shootout_played,
            "shootout_reason": shootout_reason,
            "features": {
                **match_df_1.iloc[0].to_dict(),
                "home_elo": home_elo,
                "away_elo": away_elo
            }
        }

        self.matches_played[match_no] = match_report
        return match_report

    def run_tournament(self) -> Dict[str, Any]:
        """Runs the entire FIFA 2026 Knockout Bracket (Matches 73–104)."""
        logger.info("Starting FIFA 2026 Knockout Prediction Engine...")

        # ── 1. Predict Round of 32 (Matches 73–88) ────────────────────────────
        for fixture in self.round_32_fixtures:
            self.predict_match(
                match_no=fixture["match_no"],
                round_name=fixture["round"],
                home_team=fixture["home_team"],
                away_team=fixture["away_team"],
                date_str=fixture.get("date", "2026-06-28"),
                stadium=fixture.get("stadium", "TBD"),
                city=fixture.get("city", "TBD")
            )

        # ── 2. Round of 16 (Matches 89–96) ────────────────────────────────────
        r16_pairings = [
            (89, 73, 75, "2026-07-04", "MetLife Stadium", "East Rutherford"),
            (90, 74, 77, "2026-07-04", "NRG Stadium", "Houston"),
            (91, 76, 78, "2026-07-05", "Mercedes-Benz Stadium", "Atlanta"),
            (92, 79, 80, "2026-07-05", "AT&T Stadium", "Arlington"),
            (93, 83, 84, "2026-07-06", "BMO Field", "Toronto"),
            (94, 81, 82, "2026-07-06", "Lumen Field", "Seattle"),
            (95, 86, 88, "2026-07-07", "Hard Rock Stadium", "Miami"),
            (96, 85, 87, "2026-07-07", "BC Place", "Vancouver"),
        ]
        for m_no, prev_a, prev_b, date, stad, city in r16_pairings:
            w_a = self.matches_played[prev_a]["predicted_winner"]
            w_b = self.matches_played[prev_b]["predicted_winner"]
            self.predict_match(m_no, "Round of 16", w_a, w_b, date, stad, city)

        # ── 3. Quarter-finals (Matches 97–100) ─────────────────────────────────
        qf_pairings = [
            (97, 89, 90, "2026-07-09", "Gillette Stadium", "Foxborough"),
            (98, 93, 94, "2026-07-10", "SoFi Stadium", "Inglewood"),
            (99, 91, 92, "2026-07-10", "Rose Bowl", "Pasadena"),
            (100, 95, 96, "2026-07-11", "Estadio Azteca", "Mexico City"),
        ]
        for m_no, prev_a, prev_b, date, stad, city in qf_pairings:
            w_a = self.matches_played[prev_a]["predicted_winner"]
            w_b = self.matches_played[prev_b]["predicted_winner"]
            self.predict_match(m_no, "Quarter-final", w_a, w_b, date, stad, city)

        # ── 4. Semi-finals (Matches 101–102) ──────────────────────────────────
        sf_pairings = [
            (101, 97, 98, "2026-07-14", "AT&T Stadium", "Arlington"),
            (102, 99, 100, "2026-07-15", "Mercedes-Benz Stadium", "Atlanta"),
        ]
        for m_no, prev_a, prev_b, date, stad, city in sf_pairings:
            w_a = self.matches_played[prev_a]["predicted_winner"]
            w_b = self.matches_played[prev_b]["predicted_winner"]
            self.predict_match(m_no, "Semi-final", w_a, w_b, date, stad, city)

        # ── 5. Third Place Play-off (Match 103) ───────────────────────────────
        m101 = self.matches_played[101]
        l_a = m101["home_team"] if m101["predicted_winner"] != m101["home_team"] else m101["away_team"]
        
        m102 = self.matches_played[102]
        l_b = m102["home_team"] if m102["predicted_winner"] != m102["home_team"] else m102["away_team"]
        
        self.predict_match(103, "Third Place Play-off", l_a, l_b, "2026-07-18", "Hard Rock Stadium", "Miami")

        # ── 6. Final (Match 104) ──────────────────────────────────────────────
        w_a = self.matches_played[101]["predicted_winner"]
        w_b = self.matches_played[102]["predicted_winner"]
        self.predict_match(104, "Final", w_a, w_b, "2026-07-19", "MetLife Stadium", "East Rutherford")

        return self.matches_played
