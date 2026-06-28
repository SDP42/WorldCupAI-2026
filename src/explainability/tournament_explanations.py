"""WorldCupAI — Phase 10: Tournament-Level Explanation Generator.

Generates narrative-level explanations for the complete FIFA 2026
predicted bracket, including:
  - Champion explanation
  - Runner-up explanation
  - Biggest upsets
  - Highest & lowest confidence matches
  - Most competitive & one-sided matches
  - Most influential features
  - Most important models

Reads from:
  - predictions/tournament_predictions.json (Phase 8)
  - predictions/match_explanations.json (Phase 10 local)
  - predictions/confidence_analysis.csv (Phase 10 confidence)
  - predictions/feature_importance.csv  (Phase 10 importance)

Writes to:
  - predictions/tournament_explanations.json
  - predictions/tournament_summary.csv
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

from src.explainability.utils import (
    PREDICTIONS_DIR,
    get_label, classify_confidence, ensure_dir, save_json,
)
from src.utils.logger import setup_logger

logger = setup_logger("tournament_explanations")


class TournamentExplainer:
    """Generates tournament-wide XAI narratives and summary statistics."""

    def __init__(
        self,
        tournament_predictions: List[Dict[str, Any]],
        match_explanations: Optional[List[Dict[str, Any]]] = None,
        global_importances: Optional[Dict[str, float]] = None,
        ensemble_weights: Optional[Dict[str, float]] = None,
    ):
        self.predictions     = tournament_predictions
        self.match_exps      = match_explanations or []
        self.global_imp      = global_importances or {}
        self.ensemble_weights = ensemble_weights or {}
        self._result: Dict[str, Any] = {}

    def _get_final(self) -> Optional[Dict[str, Any]]:
        return next((m for m in self.predictions if m.get("round") == "Final"), None)

    def _get_third_place(self) -> Optional[Dict[str, Any]]:
        return next((m for m in self.predictions if "Third Place" in m.get("round", "")), None)

    def _get_semis(self) -> List[Dict[str, Any]]:
        return [m for m in self.predictions if m.get("round") == "Semi-final"]

    def _get_quarters(self) -> List[Dict[str, Any]]:
        return [m for m in self.predictions if m.get("round") == "Quarter-final"]

    def _get_r16(self) -> List[Dict[str, Any]]:
        return [m for m in self.predictions if m.get("round") == "Round of 16"]

    def _get_r32(self) -> List[Dict[str, Any]]:
        return [m for m in self.predictions if m.get("round") == "Round of 32"]

    def _build_champion_explanation(self) -> Dict[str, Any]:
        """Builds a narrative explanation for the tournament champion."""
        final = self._get_final()
        if not final:
            return {}

        champion = final["predicted_winner"]
        runner_up = (
            final["away_team"] if final["predicted_winner"] == final["home_team"]
            else final["home_team"]
        )
        confidence = final["confidence"]

        # Trace champion's path
        path = []
        for m in self.predictions:
            if m.get("predicted_winner") == champion and m.get("round") != "Third Place Play-off":
                opponent = m["away_team"] if m["predicted_winner"] == m["home_team"] else m["home_team"]
                path.append({
                    "round":     m["round"],
                    "opponent":  opponent,
                    "confidence": round(m["confidence"], 4),
                    "prob_win":   round(m.get("prob_home_win" if m["predicted_winner"] == m["home_team"]
                                              else "prob_away_win", 0.0), 4),
                })

        # Top features (from global importance)
        top_feats = sorted(self.global_imp.items(), key=lambda x: -x[1])[:5]
        top_feat_labels = [get_label(f) for f, _ in top_feats]

        # Build narrative
        path_summary = " → ".join(f"{p['round']} vs {p['opponent']}" for p in path)
        narrative = (
            f"{champion} is predicted to win the FIFA World Cup 2026. "
            f"Tournament path: {path_summary}. "
            f"Final vs {runner_up} won with {confidence*100:.1f}% confidence. "
            f"Key strengths: {', '.join(top_feat_labels[:3])}."
        )

        return {
            "champion":          champion,
            "runner_up":         runner_up,
            "final_confidence":  round(confidence, 4),
            "tournament_path":   path,
            "key_strengths":     top_feat_labels,
            "narrative":         narrative,
        }

    def _build_runner_up_explanation(self) -> Dict[str, Any]:
        final = self._get_final()
        if not final:
            return {}

        champion = final["predicted_winner"]
        runner_up = (
            final["away_team"] if champion == final["home_team"]
            else final["home_team"]
        )

        path = []
        for m in self.predictions:
            if m.get("predicted_winner") == runner_up and m.get("round") != "Third Place Play-off":
                opponent = m["away_team"] if m["predicted_winner"] == m["home_team"] else m["home_team"]
                path.append({
                    "round":     m["round"],
                    "opponent":  opponent,
                    "confidence": round(m["confidence"], 4),
                })

        narrative = (
            f"{runner_up} reaches the Final but falls short against {champion}. "
            f"Path to Final: {' → '.join(p['round'] for p in path)}."
        )

        return {
            "runner_up":  runner_up,
            "champion":   champion,
            "path":       path,
            "narrative":  narrative,
        }

    def _biggest_upsets(self, n: int = 5) -> List[Dict[str, Any]]:
        """Finds matches where the underdog (lower win probability) won."""
        upsets = []
        for m in self.predictions:
            prob_home = m.get("prob_home_win", 0.0)
            prob_away = m.get("prob_away_win", 0.0)
            winner    = m["predicted_winner"]

            # Check if winner was the underdog
            if winner == m["home_team"] and prob_home < prob_away:
                upset_margin = prob_away - prob_home
                upsets.append({"match": m, "upset_margin": upset_margin})
            elif winner == m["away_team"] and prob_away < prob_home:
                upset_margin = prob_home - prob_away
                upsets.append({"match": m, "upset_margin": upset_margin})

        upsets.sort(key=lambda x: -x["upset_margin"])
        return [
            {
                "match_no":      u["match"]["match_no"],
                "round":         u["match"]["round"],
                "winner":        u["match"]["predicted_winner"],
                "loser":         u["match"]["away_team"] if u["match"]["predicted_winner"] == u["match"]["home_team"]
                                 else u["match"]["home_team"],
                "upset_margin":  round(u["upset_margin"], 4),
                "winner_prob":   round(min(u["match"]["prob_home_win"], u["match"]["prob_away_win"]), 4),
            }
            for u in upsets[:n]
        ]

    def _by_confidence(self) -> Tuple[List[Dict], List[Dict]]:
        """Returns top-3 highest and bottom-3 lowest confidence matches."""
        sorted_by_conf = sorted(self.predictions, key=lambda m: -m["confidence"])
        highest = [
            {
                "match_no": m["match_no"],
                "round": m["round"],
                "match": f"{m['home_team']} vs {m['away_team']}",
                "winner": m["predicted_winner"],
                "confidence": round(m["confidence"], 4),
                "tier": classify_confidence(m["confidence"]),
            }
            for m in sorted_by_conf[:3]
        ]
        lowest = [
            {
                "match_no": m["match_no"],
                "round": m["round"],
                "match": f"{m['home_team']} vs {m['away_team']}",
                "winner": m["predicted_winner"],
                "confidence": round(m["confidence"], 4),
                "tier": classify_confidence(m["confidence"]),
            }
            for m in sorted_by_conf[-3:]
        ]
        return highest, lowest

    def _most_competitive(self) -> List[Dict[str, Any]]:
        """Returns matches with smallest win margin (most contested)."""
        def margin(m: Dict) -> float:
            p = sorted([m.get("prob_home_win", 0), m.get("prob_draw", 0), m.get("prob_away_win", 0)], reverse=True)
            return p[0] - p[1]

        contested = sorted(self.predictions, key=margin)[:5]
        return [
            {
                "match_no":  m["match_no"],
                "round":     m["round"],
                "match":     f"{m['home_team']} vs {m['away_team']}",
                "winner":    m["predicted_winner"],
                "margin":    round(margin(m), 4),
                "confidence": round(m["confidence"], 4),
            }
            for m in contested
        ]

    def _most_onesided(self) -> List[Dict[str, Any]]:
        """Returns matches with largest win margin (most dominant)."""
        def margin(m: Dict) -> float:
            p = sorted([m.get("prob_home_win", 0), m.get("prob_draw", 0), m.get("prob_away_win", 0)], reverse=True)
            return p[0] - p[1]

        onesided = sorted(self.predictions, key=margin, reverse=True)[:5]
        return [
            {
                "match_no":  m["match_no"],
                "round":     m["round"],
                "match":     f"{m['home_team']} vs {m['away_team']}",
                "winner":    m["predicted_winner"],
                "margin":    round(margin(m), 4),
                "confidence": round(m["confidence"], 4),
            }
            for m in onesided
        ]

    def _influential_features(self) -> List[Dict[str, Any]]:
        """Top-10 most influential features from ensemble importance."""
        ranked = sorted(self.global_imp.items(), key=lambda x: -x[1])[:10]
        return [
            {
                "feature":    feat,
                "label":      get_label(feat),
                "importance": round(val, 6),
            }
            for feat, val in ranked
        ]

    def _most_important_models(self) -> List[Dict[str, Any]]:
        """Returns models sorted by ensemble weight."""
        return [
            {"model": m, "weight": round(w, 4)}
            for m, w in sorted(self.ensemble_weights.items(), key=lambda x: -x[1])
        ]

    def _round_statistics(self) -> Dict[str, Any]:
        """Per-round statistics summary."""
        rounds: Dict[str, Dict] = {}
        for m in self.predictions:
            rnd = m.get("round", "Unknown")
            if rnd not in rounds:
                rounds[rnd] = {"matches": 0, "confidence_sum": 0.0, "shootouts": 0}
            rounds[rnd]["matches"] += 1
            rounds[rnd]["confidence_sum"] += m.get("confidence", 0.0)
            if m.get("shootout_played", False):
                rounds[rnd]["shootouts"] += 1

        return {
            rnd: {
                "matches": v["matches"],
                "avg_confidence": round(v["confidence_sum"] / v["matches"], 4),
                "shootouts": v["shootouts"],
            }
            for rnd, v in rounds.items()
        }

    def explain(self) -> Dict[str, Any]:
        """Runs all tournament-level explanations."""
        logger.info("Generating tournament-level XAI explanations...")

        highest_conf, lowest_conf = self._by_confidence()

        self._result = {
            "champion_explanation":       self._build_champion_explanation(),
            "runner_up_explanation":      self._build_runner_up_explanation(),
            "biggest_upsets":             self._biggest_upsets(5),
            "highest_confidence_matches": highest_conf,
            "lowest_confidence_matches":  lowest_conf,
            "most_competitive_matches":   self._most_competitive(),
            "most_onesided_matches":      self._most_onesided(),
            "most_influential_features":  self._influential_features(),
            "most_important_models":      self._most_important_models(),
            "round_statistics":           self._round_statistics(),
            "total_matches":              len(self.predictions),
            "total_rounds":               len({m.get("round") for m in self.predictions}),
        }
        return self._result

    def export(self, output_dir: str = PREDICTIONS_DIR) -> Dict[str, str]:
        """Exports tournament explanations JSON and summary CSV."""
        ensure_dir(output_dir)
        if not self._result:
            self.explain()

        # JSON
        json_path = os.path.join(output_dir, "tournament_explanations.json")
        save_json(self._result, json_path)
        logger.info(f"Exported tournament explanations JSON → {json_path}")

        # Summary CSV (flat per match)
        rows = []
        for m in self.predictions:
            highest_conf, lowest_conf = self._by_confidence()
            rows.append({
                "match_no":         m["match_no"],
                "round":            m["round"],
                "home_team":        m["home_team"],
                "away_team":        m["away_team"],
                "predicted_winner": m["predicted_winner"],
                "confidence":       round(m["confidence"], 4),
                "confidence_tier":  classify_confidence(m["confidence"]),
                "prob_home_win":    round(m.get("prob_home_win", 0.0), 4),
                "prob_draw":        round(m.get("prob_draw", 0.0), 4),
                "prob_away_win":    round(m.get("prob_away_win", 0.0), 4),
                "shootout":         m.get("shootout_played", False),
            })

        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "tournament_summary.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported tournament summary CSV → {csv_path}")

        return {"json": json_path, "csv": csv_path}


from typing import Tuple
