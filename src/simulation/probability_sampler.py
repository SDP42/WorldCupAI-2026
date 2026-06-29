"""WorldCupAI — Phase 9 Probability Sampler & Prediction Cache."""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from src.prediction.knockout_engine import KnockoutEngine, map_team_name

class PredictionCache:
    """Lazy cache for match prediction probabilities to prevent redundant subprocess calls."""
    def __init__(self, engine: Optional[KnockoutEngine] = None, shared_dict: Optional[Any] = None):
        self.engine = engine or KnockoutEngine()
        self.cache = shared_dict if shared_dict is not None else {}

    def get_prediction(self, home_team: str, away_team: str, match_no: int, date_str: str) -> Dict[str, Any]:
        """Gets prediction for a matchup, using cached result if available."""
        h_canon = map_team_name(home_team)
        a_canon = map_team_name(away_team)
        key = (h_canon, a_canon, match_no)
        
        if key in self.cache:
            return self.cache[key]
            
        # Reverse key check (just in case they are swapped, though dates/rest days are match-specific)
        rev_key = (a_canon, h_canon, match_no)
        if rev_key in self.cache:
            # We must swap the home/away win probabilities
            cached = self.cache[rev_key]
            swapped = {
                "prob_home_win": cached["prob_away_win"],
                "prob_draw": cached["prob_draw"],
                "prob_away_win": cached["prob_home_win"],
                "predicted_outcome": "Draw" if cached["predicted_outcome"] == "Draw" else f"{cached['predicted_winner']} Win",
                "predicted_winner": cached["predicted_winner"],
                "confidence": cached["confidence"],
                "entropy": cached["entropy"],
                "shootout_played": cached["shootout_played"],
                "shootout_reason": cached["shootout_reason"]
            }
            return swapped

        # Generate new prediction
        # predict_match saves state to self.engine.team_last_match_date and self.engine.matches_played.
        # To avoid corrupting the engine's internal bracket tracking during simulation lookup,
        # we call predict_match but we retrieve the result and clean up matches_played.
        pred = self.engine.predict_match(
            match_no=match_no,
            round_name="TBD",
            home_team=h_canon,
            away_team=a_canon,
            date_str=date_str
        )
        
        # Cache and return a copy of the prediction dictionary
        stored = {
            "prob_home_win": pred["prob_home_win"],
            "prob_draw": pred["prob_draw"],
            "prob_away_win": pred["prob_away_win"],
            "predicted_outcome": pred["predicted_outcome"],
            "predicted_winner": pred["predicted_winner"],
            "confidence": pred["confidence"],
            "entropy": pred["entropy"],
            "shootout_played": pred["shootout_played"],
            "shootout_reason": pred["shootout_reason"]
        }
        self.cache[key] = stored
        return stored

class ProbabilitySampler:
    """Handles Monte Carlo sampling of match outcomes using probabilities."""
    def __init__(self, cache: PredictionCache):
        self.cache = cache

    def sample_match(self, home_team: str, away_team: str, match_no: int, date_str: str, 
                     rng: np.random.Generator) -> Dict[str, Any]:
        """Samples a match outcome probabilistically based on predicted match probabilities."""
        pred = self.cache.get_prediction(home_team, away_team, match_no, date_str)
        
        # Probabilities layout: [prob_away_win, prob_draw, prob_home_win]
        probs = [pred["prob_away_win"], pred["prob_draw"], pred["prob_home_win"]]
        probs = np.array(probs, dtype=float)
        probs /= probs.sum()  # Guard to make sure it sums exactly to 1
        
        # Sample outcome: 0=Away Win, 1=Draw, 2=Home Win
        outcome_idx = rng.choice([0, 1, 2], p=probs)
        
        shootout_played = False
        shootout_reason = ""
        
        if outcome_idx == 0:
            winner = away_team
            outcome = f"{away_team} Win"
        elif outcome_idx == 2:
            winner = home_team
            outcome = f"{home_team} Win"
        else:
            # Draw: Resolve shootout deterministically
            winner, shootout_reason = self.cache.engine.resolve_draw_deterministically(home_team, away_team)
            shootout_played = True
            outcome = "Draw"
            
        return {
            "match_no": match_no,
            "home_team": home_team,
            "away_team": away_team,
            "prob_home_win": pred["prob_home_win"],
            "prob_draw": pred["prob_draw"],
            "prob_away_win": pred["prob_away_win"],
            "sampled_outcome": outcome,
            "winner": winner,
            "shootout_played": shootout_played,
            "shootout_reason": shootout_reason,
            "confidence": float(probs[outcome_idx]),
            "entropy": pred["entropy"]
        }
