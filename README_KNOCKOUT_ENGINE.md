# ⚙️ WorldCupAI — Knockout Engine Reference

This module constructs dynamic pre-match features, runs soft-voting predictions, progress teams through the tournament tree, and implements deterministic draw resolution.

## Key Design Principles
1. **Fixture Isolation**: Read from `configs/knockout_fixtures.json`.
2. **Rest Days Calculation**: Dynamically calculated based on matches played in the tournament.
3. **Draw Resolution**: Strict deterministic shootout fallback based on pre-match statistics (Elo, FIFA Rank).
