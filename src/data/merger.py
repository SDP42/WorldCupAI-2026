import os
import pandas as pd
import numpy as np
from typing import Dict, Any
from src.utils.logger import setup_logger

logger = setup_logger("merger")

class DataMerger:
    def __init__(self):
        self.merge_log = []

    def log_merge_step(self, step_name: str, df_before: pd.DataFrame, df_after: pd.DataFrame, join_cols: list):
        rows_before = len(df_before)
        rows_after = len(df_after)
        missing_count = df_after.isnull().sum().sum()
        
        # Calculate coverage (percentage of rows in df_before that successfully matched)
        # For left joins, the row count remains same unless there are duplicate keys in right df.
        coverage = (rows_after / rows_before) * 100 if rows_before > 0 else 0
        
        step_info = {
            "step": step_name,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "change": rows_after - rows_before,
            "coverage_pct": round(coverage, 2),
            "join_keys": join_cols,
            "total_missing_values": missing_count
        }
        self.merge_log.append(step_info)
        logger.info(f"[{step_name}] Merged using keys {join_cols}. Rows: {rows_before:,} -> {rows_after:,} (Coverage: {coverage:.2f}%)")

    def merge_match_features(self, results_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """Joins results.csv with teams_match_features.csv on date and team names."""
        results_df = results_df.copy()
        features_df = features_df.copy()
        
        # Rename features columns to match results.csv join keys
        features_df = features_df.rename(columns={
            '_date': 'date',
            '_home_team': 'home_team',
            '_away_team': 'away_team',
            '_tournament': 'tournament'
        })
        
        # Ensure date types match
        results_df['date'] = pd.to_datetime(results_df['date'])
        features_df['date'] = pd.to_datetime(features_df['date'])
        
        # Deduplicate features_df on join keys to prevent duplication in left join
        features_df = features_df.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='first')
        
        # Perform left join
        merged = pd.merge(
            results_df,
            features_df.drop(columns=['tournament', 'home_goals', 'away_goals'], errors='ignore'),
            on=['date', 'home_team', 'away_team'],
            how='left'
        )
        
        self.log_merge_step("Match Features Join", results_df, merged, ['date', 'home_team', 'away_team'])
        return merged

    def merge_fifa_rankings(self, df: pd.DataFrame, rankings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs a temporal ASOF join to get the latest FIFA ranking for home and away teams
        immediately prior to the match date. This prevents temporal leakage.
        """
        df = df.copy()
        rankings_df = rankings_df.copy()
        
        df['date'] = pd.to_datetime(df['date']).astype('datetime64[ns]')
        rankings_df['date'] = pd.to_datetime(rankings_df['date']).astype('datetime64[ns]')
        
        # Sort both dataframes by date (required for merge_asof)
        df = df.sort_values('date')
        rankings_df = rankings_df.sort_values('date')
        
        # We will do two separate asof merges: one for home team, one for away team
        # 1. Home team ranking
        home_rankings = rankings_df.rename(columns={
            'team': 'home_team',
            'rank': 'home_rank',
            'total.points': 'home_fifa_points'
        })[['date', 'home_team', 'home_rank', 'home_fifa_points']]
        
        merged = pd.merge_asof(
            df,
            home_rankings,
            on='date',
            by='home_team',
            direction='backward'  # Get the latest ranking on or before the match date
        )
        
        # 2. Away team ranking
        away_rankings = rankings_df.rename(columns={
            'team': 'away_team',
            'rank': 'away_rank',
            'total.points': 'away_fifa_points'
        })[['date', 'away_team', 'away_rank', 'away_fifa_points']]
        
        merged = pd.merge_asof(
            merged,
            away_rankings,
            on='date',
            by='away_team',
            direction='backward'
        )
        
        # Calculate rank difference
        merged['rank_diff'] = merged['home_rank'] - merged['away_rank']
        
        self.log_merge_step("FIFA Rankings Temporal Join", df, merged, ['date', 'home_team/away_team'])
        return merged

    def generate_merge_report(self, report_path: str):
        """Generates a markdown report summarizing the merge steps and coverage."""
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        report_content = f"""# 🔗 Data Pipeline Merge Report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Merge Steps & Join Coverage

| Step Name | Rows Before | Rows After | Row Change | Coverage | Join Keys | Total Missing Cells |
|---|---|---|---|---|---|---|
"""
        for log in self.merge_log:
            report_content += f"| **{log['step']}** | {log['rows_before']:,} | {log['rows_after']:,} | {log['change']:,} | {log['coverage_pct']}% | `{log['join_keys']}` | {log['total_missing_values']:,} |\n"
            
        report_content += """
## Technical Analysis & Recommendations
- The **Match Features Join** maps historical results to Elo and EA FC player attributes. Any unmatched matches will have NaN values for these features.
- The **FIFA Rankings Temporal Join** uses a backward `merge_asof` to lookup rankings. Matches played before the first official FIFA ranking date (August 1993) will naturally have NaN rankings. This is expected behavior and must be handled by downstream models.
"""
        with open(report_path, "w") as f:
            f.write(report_content)
        logger.info(f"Merge report written to {report_path}")
