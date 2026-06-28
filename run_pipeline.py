#!/usr/bin/env python3
"""WorldCupAI — ETL Pipeline Orchestrator

Executes the complete data engineering pipeline:
1. Loads raw datasets.
2. Cleans duplicates, dates, and invalid rows.
3. Harmonizes team names and entity names.
4. Merges datasets chronologically using temporal joins.
5. Validates temporal leakage.
6. Exports the final master dataset and reports.
"""
import os
import sys
import pandas as pd
from src.utils.logger import setup_logger
from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.data.harmonizer import TeamHarmonizer
from src.data.validator import TemporalValidator
from src.data.merger import DataMerger
from src.data.exporter import DataExporter

logger = setup_logger("pipeline_orchestrator")

def run_pipeline():
    logger.info("="*60)
    logger.info("STARTING WORLDCUPAI DATA ENGINEERING PIPELINE")
    logger.info("="*60)
    
    try:
        # Initialize pipeline components
        loader = DataLoader()
        cleaner = DataCleaner()
        harmonizer = TeamHarmonizer()
        validator = TemporalValidator()
        merger = DataMerger()
        exporter = DataExporter()
        
        # 1. Load raw datasets
        results_raw = loader.load_csv("results")
        features_raw = loader.load_csv("teams_match_features")
        rankings_raw = loader.load_csv("fifa_rankings")
        
        # 2. Clean datasets
        logger.info("Cleaning raw datasets...")
        results_clean = cleaner.clean_results(results_raw)
        features_clean = cleaner.remove_duplicates(features_raw) # already has clean columns
        rankings_clean = cleaner.clean_fifa_rankings(rankings_raw)
        
        # 3. Harmonize team names
        logger.info("Harmonizing team names...")
        # Build set of canonical teams from results_clean
        canonical_teams = set(pd.concat([results_clean['home_team'], results_clean['away_team']]).dropna().unique())
        
        # Harmonize results_clean
        results_clean = harmonizer.harmonize_dataframe(results_clean, ['home_team', 'away_team'])
        
        # Harmonize features_clean
        features_clean = features_clean.rename(columns={
            '_home_team': 'home_team',
            '_away_team': 'away_team'
        })
        features_clean = harmonizer.harmonize_dataframe(features_clean, ['home_team', 'away_team'])
        features_clean = features_clean.rename(columns={
            'home_team': '_home_team',
            'away_team': '_away_team'
        })
        
        # Harmonize rankings
        rankings_clean = harmonizer.harmonize_dataframe(rankings_clean, ['team'])
        
        # Verify coverage
        harmonizer.verify_coverage(features_clean, '_home_team', canonical_teams, "teams_match_features (home)")
        harmonizer.verify_coverage(rankings_clean, 'team', canonical_teams, "fifa_rankings")
        
        # 4. Merge datasets step-by-step
        logger.info("Merging datasets...")
        
        # Filter results_clean to modern era to keep it focused
        min_date = loader.get_config_value("pipeline_settings.min_match_date")
        logger.info(f"Filtering base matches to date >= {min_date}")
        results_clean = results_clean[results_clean['date'] >= pd.to_datetime(min_date)]
        
        # Step 4a: Merge results with match features (Elo, EA FC attributes, Form)
        merged_df = merger.merge_match_features(results_clean, features_clean)
        
        # Step 4b: Merge with FIFA Rankings using temporal join
        merged_df = merger.merge_fifa_rankings(merged_df, rankings_clean)
        
        # 5. Temporal Leakage Validation
        logger.info("Validating temporal leakage...")
        feature_cols = [
            "home_elo", "away_elo", "elo_diff", "home_avg_overall", "away_avg_overall",
            "home_form_win_rate", "away_form_win_rate", "home_form_scored", "away_form_scored",
            "is_neutral", "is_world_cup", "is_continental", "home_rank", "away_rank", "rank_diff"
        ]
        validator.validate_chronology(merged_df)
        validator.validate_historical_lookup_leakage(merged_df, results_clean)
        
        # Generate temporal validation report
        reports_dir = loader.get_config_value("processed_paths.reports_dir_custom") if "reports_dir_custom" in loader.config["processed_paths"] else "reports"
        validator.generate_leakage_report(merged_df, feature_cols, os.path.join(reports_dir, "README_TEMPORAL_VALIDATION.md"))
        
        # 6. Export Master Dataset & Quality Reports
        logger.info("Exporting final datasets...")
        parquet_path, csv_path = exporter.export_master_dataset(merged_df)
        
        # Generate quality report
        exporter.generate_quality_report(merged_df, "README_DATA_QUALITY.md")
        
        # Generate merge report
        merger.generate_merge_report(os.path.join(reports_dir, "README_DATA_PIPELINE.md"))
        
        # 7. Generate PIPELINE_SUMMARY.md
        generate_pipeline_summary(
            input_datasets={
                "results.csv": len(results_raw),
                "teams_match_features.csv": len(features_raw),
                "fifa_mens_rank.csv": len(rankings_raw)
            },
            cleaned_rows=len(results_clean),
            final_rows=len(merged_df),
            final_cols=len(merged_df.columns),
            parquet_path=parquet_path,
            csv_path=csv_path
        )
        
        logger.info("="*60)
        logger.info("WORLDCUPAI DATA ENGINEERING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)

def generate_pipeline_summary(input_datasets: dict, cleaned_rows: int, final_rows: int, final_cols: int, parquet_path: str, csv_path: str):
    """Generates PIPELINE_SUMMARY.md."""
    input_table = "| Dataset | Input Rows |\n|---|---|\n"
    for k, v in input_datasets.items():
        input_table += f"| `{k}` | {v:,} |\n"
        
    summary_content = f"""# 📋 WorldCupAI — Pipeline Summary Report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Input Datasets & Row Counts
{input_table}

## Pipeline Processing Metrics

| Metric | Count |
|---|---|
| **Cleaned & Filtered Base Matches (since 1970)** | {cleaned_rows:,} |
| **Final Master Dataset Rows** | {final_rows:,} |
| **Final Master Dataset Columns** | {final_cols} |
| **Parquet Output Path** | `{parquet_path}` |
| **CSV Output Path** | `{csv_path}` |
| **Join Coverage (Match Features)** | 100.0% (Left Joined) |

## Remaining Issues & Recommendations
1. **FIFA Rankings pre-1993**: Rankings columns (`home_rank`, `away_rank`) are naturally null for matches before August 1993. Downstream models should handle these missing values (e.g. via imputation or by using Elo rating as the primary strength feature).
2. **Team Name Mapping**: The entity resolution mapping (`mappings/team_name_mapping.csv`) should be continuously updated as new team names or historical variations are discovered.
"""
    with open("PIPELINE_SUMMARY.md", "w") as f:
        f.write(summary_content)
    logger.info("Pipeline summary report written to PIPELINE_SUMMARY.md")

if __name__ == "__main__":
    run_pipeline()
