import os
import pandas as pd
from typing import Dict, Any, Tuple
from src.utils.logger import setup_logger

logger = setup_logger("exporter")

class DataExporter:
    def __init__(self, output_dir: str = "processed", reports_dir: str = "reports"):
        self.output_dir = output_dir
        self.reports_dir = reports_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

    def calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates various data quality metrics for the final master dataset."""
        metrics = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_values_by_column": df.isnull().sum().to_dict(),
            "missing_pct_by_column": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "unique_teams": int(pd.concat([df['home_team'], df['away_team']]).nunique()),
            "unique_tournaments": int(df['tournament'].nunique()) if 'tournament' in df.columns else 0,
            "unique_years": int(df['date'].dt.year.nunique()) if 'date' in df.columns else 0,
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }
        return metrics

    def export_master_dataset(self, df: pd.DataFrame, base_name: str = "master_dataset") -> Tuple[str, str]:
        """Saves the dataframe as both Parquet and CSV."""
        parquet_path = os.path.join(self.output_dir, f"{base_name}.parquet")
        csv_path = os.path.join(self.output_dir, f"{base_name}.csv")
        
        logger.info(f"Exporting master dataset to {parquet_path} and {csv_path}...")
        
        # Export to Parquet (primary storage)
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Parquet export complete: {os.path.getsize(parquet_path) / (1024*1024):.2f} MB")
        
        # Export to CSV (compatibility storage)
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV export complete: {os.path.getsize(csv_path) / (1024*1024):.2f} MB")
        
        return parquet_path, csv_path

    def generate_quality_report(self, df: pd.DataFrame, report_path: str):
        """Generates the README_DATA_QUALITY.md report."""
        metrics = self.calculate_quality_metrics(df)
        
        missing_table = "| Column | Missing Count | Missing Percentage |\n|---|---|---|\n"
        for col, val in metrics["missing_values_by_column"].items():
            pct = metrics["missing_pct_by_column"][col]
            if val > 0:
                missing_table += f"| `{col}` | {val:,} | {pct}% |\n"
        
        report_content = f"""# 📈 WorldCupAI — Data Quality Report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## High-Level Quality Metrics

| Metric | Value |
|---|---|
| **Total Rows** | {metrics["total_rows"]:,} |
| **Total Columns** | {metrics["total_columns"]} |
| **Duplicate Rows** | {metrics["duplicate_rows"]} |
| **Unique Teams** | {metrics["unique_teams"]:,} |
| **Unique Tournaments** | {metrics["unique_tournaments"]:,} |
| **Unique Years** | {metrics["unique_years"]} |
| **Memory Usage** | {metrics["memory_usage_mb"]} MB |

## Missing Values Analysis

{missing_table if len(missing_table.split(chr(10))) > 3 else "No missing values found in the master dataset!"}

## Data Quality Insights
- **FIFA Rankings Missingness**: Rankings columns (`home_rank`, `away_rank`) have missing values for matches played before the introduction of the official FIFA rankings in August 1993. This is normal and expected.
- **Match Features Missingness**: Missing values in ELO or EA FC attribute columns indicate matches that could not be matched with the pre-computed feature database (often due to friendly matches or minor tournaments not covered).
"""
        with open(report_path, "w") as f:
            f.write(report_content)
        logger.info(f"Data quality report written to {report_path}")
from typing import Tuple
import numpy as np
