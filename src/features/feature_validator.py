import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.utils.logger import setup_logger

logger = setup_logger("feature_validator")

class FeatureValidator:
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

    def validate_features(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Calculates quality metrics for all features: missing, cardinality, variance, outliers."""
        logger.info(f"Validating {len(feature_cols)} features...")
        
        report = {}
        for col in feature_cols:
            series = df[col]
            
            # Handle numeric vs non-numeric
            is_numeric = np.issubdtype(series.dtype, np.number)
            
            missing_count = int(series.isnull().sum())
            missing_pct = round(missing_count / len(df) * 100, 2)
            unique_count = int(series.nunique())
            
            if is_numeric:
                variance = float(series.var())
                mean = float(series.mean())
                std = float(series.std())
                
                # Outlier detection (using IQR method)
                q25 = series.quantile(0.25)
                q75 = series.quantile(0.75)
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                outliers = int(((series < lower_bound) | (series > upper_bound)).sum())
                
                report[col] = {
                    "type": str(series.dtype),
                    "missing_count": missing_count,
                    "missing_pct": missing_pct,
                    "unique_count": unique_count,
                    "variance": round(variance, 4),
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "outliers": outliers,
                    "constant": bool(variance == 0 or unique_count <= 1)
                }
            else:
                report[col] = {
                    "type": str(series.dtype),
                    "missing_count": missing_count,
                    "missing_pct": missing_pct,
                    "unique_count": unique_count,
                    "variance": None,
                    "mean": None,
                    "std": None,
                    "outliers": None,
                    "constant": bool(unique_count <= 1)
                }
                
        return report

    def generate_validation_report(self, df: pd.DataFrame, feature_cols: List[str], output_path: str):
        """Generates README_FEATURE_VALIDATION.md."""
        report_data = self.validate_features(df, feature_cols)
        
        table = "| Feature Name | Type | Missing % | Unique Values | Variance | Outliers | Constant? |\n|---|---|---|---|---|---|---|\n"
        for col, metrics in report_data.items():
            var_str = f"{metrics['variance']}" if metrics['variance'] is not None else "N/A"
            out_str = f"{metrics['outliers']:,}" if metrics['outliers'] is not None else "N/A"
            table += f"| `{col}` | {metrics['type']} | {metrics['missing_pct']}% | {metrics['unique_count']:,} | {var_str} | {out_str} | {'Yes ⚠️' if metrics['constant'] else 'No'} |\n"
            
        report_content = f"""# 📊 WorldCupAI — Feature Validation Report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## High-Level Summary
- **Total Features Validated**: {len(feature_cols)}
- **Constant Features**: {sum(1 for m in report_data.values() if m['constant'])}
- **High Missing Value Features (>10%)**: {sum(1 for m in report_data.values() if m['missing_pct'] > 10.0)}

## Detailed Feature Metrics
{table}

## Quality Analysis
- **Rankings Missingness**: FIFA rank features (`home_rank`, `away_rank`) naturally contain missing values for historical matches before 1993. This is expected.
- **Outliers**: Outliers are present in rolling rest days and goal differences. These represent extreme congestion or high-scoring matches and should be handled by robust scaling in the ML phase.
"""
        with open(output_path, "w") as f:
            f.write(report_content)
        logger.info(f"Feature validation report written to {output_path}")
