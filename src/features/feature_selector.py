import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from src.utils.logger import setup_logger

logger = setup_logger("feature_selector")

class FeatureSelector:
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

    def analyze_correlation(self, df: pd.DataFrame, feature_cols: List[str], threshold: float = 0.9) -> List[Tuple[str, str, float]]:
        """Identifies pairs of features with correlation above the threshold."""
        logger.info(f"Analyzing collinearity with threshold {threshold}...")
        
        # Keep only numeric columns
        numeric_cols = [col for col in feature_cols if np.issubdtype(df[col].dtype, np.number)]
        corr_matrix = df[numeric_cols].corr().abs()
        
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        high_corr_pairs = []
        for col in upper.columns:
            matches = upper[col][upper[col] > threshold]
            for index, val in matches.items():
                high_corr_pairs.append((index, col, float(val)))
                
        logger.info(f"Found {len(high_corr_pairs)} highly correlated feature pairs.")
        return high_corr_pairs

    def analyze_mutual_info(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
        """Calculates Mutual Information scores between features and the match outcome."""
        logger.info("Calculating Mutual Information scores...")
        
        # Define target variable: 1 if home win, 0 if draw, -1 if away win (or simple binary home win)
        # Let's use a binary target: home_score > away_score
        target = (df['home_score'] > df['away_score']).astype(int)
        
        # Fill NaNs for MI calculation
        X = df[feature_cols].copy()
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median() if np.issubdtype(X[col].dtype, np.number) else "missing")
                
        # One-hot encode non-numeric columns
        X = pd.get_dummies(X, drop_first=True)
        
        try:
            from sklearn.feature_selection import mutual_info_classif
            mi_scores = mutual_info_classif(X, target, random_state=42)
            
            # Map back to original feature names (taking max if one-hot encoded)
            scores = {}
            for i, col in enumerate(X.columns):
                # Find original column name
                orig_col = col
                for fc in feature_cols:
                    if col.startswith(fc):
                        orig_col = fc
                        break
                scores[orig_col] = max(scores.get(orig_col, 0.0), float(mi_scores[i]))
                
            return scores
        except ImportError:
            logger.warning("scikit-learn is not installed. Skipping Mutual Information calculation.")
            return {col: 0.0 for col in feature_cols}

    def analyze_tree_importance(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
        """Runs a quick Random Forest classifier to get baseline feature importances."""
        logger.info("Calculating Tree-based feature importances...")
        
        target = (df['home_score'] > df['away_score']).astype(int)
        
        X = df[feature_cols].copy()
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median() if np.issubdtype(X[col].dtype, np.number) else "missing")
                
        X = pd.get_dummies(X, drop_first=True)
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
            rf.fit(X, target)
            
            importances = {}
            for i, col in enumerate(X.columns):
                orig_col = col
                for fc in feature_cols:
                    if col.startswith(fc):
                        orig_col = fc
                        break
                importances[orig_col] = max(importances.get(orig_col, 0.0), float(rf.feature_importances_[i]))
                
            return importances
        except ImportError:
            logger.warning("scikit-learn is not installed. Skipping Tree-based importance calculation.")
            return {col: 0.0 for col in feature_cols}

    def generate_selection_report(self, df: pd.DataFrame, feature_cols: List[str], output_path: str):
        """Generates FEATURE_SELECTION_REPORT.md."""
        high_corr = self.analyze_correlation(df, feature_cols)
        mi_scores = self.analyze_mutual_info(df, feature_cols)
        tree_imp = self.analyze_tree_importance(df, feature_cols)
        
        # Sort features by tree importance
        sorted_features = sorted(feature_cols, key=lambda c: tree_imp.get(c, 0.0), reverse=True)
        
        # Decide Keep/Review/Discard
        # We will keep features with tree importance > 0.005
        # We will mark features with high correlation as "Review" or "Discard"
        collinear_features = set()
        for f1, f2, val in high_corr:
            # Discard the one with lower tree importance
            discarded = f2 if tree_imp.get(f1, 0.0) >= tree_imp.get(f2, 0.0) else f1
            collinear_features.add(discarded)
            
        table = "| Feature Name | Mutual Information | Tree Importance | Recommendation | Rationale |\n|---|---|---|---|---|\n"
        for col in sorted_features:
            mi = mi_scores.get(col, 0.0)
            ti = tree_imp.get(col, 0.0)
            
            if col in collinear_features:
                rec = "Discard ❌"
                rat = "High collinearity with another feature."
            elif ti < 0.002:
                rec = "Review ⚠️"
                rat = "Very low predictive importance in baseline tree."
            else:
                rec = "Keep 🟢"
                rat = "Strong predictive power and low collinearity."
                
            table += f"| `{col}` | {mi:.4f} | {ti:.4f} | {rec} | {rat} |\n"
            
        corr_table = "| Feature 1 | Feature 2 | Correlation |\n|---|---|---|\n"
        for f1, f2, val in high_corr:
            corr_table += f"| `{f1}` | `{f2}` | {val:.4f} |\n"
            
        report_content = f"""# 🔍 WorldCupAI — Feature Selection Report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Features Analyzed**: {len(feature_cols)}
- **Features Retained (Keep)**: {sum(1 for col in feature_cols if col not in collinear_features and tree_imp.get(col, 0.0) >= 0.002)}
- **Features Flagged for Review**: {sum(1 for col in feature_cols if col not in collinear_features and tree_imp.get(col, 0.0) < 0.002)}
- **Features Discarded (Collinear)**: {len(collinear_features)}

## Collinear Feature Pairs (>0.9 Correlation)
{corr_table if len(high_corr) > 0 else "No highly collinear feature pairs found."}

## Feature Recommendations & Rankings
{table}
"""
        with open(output_path, "w") as f:
            f.write(report_content)
        logger.info(f"Feature selection report written to {output_path}")
