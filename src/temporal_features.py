"""Temporal signal engineering module.

Calculates time-series features for each product and period:
- Complaint frequency and proportion
- Average complaint severity
- Complaint growth rate
- Review arrival velocity
- Symptom diversity
- Historical baselines and deviations

All features are defined mathematically in the README.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ASPECT_LEXICON, EWMA_SPAN


def calculate_period_features(
    df: pd.DataFrame,
    aspects: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Calculate temporal features for each period in the DataFrame.
    
    Expects a DataFrame with columns: period, review_text, aspects (list),
    combined_severity.
    
    Args:
        df: DataFrame with period column and analysis results.
        aspects: List of aspect names to track. Uses config default if None.
        
    Returns:
        DataFrame with one row per period and computed features.
    """
    if aspects is None:
        aspects = list(ASPECT_LEXICON.keys())
    
    if df.empty:
        return _empty_features_df(aspects)
    
    periods = sorted(df["period"].unique())
    rows = []
    
    for period in periods:
        period_df = df[df["period"] == period]
        row = _compute_period_row(period_df, period, aspects)
        rows.append(row)
    
    features_df = pd.DataFrame(rows)
    
    # Calculate growth rates (requires sequential periods)
    features_df = _add_growth_rates(features_df, aspects)
    
    # Calculate EWMA baselines
    features_df = _add_ewma_baselines(features_df, aspects)
    
    # Calculate deviations from baseline
    features_df = _add_deviations(features_df, aspects)
    
    return features_df


def _compute_period_row(
    period_df: pd.DataFrame,
    period,
    aspects: List[str],
) -> Dict:
    """Compute features for a single period."""
    total_reviews = len(period_df)
    
    row = {
        "period": period,
        "total_reviews": total_reviews,
        "review_velocity": total_reviews,  # Reviews per period
    }
    
    # Count complaints per aspect
    all_aspects_in_period = set()
    for aspect in aspects:
        if "aspects" in period_df.columns:
            aspect_mask = period_df["aspects"].apply(
                lambda x: aspect in x if isinstance(x, list) else False
            )
        else:
            aspect_mask = pd.Series([False] * len(period_df))
        
        count = int(aspect_mask.sum())
        row[f"complaint_count_{aspect}"] = count
        
        # Complaint proportion (avoid division by zero)
        proportion = count / total_reviews if total_reviews > 0 else 0.0
        row[f"complaint_proportion_{aspect}"] = proportion
        
        # Average severity for this aspect
        if count > 0 and "combined_severity" in period_df.columns:
            avg_severity = period_df.loc[aspect_mask, "combined_severity"].mean()
        else:
            avg_severity = 0.0
        row[f"avg_severity_{aspect}"] = avg_severity
        
        if count > 0:
            all_aspects_in_period.add(aspect)
    
    # 1. Total aspect mentions across all aspects in this period
    total_aspect_mentions = sum(row.get(f"complaint_count_{a}", 0) for a in aspects)
    row["total_aspect_mentions"] = total_aspect_mentions
    row["total_complaints"] = total_aspect_mentions  # Retained for backward compatibility
    row["aspect_mentions_per_review"] = total_aspect_mentions / total_reviews if total_reviews > 0 else 0.0

    # 2. Count of unique reviews containing at least one complaint
    # A review mentioning multiple complaint aspects is counted exactly ONCE here.
    if "is_complaint" in period_df.columns:
        complaint_reviews_count = int(period_df["is_complaint"].sum())
    elif "aspects" in period_df.columns:
        complaint_reviews_count = int(period_df["aspects"].apply(
            lambda x: len(x) > 0 if isinstance(x, list) else False
        ).sum())
    else:
        complaint_reviews_count = 0

    row["complaint_reviews_count"] = complaint_reviews_count

    # 3. Overall complaint-review proportion (strictly bounded in [0.0, 1.0], never exceeds 100%)
    overall_complaint_proportion = complaint_reviews_count / total_reviews if total_reviews > 0 else 0.0
    row["overall_complaint_proportion"] = min(1.0, max(0.0, overall_complaint_proportion))
    
    # Overall average severity
    if "combined_severity" in period_df.columns:
        row["overall_avg_severity"] = period_df["combined_severity"].mean()
    else:
        row["overall_avg_severity"] = 0.0
    
    # Symptom diversity: number of distinct aspects / total supported aspects
    row["symptom_diversity"] = len(all_aspects_in_period) / len(aspects) if len(aspects) > 0 else 0.0
    row["num_distinct_aspects"] = len(all_aspects_in_period)
    
    return row


def _add_growth_rates(df: pd.DataFrame, aspects: List[str]) -> pd.DataFrame:
    """Add complaint growth rates comparing each period to the previous one.
    
    Growth = (current_proportion - previous_proportion) / max(previous_proportion, min_baseline)
    
    We use proportions rather than raw counts to avoid interpreting
    increased review volume as product deterioration.
    Using max(previous_proportion, 0.05) ensures safe handling of zero denominators
    without generating astronomical growth values (e.g. 333,333%).
    """
    min_baseline = 0.05
    
    # Overall complaint growth
    df["complaint_growth"] = 0.0
    for i in range(1, len(df)):
        prev = df.iloc[i-1]["overall_complaint_proportion"]
        curr = df.iloc[i]["overall_complaint_proportion"]
        if curr == 0.0 and prev == 0.0:
            df.loc[df.index[i], "complaint_growth"] = 0.0
        else:
            denom = max(prev, min_baseline)
            df.loc[df.index[i], "complaint_growth"] = (curr - prev) / denom
    
    # Per-aspect growth
    for aspect in aspects:
        col = f"complaint_proportion_{aspect}"
        growth_col = f"growth_{aspect}"
        df[growth_col] = 0.0
        for i in range(1, len(df)):
            prev = df.iloc[i-1][col]
            curr = df.iloc[i][col]
            if curr == 0.0 and prev == 0.0:
                df.loc[df.index[i], growth_col] = 0.0
            else:
                denom = max(prev, min_baseline)
                df.loc[df.index[i], growth_col] = (curr - prev) / denom
    
    return df


def _add_ewma_baselines(df: pd.DataFrame, aspects: List[str]) -> pd.DataFrame:
    """Add EWMA baselines for key metrics."""
    span = EWMA_SPAN
    
    # Overall complaint proportion baseline
    df["baseline_complaint_proportion"] = (
        df["overall_complaint_proportion"]
        .ewm(span=span, min_periods=1)
        .mean()
        .shift(1)
        .fillna(0)
    )
    
    # Overall severity baseline
    df["baseline_severity"] = (
        df["overall_avg_severity"]
        .ewm(span=span, min_periods=1)
        .mean()
        .shift(1)
        .fillna(0)
    )
    
    # Review velocity baseline
    df["baseline_velocity"] = (
        df["review_velocity"]
        .ewm(span=span, min_periods=1)
        .mean()
        .shift(1)
        .fillna(df["review_velocity"].iloc[0] if len(df) > 0 else 0)
    )
    
    return df


def _add_deviations(df: pd.DataFrame, aspects: List[str]) -> pd.DataFrame:
    """Add deviation from historical baselines."""
    epsilon = 1e-6
    
    df["deviation_complaint_proportion"] = (
        df["overall_complaint_proportion"] - df["baseline_complaint_proportion"]
    ) / (df["baseline_complaint_proportion"] + epsilon)
    
    df["deviation_severity"] = (
        df["overall_avg_severity"] - df["baseline_severity"]
    ) / (df["baseline_severity"] + epsilon)
    
    df["deviation_velocity"] = (
        df["review_velocity"] - df["baseline_velocity"]
    ) / (df["baseline_velocity"] + epsilon)
    
    return df


def _empty_features_df(aspects: List[str]) -> pd.DataFrame:
    """Return empty features DataFrame with correct columns."""
    cols = ["period", "total_reviews", "review_velocity", "total_complaints",
            "total_aspect_mentions", "complaint_reviews_count", "aspect_mentions_per_review",
            "overall_complaint_proportion", "overall_avg_severity",
            "symptom_diversity", "num_distinct_aspects", "complaint_growth"]
    for aspect in aspects:
        cols.extend([
            f"complaint_count_{aspect}",
            f"complaint_proportion_{aspect}",
            f"avg_severity_{aspect}",
            f"growth_{aspect}",
        ])
    cols.extend(["baseline_complaint_proportion", "baseline_severity",
                 "baseline_velocity", "deviation_complaint_proportion",
                 "deviation_severity", "deviation_velocity"])
    return pd.DataFrame(columns=cols)
