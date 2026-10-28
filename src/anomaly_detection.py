"""Anomaly detection and early warning module.

Implements EWMA-based and rolling baseline deviation methods to detect
abnormal increases in complaint signals.

Key principles:
- Never uses future observations for historical warnings
- Requires sufficient history before high-confidence warnings
- Configurable thresholds
- Generates evidence-based explanations
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ANOMALY_THRESHOLD,
    MIN_HISTORY_PERIODS,
    MIN_REVIEWS_FOR_WARNING,
    EWMA_SPAN,
    ASPECT_LEXICON,
    MIN_STD_FLOOR,
    MIN_ABSOLUTE_INCREASE,
)


def detect_anomalies(
    features_df: pd.DataFrame,
    threshold: float = ANOMALY_THRESHOLD,
    min_history: int = MIN_HISTORY_PERIODS,
    min_reviews: int = MIN_REVIEWS_FOR_WARNING,
    min_std_floor: float = MIN_STD_FLOOR,
    min_absolute_increase: float = MIN_ABSOLUTE_INCREASE,
) -> pd.DataFrame:
    """Detect anomalous periods using rolling baseline deviation.
    
    Method: For each period after min_history periods, compare the current
    signals against the historical mean and standard deviation computed strictly
    from prior periods (no future leakage).
    
    To guard against tiny-sample instability and zero variance, a minimum standard
    deviation floor (min_std_floor) and minimum absolute increase (min_absolute_increase)
    are applied.
    
    Confidence is explicitly documented as a heuristic measure based on cumulative
    review sample size, not an empirically calibrated statistical probability.
    """
    if features_df.empty:
        return features_df
    
    df = features_df.copy()
    n = len(df)
    
    # Initialize anomaly columns
    df["is_anomaly"] = False
    df["anomaly_score"] = 0.0
    df["anomaly_signals"] = [[] for _ in range(n)]
    df["confidence"] = "insufficient_data"
    
    for i in range(n):
        if i < min_history:
            continue
        
        # Historical window: strictly all periods BEFORE current (no future leakage)
        history = df.iloc[:i]
        current = df.iloc[i]
        
        # Check cumulative review count
        cumulative_reviews = int(history["total_reviews"].sum())
        
        signals_triggered = []
        max_deviation = 0.0
        
        # 1. Check complaint proportion
        hist_mean = history["overall_complaint_proportion"].mean()
        hist_std = history["overall_complaint_proportion"].std()
        effective_std = max(hist_std if pd.notna(hist_std) else 0.0, min_std_floor)
        diff_prop = current["overall_complaint_proportion"] - hist_mean
        if diff_prop >= min_absolute_increase:
            z_score = diff_prop / effective_std
            if z_score > threshold:
                signals_triggered.append({
                    "signal": "complaint_proportion",
                    "current": round(float(current["overall_complaint_proportion"]), 4),
                    "baseline": round(float(hist_mean), 4),
                    "z_score": round(float(z_score), 2),
                    "method": "rolling_baseline_deviation",
                })
                max_deviation = max(max_deviation, z_score)
        
        # 2. Check severity
        hist_mean_sev = history["overall_avg_severity"].mean()
        hist_std_sev = history["overall_avg_severity"].std()
        effective_std_sev = max(hist_std_sev if pd.notna(hist_std_sev) else 0.0, min_std_floor)
        diff_sev = current["overall_avg_severity"] - hist_mean_sev
        if diff_sev >= min_absolute_increase:
            z_sev = diff_sev / effective_std_sev
            if z_sev > threshold:
                signals_triggered.append({
                    "signal": "severity",
                    "current": round(float(current["overall_avg_severity"]), 4),
                    "baseline": round(float(hist_mean_sev), 4),
                    "z_score": round(float(z_sev), 2),
                    "method": "rolling_baseline_deviation",
                })
                max_deviation = max(max_deviation, z_sev)
        
        # 3. Check symptom diversity
        hist_mean_div = history["symptom_diversity"].mean()
        hist_std_div = history["symptom_diversity"].std()
        effective_std_div = max(hist_std_div if pd.notna(hist_std_div) else 0.0, min_std_floor)
        diff_div = current["symptom_diversity"] - hist_mean_div
        if diff_div >= min_absolute_increase:
            z_div = diff_div / effective_std_div
            if z_div > threshold:
                signals_triggered.append({
                    "signal": "symptom_diversity",
                    "current": round(float(current["symptom_diversity"]), 4),
                    "baseline": round(float(hist_mean_div), 4),
                    "z_score": round(float(z_div), 2),
                    "method": "rolling_baseline_deviation",
                })
                max_deviation = max(max_deviation, z_div)
        
        # 4. Check review velocity
        hist_mean_vel = history["review_velocity"].mean()
        hist_std_vel = history["review_velocity"].std()
        effective_std_vel = max(hist_std_vel if pd.notna(hist_std_vel) else 0.0, 1.0)
        diff_vel = current["review_velocity"] - hist_mean_vel
        if diff_vel >= 1.0:
            z_vel = diff_vel / effective_std_vel
            if z_vel > threshold:
                signals_triggered.append({
                    "signal": "review_velocity",
                    "current": round(float(current["review_velocity"]), 1),
                    "baseline": round(float(hist_mean_vel), 1),
                    "z_score": round(float(z_vel), 2),
                    "method": "rolling_baseline_deviation",
                })
                max_deviation = max(max_deviation, z_vel)
        
        if signals_triggered:
            df.at[df.index[i], "is_anomaly"] = True
            df.at[df.index[i], "anomaly_score"] = round(float(max_deviation), 2)
            df.at[df.index[i], "anomaly_signals"] = signals_triggered
            
            # Explicitly label confidence as heuristic based on sample size
            if cumulative_reviews >= min_reviews:
                df.at[df.index[i], "confidence"] = "heuristic (adequate sample)"
            else:
                df.at[df.index[i], "confidence"] = "heuristic (small sample)"
        else:
            df.at[df.index[i], "confidence"] = "normal"
    
    return df


def get_anomaly_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only anomalous periods from the results."""
    if "is_anomaly" not in df.columns:
        return pd.DataFrame()
    return df[df["is_anomaly"] == True].copy()


def get_top_anomaly_aspects(
    features_df: pd.DataFrame,
    period_idx: int,
    aspects: Optional[List[str]] = None,
    top_n: int = 3,
) -> List[Dict]:
    """Find top aspects in an anomalous period.
    
    Aspects are sorted primarily by complaint count and proportion in the
    period, ensuring prominent complaints (like Heating and Battery) are
    accurately highlighted.
    """
    if aspects is None:
        aspects = list(ASPECT_LEXICON.keys())
    
    if period_idx >= len(features_df) or period_idx < 0:
        return []
    
    row = features_df.iloc[period_idx]
    aspect_data = []
    
    for aspect in aspects:
        count = row.get(f"complaint_count_{aspect}", 0)
        proportion = row.get(f"complaint_proportion_{aspect}", 0)
        growth = row.get(f"growth_{aspect}", 0)
        severity = row.get(f"avg_severity_{aspect}", 0)
        
        if count > 0:
            aspect_data.append({
                "aspect": aspect,
                "count": count,
                "proportion": round(proportion, 4),
                "growth": round(growth, 4),
                "severity": round(severity, 4),
            })
    
    # Sort primarily by complaint count and proportion, then severity and positive growth
    aspect_data.sort(
        key=lambda x: (x["count"], x["proportion"], x["severity"], max(0, x["growth"])),
        reverse=True,
    )
    
    return aspect_data[:top_n]
