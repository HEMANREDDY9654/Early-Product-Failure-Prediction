"""Product Risk Scoring module.

Calculates a transparent, configurable risk score from 0 to 100 based on:
1. Complaint Frequency (proportion of reviews with complaints)
2. Complaint Severity (average severity of complaint reviews)
3. Complaint Growth (rate of change in complaint proportion)
4. Symptom Diversity (number of distinct complaint aspects)
5. Review Arrival Velocity (review volume relative to baseline)

Weights are configurable. Default weights are engineering estimates,
not empirically optimized values.

The risk score is a heuristic indicator, NOT a calibrated probability of failure.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_RISK_WEIGHTS, RISK_BANDS


def normalize_signal(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize a signal to [0, 1] range using min-max normalization.
    
    Args:
        value: Raw signal value.
        min_val: Minimum expected value.
        max_val: Maximum expected value.
        
    Returns:
        Normalized value clipped to [0, 1].
    """
    if max_val <= min_val:
        return 0.0
    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


def calculate_risk_score(
    complaint_proportion: float,
    avg_severity: float,
    growth_rate: float,
    symptom_diversity: float,
    review_velocity_ratio: float,
    weights: Optional[Dict[str, float]] = None,
    history_available: bool = True,
) -> Dict:
    """Calculate the product risk score.
    
    Each signal is normalized to [0, 1] then combined using weights.
    The result is scaled to [0, 100].
    
    Normalization approach:
    - Frequency: complaint_proportion is already in [0, 1]
    - Severity: avg_severity is already in [0, 1]
    - Growth: growth_rate is unbounded; we use tanh to map to [0, 1]
      where 0 = no growth, 1 = very high growth
    - Symptom diversity: already in [0, 1]
    - Velocity ratio: ratio of current to baseline; use tanh for normalization
    
    Args:
        complaint_proportion: Proportion of reviews with complaints (0-1).
        avg_severity: Average complaint severity (0-1).
        growth_rate: Complaint growth rate (can be negative).
        symptom_diversity: Normalized distinct aspect count (0-1).
        review_velocity_ratio: Current velocity / baseline velocity.
        weights: Custom weights dict. Uses defaults if None.
        history_available: Whether sufficient history exists.
        
    Returns:
        Dict with 'risk_score', 'risk_level', 'signal_contributions', 
        'normalized_signals', and 'weights_used'.
    """
    if weights is None:
        weights = DEFAULT_RISK_WEIGHTS.copy()
    
    # Ensure weights sum to 1.0
    weight_sum = sum(weights.values())
    if weight_sum > 0 and abs(weight_sum - 1.0) > 1e-6:
        weights = {k: v / weight_sum for k, v in weights.items()}
    
    # Normalize signals
    norm_frequency = normalize_signal(complaint_proportion, 0.0, 1.0)
    norm_severity = normalize_signal(avg_severity, 0.0, 1.0)
    
    # Growth: use tanh to handle unbounded values; only positive growth contributes to risk
    norm_growth = max(0.0, float(np.tanh(growth_rate))) if history_available else 0.0
    
    norm_diversity = normalize_signal(symptom_diversity, 0.0, 1.0)
    
    # Velocity: ratio > 1 means more reviews than baseline
    # Higher velocity can indicate emerging issues (but alone is not sufficient)
    velocity_excess = max(0.0, review_velocity_ratio - 1.0) if history_available else 0.0
    norm_velocity = float(np.tanh(velocity_excess))
    
    normalized_signals = {
        "frequency": norm_frequency,
        "severity": norm_severity,
        "growth": norm_growth,
        "symptom_diversity": norm_diversity,
        "review_velocity": norm_velocity,
    }
    
    # Calculate weighted score
    signal_contributions = {}
    total_score = 0.0
    for signal_name, norm_value in normalized_signals.items():
        w = weights.get(signal_name, 0.0)
        contribution = norm_value * w * 100
        signal_contributions[signal_name] = contribution
        total_score += contribution
    
    # Clip to [0, 100]
    risk_score = max(0.0, min(100.0, total_score))
    
    # Determine risk level
    risk_level = classify_risk(risk_score)
    
    return {
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "signal_contributions": signal_contributions,
        "normalized_signals": normalized_signals,
        "weights_used": weights,
    }


def classify_risk(score: float, bands: Optional[Dict] = None) -> str:
    """Classify risk score into a risk band.
    
    Args:
        score: Risk score 0-100.
        bands: Custom risk bands. Uses defaults if None.
        
    Returns:
        Risk level string.
    """
    if bands is None:
        bands = RISK_BANDS
    
    for level, (low, high) in bands.items():
        if low <= score < high:
            return level
    
    # Edge case: score exactly 100
    return "Critical"


def calculate_risk_timeseries(
    features_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    min_history_periods: int = 3,
) -> pd.DataFrame:
    """Calculate risk scores for each period in the features DataFrame.
    
    Args:
        features_df: Output from temporal_features.calculate_period_features().
        weights: Custom risk weights.
        min_history_periods: Minimum periods before full risk assessment.
        
    Returns:
        DataFrame with added risk score columns.
    """
    if features_df.empty:
        return features_df
    
    df = features_df.copy()
    risk_scores = []
    risk_levels = []
    contributions_list = []
    
    for i, row in df.iterrows():
        idx = df.index.get_loc(i)
        history_available = idx >= min_history_periods
        
        # Calculate velocity ratio
        baseline_vel = row.get("baseline_velocity", 0)
        current_vel = row.get("review_velocity", 0)
        velocity_ratio = current_vel / baseline_vel if baseline_vel > 0 else 1.0
        
        result = calculate_risk_score(
            complaint_proportion=row.get("overall_complaint_proportion", 0),
            avg_severity=row.get("overall_avg_severity", 0),
            growth_rate=row.get("complaint_growth", 0),
            symptom_diversity=row.get("symptom_diversity", 0),
            review_velocity_ratio=velocity_ratio,
            weights=weights,
            history_available=history_available,
        )
        
        risk_scores.append(result["risk_score"])
        risk_levels.append(result["risk_level"])
        contributions_list.append(result["signal_contributions"])
    
    df["risk_score"] = risk_scores
    df["risk_level"] = risk_levels
    
    # Add individual contributions as columns
    contrib_df = pd.DataFrame(contributions_list)
    for col in contrib_df.columns:
        df[f"contrib_{col}"] = contrib_df[col].values
    
    return df


def ablation_analysis(
    features_df: pd.DataFrame,
    base_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, pd.DataFrame]:
    """Perform ablation analysis by removing one signal at a time.
    
    For each signal, recalculate risk scores with that signal's weight
    set to zero (and remaining weights renormalized).
    
    Args:
        features_df: Features DataFrame.
        base_weights: Base weights to use.
        
    Returns:
        Dict mapping signal name to results DataFrame.
    """
    if base_weights is None:
        base_weights = DEFAULT_RISK_WEIGHTS.copy()
    
    results = {}
    
    # Full model
    results["full_model"] = calculate_risk_timeseries(features_df, base_weights)
    
    # Remove each signal
    for signal_to_remove in base_weights.keys():
        ablated_weights = base_weights.copy()
        ablated_weights[signal_to_remove] = 0.0
        
        # Renormalize remaining weights
        remaining_sum = sum(ablated_weights.values())
        if remaining_sum > 0:
            ablated_weights = {k: v / remaining_sum for k, v in ablated_weights.items()}
        
        results[f"without_{signal_to_remove}"] = calculate_risk_timeseries(
            features_df, ablated_weights
        )
    
    return results
