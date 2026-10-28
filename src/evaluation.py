"""Evaluation module for Early Product Failure Prediction.

Provides:
- Frequency-based baseline comparison
- Ablation analysis utilities
- Metrics calculation (when valid labels are available)
- Synthetic test scenario evaluation

IMPORTANT: This module clearly separates:
1. Software testing (verifying code correctness)
2. Anomaly detection evaluation (descriptive analysis of detected anomalies)
3. Validated prediction evaluation (requires real failure labels)

Without independently verified product-failure labels, standard failure-prediction
precision, recall, and F1 cannot be validly calculated.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASELINE_COMPLAINT_THRESHOLD, DEFAULT_RISK_WEIGHTS


def frequency_baseline_predict(
    features_df: pd.DataFrame,
    threshold: float = BASELINE_COMPLAINT_THRESHOLD,
    min_history: int = 3,
) -> pd.Series:
    """Simple frequency-based baseline prediction.
    
    Flags a period as 'at risk' if the overall complaint proportion
    exceeds the threshold.
    
    This is the baseline against which the multi-signal approach is compared.
    
    Args:
        features_df: Features DataFrame.
        threshold: Complaint proportion threshold.
        min_history: Minimum periods before prediction.
        
    Returns:
        Boolean Series indicating baseline risk flags.
    """
    if features_df.empty:
        return pd.Series(dtype=bool)
    
    flags = pd.Series(False, index=features_df.index)
    
    for i in range(len(features_df)):
        if i < min_history:
            continue
        proportion = features_df.iloc[i].get("overall_complaint_proportion", 0)
        if proportion >= threshold:
            flags.iloc[i] = True
    
    return flags


def multisignal_predict(
    risk_df: pd.DataFrame,
    risk_threshold: float = 50.0,
    min_history: int = 3,
) -> pd.Series:
    """Multi-signal risk prediction.
    
    Flags a period as 'at risk' if the multi-signal risk score
    exceeds the threshold.
    
    Args:
        risk_df: DataFrame with 'risk_score' column.
        risk_threshold: Risk score threshold for flagging.
        min_history: Minimum periods before prediction.
        
    Returns:
        Boolean Series indicating multi-signal risk flags.
    """
    if risk_df.empty or "risk_score" not in risk_df.columns:
        return pd.Series(dtype=bool)
    
    flags = pd.Series(False, index=risk_df.index)
    
    for i in range(len(risk_df)):
        if i < min_history:
            continue
        score = risk_df.iloc[i]["risk_score"]
        if score >= risk_threshold:
            flags.iloc[i] = True
    
    return flags


def compare_approaches(
    features_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    true_labels: Optional[pd.Series] = None,
    baseline_threshold: float = BASELINE_COMPLAINT_THRESHOLD,
    risk_threshold: float = 50.0,
) -> Dict:
    """Compare frequency baseline with multi-signal approach.
    
    If true_labels are provided (verified failure labels), compute
    precision, recall, and F1 for both approaches.
    
    If true_labels are NOT available, provide descriptive comparison
    showing agreement/disagreement between approaches.
    
    Args:
        features_df: Features DataFrame.
        risk_df: DataFrame with risk scores.
        true_labels: Optional ground truth labels (boolean Series).
        baseline_threshold: Threshold for frequency baseline.
        risk_threshold: Threshold for multi-signal approach.
        
    Returns:
        Comparison results dictionary.
    """
    baseline_flags = frequency_baseline_predict(features_df, baseline_threshold)
    multisignal_flags = multisignal_predict(risk_df, risk_threshold)
    
    result = {
        "has_true_labels": true_labels is not None,
        "baseline_flags_count": int(baseline_flags.sum()),
        "multisignal_flags_count": int(multisignal_flags.sum()),
        "total_periods": len(features_df),
    }
    
    # Agreement analysis
    if len(baseline_flags) == len(multisignal_flags) and len(baseline_flags) > 0:
        both_flag = (baseline_flags & multisignal_flags).sum()
        only_baseline = (baseline_flags & ~multisignal_flags).sum()
        only_multisignal = (~baseline_flags & multisignal_flags).sum()
        neither = (~baseline_flags & ~multisignal_flags).sum()
        
        result["agreement"] = {
            "both_flagged": int(both_flag),
            "only_baseline": int(only_baseline),
            "only_multisignal": int(only_multisignal),
            "neither_flagged": int(neither),
            "agreement_rate": round((int(both_flag) + int(neither)) / len(baseline_flags), 4),
        }
    
    # If true labels are available, compute metrics
    if true_labels is not None and len(true_labels) == len(baseline_flags):
        true_arr = true_labels.astype(bool).values
        
        result["baseline_metrics"] = _compute_metrics(true_arr, baseline_flags.values)
        result["multisignal_metrics"] = _compute_metrics(true_arr, multisignal_flags.values)
    else:
        result["note"] = (
            "Without independently verified product-failure labels, standard "
            "failure-prediction precision, recall, and F1 cannot be validly calculated. "
            "The comparison shows agreement between approaches as a descriptive analysis."
        )
    
    return result


def _compute_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict:
    """Compute classification metrics."""
    if len(true_labels) == 0 or len(pred_labels) == 0:
        return {"error": "Empty labels"}
    
    # Handle all-same-class cases
    if len(np.unique(true_labels)) < 2 or len(np.unique(pred_labels)) < 2:
        return {
            "precision": float(precision_score(true_labels, pred_labels, zero_division=0)),
            "recall": float(recall_score(true_labels, pred_labels, zero_division=0)),
            "f1": float(f1_score(true_labels, pred_labels, zero_division=0)),
            "confusion_matrix": confusion_matrix(true_labels, pred_labels).tolist(),
            "warning": "Limited class diversity in labels."
        }
    
    return {
        "precision": float(precision_score(true_labels, pred_labels, zero_division=0)),
        "recall": float(recall_score(true_labels, pred_labels, zero_division=0)),
        "f1": float(f1_score(true_labels, pred_labels, zero_division=0)),
        "confusion_matrix": confusion_matrix(true_labels, pred_labels).tolist(),
    }


def evaluate_synthetic_scenario(
    risk_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    injection_start_period: int,
) -> Dict:
    """Evaluate detection on a synthetic scenario with known injection point.
    
    This evaluates SOFTWARE BEHAVIOR - whether the system correctly
    identifies the injected anomaly pattern. Results should NOT be
    reported as real-world predictive performance.
    
    Args:
        risk_df: DataFrame with risk scores.
        anomaly_df: DataFrame with anomaly flags.
        injection_start_period: Period index where anomaly was injected.
        
    Returns:
        Evaluation results.
    """
    if risk_df.empty or anomaly_df.empty:
        return {"error": "Empty DataFrames"}
    
    n = len(risk_df)
    
    # True labels: periods from injection_start onwards are "anomalous"
    true_labels = pd.Series(False, index=range(n))
    true_labels[injection_start_period:] = True
    
    # Multi-signal predictions
    pred_risk = risk_df["risk_score"].values >= 50.0
    pred_anomaly = anomaly_df["is_anomaly"].values if "is_anomaly" in anomaly_df.columns else np.zeros(n, dtype=bool)
    
    result = {
        "scenario_type": "synthetic_test",
        "disclaimer": (
            "These results evaluate software behavior on synthetic data. "
            "They should NOT be reported as real-world predictive performance."
        ),
        "injection_start": injection_start_period,
        "total_periods": n,
    }
    
    if len(true_labels) > 0:
        result["risk_score_metrics"] = _compute_metrics(
            true_labels.values, pred_risk
        )
        result["anomaly_detection_metrics"] = _compute_metrics(
            true_labels.values, pred_anomaly
        )
    
    return result


def ablation_summary(
    ablation_results: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize ablation analysis results.
    
    Args:
        ablation_results: Dict from risk_scoring.ablation_analysis().
        
    Returns:
        Summary DataFrame with mean risk scores for each configuration.
    """
    summary_rows = []
    
    for config_name, df in ablation_results.items():
        if "risk_score" in df.columns and len(df) > 0:
            summary_rows.append({
                "configuration": config_name,
                "mean_risk_score": round(df["risk_score"].mean(), 2),
                "max_risk_score": round(df["risk_score"].max(), 2),
                "min_risk_score": round(df["risk_score"].min(), 2),
                "std_risk_score": round(df["risk_score"].std(), 2),
                "periods_high_risk": int((df["risk_score"] >= 50).sum()),
                "periods_critical": int((df["risk_score"] >= 75).sum()),
            })
    
    return pd.DataFrame(summary_rows)
