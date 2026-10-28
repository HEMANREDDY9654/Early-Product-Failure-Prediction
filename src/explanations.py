"""Explanation generation module.

Generates human-readable explanations for risk scores and anomaly warnings.
All explanations are derived from actual calculated values - nothing is hardcoded.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ASPECT_LEXICON, RISK_BANDS


def generate_risk_explanation(
    risk_result: Dict,
    period_features: Dict,
    top_aspects: Optional[List[Dict]] = None,
) -> str:
    """Generate a human-readable explanation of the risk score.
    
    Args:
        risk_result: Output from risk_scoring.calculate_risk_score().
        period_features: Dict of features for the current period.
        top_aspects: Optional list of top contributing aspects.
        
    Returns:
        Multi-sentence explanation string.
    """
    score = risk_result.get("risk_score", 0)
    level = risk_result.get("risk_level", "Unknown")
    contributions = risk_result.get("signal_contributions", {})
    normalized = risk_result.get("normalized_signals", {})
    
    parts = []
    
    # Overall assessment
    parts.append(
        f"Product risk score is {score:.1f}/100, classified as '{level}'."
    )
    
    # Top contributing signals
    sorted_contribs = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    top_signals = [s for s, c in sorted_contribs if c > 0][:3]
    
    if top_signals:
        signal_descriptions = []
        for signal in top_signals:
            contrib_val = contributions.get(signal, 0)
            norm_val = normalized.get(signal, 0)
            desc = _describe_signal(signal, norm_val, contrib_val, period_features)
            if desc:
                signal_descriptions.append(desc)
        
        if signal_descriptions:
            parts.append("Key contributing factors: " + "; ".join(signal_descriptions) + ".")
    
    # Top aspects
    if top_aspects:
        aspect_names = [a["aspect"] for a in top_aspects[:3]]
        if aspect_names:
            parts.append(
                f"Most prominent complaint aspects: {', '.join(aspect_names)}."
            )
    
    # Caveats
    if score < 25:
        parts.append("No significant concern detected in this period.")
    elif score >= 75:
        parts.append(
            "This is a heuristic risk estimate based on review patterns. "
            "It indicates elevated complaint signals but does not confirm product failure."
        )
    
    return " ".join(parts)


def _describe_signal(
    signal: str,
    normalized_value: float,
    contribution: float,
    features: Dict,
) -> str:
    """Generate description for a single signal."""
    if contribution <= 0:
        return ""
    
    signal_names = {
        "frequency": "Complaint frequency",
        "severity": "Complaint severity",
        "growth": "Complaint growth",
        "symptom_diversity": "Symptom diversity",
        "review_velocity": "Review velocity",
    }
    
    name = signal_names.get(signal, signal)
    
    if signal == "frequency":
        prop = features.get("overall_complaint_proportion", 0)
        complaints_n = features.get("complaint_reviews_count", features.get("total_complaints", 0))
        total_n = features.get("total_reviews", 0)
        return f"{name} is at {prop:.1%} ({complaints_n}/{total_n} valid reviews)"
    elif signal == "severity":
        sev = features.get("overall_avg_severity", 0)
        return f"{name} is elevated at {sev:.2f} (scale 0-1)"
    elif signal == "growth":
        growth = features.get("complaint_growth", 0)
        if growth > 0:
            return f"{name} shows {growth:.1%} period-over-period increase"
        return ""
    elif signal == "symptom_diversity":
        n_aspects = features.get("num_distinct_aspects", 0)
        return f"{n_aspects}/9 distinct complaint aspects detected"
    elif signal == "review_velocity":
        vel = features.get("review_velocity", 0)
        return f"{vel} valid reviews received this period"
    
    return f"{name} contributing {contribution:.1f} points"


def generate_anomaly_explanation(
    anomaly_row: pd.Series,
    top_aspects: Optional[List[Dict]] = None,
) -> str:
    """Generate explanation for an anomaly detection alert.
    
    All values in the explanation are derived from actual calculated data.
    Explicitly explains the denominator for proportions and states the deviation method.
    
    Args:
        anomaly_row: Row from anomaly detection results.
        top_aspects: Optional list of top contributing aspects.
        
    Returns:
        Explanation string.
    """
    parts = []
    signals = anomaly_row.get("anomaly_signals", [])
    confidence = anomaly_row.get("confidence", "heuristic")
    total_reviews = anomaly_row.get("total_reviews", None)
    
    if not signals:
        return "No anomaly signals detected in this period."
    
    parts.append(f"[ALERT] Anomaly detected (confidence: {confidence}).")
    
    for sig_info in signals:
        signal = sig_info.get("signal", "")
        current = sig_info.get("current", 0)
        baseline = sig_info.get("baseline", 0)
        z_score = sig_info.get("z_score", 0)
        method = sig_info.get("method", "rolling baseline deviation")
        
        if signal == "complaint_proportion":
            parts.append(
                f"Complaint-review proportion ({current:.1%}) has increased compared "
                f"with the product's historical baseline ({baseline:.1%}), "
                f"deviating by {z_score:.1f} standard deviations (method: {method})."
            )
        elif signal == "severity":
            parts.append(
                f"Average complaint severity ({current:.2f}) is elevated "
                f"compared to baseline ({baseline:.2f}), "
                f"deviating by {z_score:.1f} standard deviations."
            )
        elif signal == "symptom_diversity":
            parts.append(
                f"Symptom diversity ({current:.2f}) has increased beyond "
                f"the historical norm ({baseline:.2f}), "
                f"suggesting complaints are spreading across more aspects."
            )
        elif signal == "review_velocity":
            parts.append(
                f"Review volume ({current:.0f}) has spiked compared "
                f"to the historical baseline average ({baseline:.1f})."
            )
    
    # Add aspect information with clear counts and severity
    if top_aspects:
        aspect_details = []
        for a in top_aspects[:3]:
            count = a.get("count", 0)
            prop = a.get("proportion", 0)
            sev = a.get("severity", 0)
            detail = f"{a['aspect']} (count: {count}, proportion: {prop:.1%}, severity: {sev:.2f})"
            aspect_details.append(detail)
        if aspect_details:
            parts.append(
                "Top complaint aspects in this period: " + ", ".join(aspect_details) + "."
            )
    
    return " ".join(parts)


def generate_comparison_explanation(comparison: Dict) -> str:
    """Generate explanation for baseline vs multi-signal comparison."""
    parts = []
    
    baseline_count = comparison.get("baseline_flags_count", 0)
    multi_count = comparison.get("multisignal_flags_count", 0)
    total = comparison.get("total_periods", 0)
    
    parts.append(
        f"Out of {total} time periods analyzed: "
        f"the frequency baseline flagged {baseline_count} periods, "
        f"while the multi-signal approach flagged {multi_count} periods."
    )
    
    agreement = comparison.get("agreement", {})
    if agreement:
        rate = agreement.get("agreement_rate", 0)
        both = agreement.get("both_flagged", 0)
        only_base = agreement.get("only_baseline", 0)
        only_multi = agreement.get("only_multisignal", 0)
        
        parts.append(f"Agreement rate: {rate:.1%}.")
        if both > 0:
            parts.append(f"{both} periods were flagged by both approaches.")
        if only_base > 0:
            parts.append(
                f"{only_base} periods were flagged only by the frequency baseline "
                f"(high complaint count but possibly low severity or diversity)."
            )
        if only_multi > 0:
            parts.append(
                f"{only_multi} periods were flagged only by the multi-signal approach "
                f"(complaint proportion below threshold but other risk signals elevated)."
            )
    
    if comparison.get("has_true_labels"):
        parts.append("Validated metrics are available — see detailed results.")
    else:
        parts.append(
            comparison.get("note", 
            "No verified failure labels available for validated evaluation.")
        )
    
    return " ".join(parts)
