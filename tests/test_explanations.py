"""Tests for explanations module."""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.explanations import (
    generate_risk_explanation,
    generate_anomaly_explanation,
    generate_comparison_explanation,
)


class TestRiskExplanations:
    """Test risk explanation text generation."""

    def test_low_risk_explanation(self):
        risk_result = {
            "risk_score": 12.5,
            "risk_level": "Low",
            "signal_contributions": {
                "frequency": 3.0,
                "severity": 4.0,
                "growth": 0.0,
                "symptom_diversity": 2.5,
                "review_velocity": 3.0,
            },
            "normalized_signals": {"frequency": 0.12, "severity": 0.2},
        }
        period_features = {
            "overall_complaint_proportion": 0.12,
            "overall_avg_severity": 0.25,
            "complaint_growth": 0.0,
            "num_distinct_aspects": 1,
            "review_velocity": 10,
        }
        text = generate_risk_explanation(risk_result, period_features)
        assert "12.5/100" in text
        assert "Low" in text
        assert "No significant concern detected" in text

    def test_critical_risk_explanation_with_aspects(self):
        risk_result = {
            "risk_score": 82.0,
            "risk_level": "Critical",
            "signal_contributions": {
                "frequency": 22.0,
                "severity": 18.0,
                "growth": 20.0,
                "symptom_diversity": 12.0,
                "review_velocity": 10.0,
            },
            "normalized_signals": {"frequency": 0.88, "severity": 0.9, "growth": 0.8},
        }
        period_features = {
            "overall_complaint_proportion": 0.88,
            "overall_avg_severity": 0.9,
            "complaint_growth": 0.65,
            "num_distinct_aspects": 4,
            "review_velocity": 45,
        }
        top_aspects = [
            {"aspect": "Battery", "count": 12, "proportion": 0.45, "growth": 0.5, "severity": 0.85},
            {"aspect": "Heating", "count": 8, "proportion": 0.30, "growth": 0.4, "severity": 0.90},
        ]
        text = generate_risk_explanation(risk_result, period_features, top_aspects=top_aspects)
        assert "82.0/100" in text
        assert "Critical" in text
        assert "Battery" in text
        assert "Heating" in text
        assert "heuristic risk estimate" in text


class TestAnomalyExplanations:
    """Test anomaly warning text generation."""

    def test_anomaly_explanation_with_signals(self):
        anomaly_row = pd.Series({
            "confidence": "high",
            "anomaly_signals": [
                {
                    "signal": "complaint_proportion",
                    "current": 0.65,
                    "baseline": 0.20,
                    "z_score": 3.2,
                },
                {
                    "signal": "severity",
                    "current": 0.82,
                    "baseline": 0.35,
                    "z_score": 2.8,
                },
            ],
        })
        top_aspects = [
            {"aspect": "Battery", "count": 10, "severity": 0.85},
        ]
        text = generate_anomaly_explanation(anomaly_row, top_aspects)
        assert "high" in text
        assert "65.0%" in text
        assert "3.2 standard deviations" in text
        assert "Battery" in text

    def test_no_signals(self):
        anomaly_row = pd.Series({"anomaly_signals": []})
        text = generate_anomaly_explanation(anomaly_row)
        assert "No anomaly signals detected" in text


class TestComparisonExplanations:
    """Test baseline vs multi-signal comparison explanation."""

    def test_comparison_explanation_text(self):
        comp = {
            "total_periods": 20,
            "baseline_flags_count": 5,
            "multisignal_flags_count": 7,
            "agreement": {
                "agreement_rate": 0.85,
                "both_flagged": 4,
                "only_baseline": 1,
                "only_multisignal": 3,
                "neither_flagged": 12,
            },
            "has_true_labels": False,
            "note": "No verified failure labels available.",
        }
        text = generate_comparison_explanation(comp)
        assert "20 time periods" in text
        assert "85.0%" in text
        assert "both approaches" in text
