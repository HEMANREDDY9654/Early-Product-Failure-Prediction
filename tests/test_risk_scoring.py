"""Tests for risk_scoring module."""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.risk_scoring import (
    normalize_signal,
    calculate_risk_score,
    classify_risk,
    calculate_risk_timeseries,
    ablation_analysis,
)
from config import DEFAULT_RISK_WEIGHTS


class TestNormalizeSignal:
    """Test signal normalization."""

    def test_within_range(self):
        assert normalize_signal(0.5, 0.0, 1.0) == 0.5

    def test_at_min(self):
        assert normalize_signal(0.0, 0.0, 1.0) == 0.0

    def test_at_max(self):
        assert normalize_signal(1.0, 0.0, 1.0) == 1.0

    def test_above_max(self):
        assert normalize_signal(2.0, 0.0, 1.0) == 1.0

    def test_below_min(self):
        assert normalize_signal(-1.0, 0.0, 1.0) == 0.0

    def test_equal_min_max(self):
        assert normalize_signal(5.0, 5.0, 5.0) == 0.0


class TestCalculateRiskScore:
    """Test risk score calculation."""

    def test_zero_signals(self):
        result = calculate_risk_score(0, 0, 0, 0, 0)
        assert result["risk_score"] == 0.0
        assert result["risk_level"] == "Low"

    def test_max_signals(self):
        result = calculate_risk_score(1.0, 1.0, 5.0, 1.0, 5.0)
        assert result["risk_score"] > 0
        assert result["risk_score"] <= 100

    def test_score_bounds(self):
        """Risk score should always be between 0 and 100."""
        for _ in range(100):
            freq = np.random.uniform(0, 1)
            sev = np.random.uniform(0, 1)
            growth = np.random.uniform(-5, 5)
            div = np.random.uniform(0, 1)
            vel = np.random.uniform(0, 5)
            result = calculate_risk_score(freq, sev, growth, div, vel)
            assert 0 <= result["risk_score"] <= 100

    def test_weight_normalization(self):
        """Weights should be normalized if they don't sum to 1."""
        weights = {"frequency": 0.5, "severity": 0.5, "growth": 0.5,
                   "symptom_diversity": 0.5, "review_velocity": 0.5}
        result = calculate_risk_score(0.5, 0.5, 0.5, 0.5, 1.5, weights=weights)
        assert 0 <= result["risk_score"] <= 100

    def test_signal_contributions_sum(self):
        """Signal contributions should sum to approximately the risk score."""
        result = calculate_risk_score(0.5, 0.5, 1.0, 0.5, 1.5)
        total_contrib = sum(result["signal_contributions"].values())
        assert abs(total_contrib - result["risk_score"]) < 0.1

    def test_no_history(self):
        """Without history, growth and velocity contributions should be zero."""
        result = calculate_risk_score(0.5, 0.5, 2.0, 0.5, 3.0, history_available=False)
        assert result["signal_contributions"]["growth"] == 0.0
        assert result["signal_contributions"]["review_velocity"] == 0.0

    def test_custom_weights(self):
        """Custom weights should be used."""
        weights = {"frequency": 1.0, "severity": 0.0, "growth": 0.0,
                   "symptom_diversity": 0.0, "review_velocity": 0.0}
        result = calculate_risk_score(0.5, 1.0, 5.0, 1.0, 5.0, weights=weights)
        # Only frequency should contribute
        assert result["signal_contributions"]["severity"] == 0.0
        assert result["signal_contributions"]["growth"] == 0.0


class TestClassifyRisk:
    """Test risk level classification."""

    def test_low(self):
        assert classify_risk(10) == "Low"

    def test_moderate(self):
        assert classify_risk(30) == "Moderate"

    def test_high(self):
        assert classify_risk(60) == "High"

    def test_critical(self):
        assert classify_risk(80) == "Critical"

    def test_boundary_low_moderate(self):
        assert classify_risk(25) == "Moderate"

    def test_exact_100(self):
        assert classify_risk(100) == "Critical"

    def test_zero(self):
        assert classify_risk(0) == "Low"


class TestRiskTimeseries:
    """Test risk score timeseries calculation."""

    def test_timeseries(self):
        features = pd.DataFrame({
            "period": pd.date_range("2024-01-01", periods=10, freq="W"),
            "overall_complaint_proportion": np.linspace(0.1, 0.6, 10),
            "overall_avg_severity": np.linspace(0.3, 0.8, 10),
            "complaint_growth": np.linspace(0, 0.5, 10),
            "symptom_diversity": np.linspace(0.1, 0.5, 10),
            "review_velocity": np.random.randint(5, 20, 10),
            "baseline_velocity": [10] * 10,
        })
        result = calculate_risk_timeseries(features)
        assert "risk_score" in result.columns
        assert "risk_level" in result.columns
        assert len(result) == 10

    def test_empty_dataframe(self):
        result = calculate_risk_timeseries(pd.DataFrame())
        assert len(result) == 0


class TestAblationAnalysis:
    """Test ablation analysis."""

    def test_ablation(self):
        features = pd.DataFrame({
            "period": pd.date_range("2024-01-01", periods=8, freq="W"),
            "overall_complaint_proportion": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
            "overall_avg_severity": [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
            "complaint_growth": [0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
            "symptom_diversity": [0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5],
            "review_velocity": [10, 12, 14, 16, 18, 20, 22, 24],
            "baseline_velocity": [10] * 8,
        })
        results = ablation_analysis(features)
        assert "full_model" in results
        assert "without_frequency" in results
        assert "without_severity" in results
        assert "without_growth" in results
        assert "without_symptom_diversity" in results
        assert "without_review_velocity" in results
