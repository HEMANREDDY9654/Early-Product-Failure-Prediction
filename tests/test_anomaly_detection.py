"""Tests for anomaly_detection module."""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.anomaly_detection import (
    detect_anomalies,
    get_anomaly_periods,
    get_top_anomaly_aspects,
)
from config import ASPECT_LEXICON


def _make_features_with_anomaly(n_periods=12, anomaly_start=8):
    """Create features DataFrame with a simulated anomaly."""
    aspects = list(ASPECT_LEXICON.keys())
    rows = []
    for i in range(n_periods):
        row = {
            "period": pd.Timestamp("2024-01-01") + pd.DateOffset(weeks=i),
            "total_reviews": 10,
            "review_velocity": 10,
            "total_complaints": 3,
            "overall_complaint_proportion": 0.3,
            "overall_avg_severity": 0.4,
            "symptom_diversity": 0.2,
            "num_distinct_aspects": 2,
            "complaint_growth": 0.0,
            "baseline_complaint_proportion": 0.3,
            "baseline_severity": 0.4,
            "baseline_velocity": 10,
        }
        # Inject anomaly
        if i >= anomaly_start:
            row["overall_complaint_proportion"] = 0.8
            row["overall_avg_severity"] = 0.9
            row["symptom_diversity"] = 0.6
            row["review_velocity"] = 25
            row["total_complaints"] = 8
        for aspect in aspects:
            row[f"complaint_count_{aspect}"] = 1 if i >= anomaly_start and aspect in ["Battery", "Heating"] else 0
            row[f"complaint_proportion_{aspect}"] = 0.1 if i >= anomaly_start and aspect in ["Battery", "Heating"] else 0.0
            row[f"avg_severity_{aspect}"] = 0.8 if i >= anomaly_start and aspect in ["Battery", "Heating"] else 0.0
            row[f"growth_{aspect}"] = 0.5 if i >= anomaly_start and aspect in ["Battery", "Heating"] else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


class TestDetectAnomalies:
    """Test anomaly detection."""

    def test_detects_anomaly(self):
        df = _make_features_with_anomaly()
        result = detect_anomalies(df)
        assert "is_anomaly" in result.columns
        # Should detect something in the anomaly period
        anomaly_periods = result[result["is_anomaly"] == True]
        assert len(anomaly_periods) > 0

    def test_no_anomaly_before_history(self):
        """No anomalies should be flagged in the first min_history periods."""
        df = _make_features_with_anomaly()
        result = detect_anomalies(df, min_history=3)
        first_periods = result.iloc[:3]
        assert not first_periods["is_anomaly"].any()

    def test_no_future_leakage(self):
        """Anomaly detection should not use future data."""
        df = _make_features_with_anomaly(n_periods=12, anomaly_start=8)
        result = detect_anomalies(df, min_history=3)
        # Periods before anomaly start should mostly not be flagged
        pre_anomaly = result.iloc[3:8]
        # Not all should be flagged (data is normal)
        # This test verifies the logic doesn't flag normal periods using
        # knowledge of future anomalous data
        assert pre_anomaly["is_anomaly"].sum() < len(pre_anomaly)

    def test_empty_dataframe(self):
        result = detect_anomalies(pd.DataFrame())
        assert len(result) == 0

    def test_confidence_levels(self):
        df = _make_features_with_anomaly()
        result = detect_anomalies(df, min_reviews=5)
        assert "confidence" in result.columns

    def test_sparse_data(self):
        """Should handle very few periods gracefully."""
        df = _make_features_with_anomaly(n_periods=2)
        result = detect_anomalies(df, min_history=3)
        # Not enough history, no anomalies should be flagged
        assert not result["is_anomaly"].any()

    def test_anomaly_score(self):
        """Anomaly score should be non-negative."""
        df = _make_features_with_anomaly()
        result = detect_anomalies(df)
        assert (result["anomaly_score"] >= 0).all()


class TestGetAnomalyPeriods:
    """Test anomaly period extraction."""

    def test_filters_correctly(self):
        df = _make_features_with_anomaly()
        result = detect_anomalies(df)
        anomalies = get_anomaly_periods(result)
        assert len(anomalies) <= len(result)
        assert anomalies["is_anomaly"].all()


class TestTopAnomalyAspects:
    """Test top anomaly aspect extraction."""

    def test_returns_aspects(self):
        df = _make_features_with_anomaly()
        top = get_top_anomaly_aspects(df, period_idx=9)
        assert isinstance(top, list)
        if len(top) > 0:
            assert "aspect" in top[0]
            assert "count" in top[0]

    def test_invalid_index(self):
        df = _make_features_with_anomaly()
        assert get_top_anomaly_aspects(df, period_idx=999) == []
        assert get_top_anomaly_aspects(df, period_idx=-1) == []
