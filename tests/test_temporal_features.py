"""Tests for temporal_features module."""
import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.temporal_features import calculate_period_features


def _make_test_df(n_periods=6, reviews_per_period=5):
    """Create a test DataFrame with period, aspects, and severity columns."""
    rows = []
    aspects_options = [
        ["Battery"], ["Camera"], ["Battery", "Heating"],
        ["Software"], ["Display"], [],
    ]
    start = datetime(2024, 1, 1)
    for p in range(n_periods):
        period_start = pd.Timestamp(start) + pd.DateOffset(weeks=p)
        for r in range(reviews_per_period):
            rows.append({
                "period": period_start,
                "review_text": f"Review {p}-{r}",
                "aspects": aspects_options[r % len(aspects_options)],
                "combined_severity": np.random.uniform(0.2, 0.9),
            })
    return pd.DataFrame(rows)


class TestCalculatePeriodFeatures:
    """Test temporal feature calculations."""

    def test_basic_features(self):
        df = _make_test_df()
        features = calculate_period_features(df)
        assert len(features) > 0
        assert "total_reviews" in features.columns
        assert "review_velocity" in features.columns
        assert "symptom_diversity" in features.columns
        assert "overall_complaint_proportion" in features.columns

    def test_complaint_counts(self):
        df = _make_test_df()
        features = calculate_period_features(df)
        # Every period should have some reviews
        assert (features["total_reviews"] > 0).all()

    def test_growth_rates(self):
        df = _make_test_df()
        features = calculate_period_features(df)
        assert "complaint_growth" in features.columns
        # First period growth should be 0 (no previous)
        assert features.iloc[0]["complaint_growth"] == 0.0

    def test_baselines(self):
        df = _make_test_df()
        features = calculate_period_features(df)
        assert "baseline_complaint_proportion" in features.columns
        assert "baseline_severity" in features.columns
        assert "baseline_velocity" in features.columns

    def test_deviations(self):
        df = _make_test_df()
        features = calculate_period_features(df)
        assert "deviation_complaint_proportion" in features.columns

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        features = calculate_period_features(df)
        assert len(features) == 0

    def test_single_period(self):
        rows = [{
            "period": pd.Timestamp("2024-01-01"),
            "review_text": "Test review",
            "aspects": ["Battery"],
            "combined_severity": 0.8,
        }]
        df = pd.DataFrame(rows)
        features = calculate_period_features(df)
        assert len(features) == 1
        assert features.iloc[0]["total_reviews"] == 1

    def test_zero_denominator_growth(self):
        """Test growth calculation when previous proportion is zero."""
        rows = []
        for i in range(3):
            period = pd.Timestamp("2024-01-01") + pd.DateOffset(weeks=i)
            aspects = [] if i < 2 else ["Battery"]
            rows.append({
                "period": period,
                "review_text": "Test",
                "aspects": aspects,
                "combined_severity": 0.5,
            })
        df = pd.DataFrame(rows)
        features = calculate_period_features(df)
        # Should not produce inf or NaN
        assert np.isfinite(features["complaint_growth"]).all()

    def test_chronological_ordering(self):
        """Features should be sorted chronologically."""
        df = _make_test_df()
        features = calculate_period_features(df)
        periods = features["period"].tolist()
        assert periods == sorted(periods)

    def test_symptom_diversity_range(self):
        """Symptom diversity should be between 0 and 1."""
        df = _make_test_df()
        features = calculate_period_features(df)
        assert (features["symptom_diversity"] >= 0).all()
        assert (features["symptom_diversity"] <= 1).all()
