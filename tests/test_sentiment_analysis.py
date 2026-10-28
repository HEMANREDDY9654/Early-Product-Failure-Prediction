"""Tests for sentiment_analysis module."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sentiment_analysis import (
    analyze_sentiment_fallback,
    analyze_sentiment,
    combine_severity_with_rating,
    analyze_batch,
    reset_model_cache,
    is_transformer_available,
)


class TestSentimentFallback:
    """Test rule-based fallback sentiment analysis."""

    def test_positive_text(self):
        result = analyze_sentiment_fallback("This phone is amazing, excellent battery and wonderful camera!")
        assert result["label"] == "POSITIVE"
        assert result["method"] == "fallback"
        assert result["severity"] < 0.5

    def test_negative_text(self):
        result = analyze_sentiment_fallback("Terrible device, horrible overheating, worst purchase ever!")
        assert result["label"] == "NEGATIVE"
        assert result["method"] == "fallback"
        assert result["severity"] > 0.5

    def test_neutral_text(self):
        result = analyze_sentiment_fallback("The package arrived yesterday.")
        assert result["label"] == "NEUTRAL"
        assert result["severity"] == 0.5

    def test_negation_handling(self):
        # Negation turns positive into negative
        result = analyze_sentiment_fallback("This is not good, not reliable at all.")
        assert result["severity"] > 0.5

    def test_empty_and_none_text(self):
        res_empty = analyze_sentiment_fallback("")
        assert res_empty["label"] == "NEUTRAL"
        assert res_empty["severity"] == 0.5

        res_none = analyze_sentiment_fallback(None)
        assert res_none["label"] == "NEUTRAL"
        assert res_none["severity"] == 0.5

    def test_severity_bounds(self):
        texts = [
            "Superb outstanding masterpiece",
            "Disastrous catastrophic broken defect",
            "Completely average device",
            "Not bad at all",
            "Very very terrible and frustrating",
        ]
        for t in texts:
            res = analyze_sentiment_fallback(t)
            assert 0.0 <= res["severity"] <= 1.0
            assert 0.0 <= res["score"] <= 1.0


class TestCombineSeverityWithRating:
    """Test combining text severity with star rating."""

    def test_one_star_rating_increases_severity(self):
        # 1 star -> rating_severity = 1.0
        text_sev = 0.5
        combined = combine_severity_with_rating(text_sev, rating=1.0)
        # 0.7 * 0.5 + 0.3 * 1.0 = 0.35 + 0.30 = 0.65
        assert combined > text_sev
        assert round(combined, 2) == 0.65

    def test_five_star_rating_decreases_severity(self):
        # 5 star -> rating_severity = 0.0
        text_sev = 0.5
        combined = combine_severity_with_rating(text_sev, rating=5.0)
        # 0.7 * 0.5 + 0.3 * 0.0 = 0.35
        assert combined < text_sev
        assert round(combined, 2) == 0.35

    def test_nan_rating_preserves_text_severity(self):
        text_sev = 0.75
        combined = combine_severity_with_rating(text_sev, rating=np.nan)
        assert combined == text_sev

    def test_none_rating_preserves_text_severity(self):
        text_sev = 0.82
        combined = combine_severity_with_rating(text_sev, rating=None)
        assert combined == text_sev

    def test_bounds_preservation(self):
        assert combine_severity_with_rating(1.0, 1.0) <= 1.0
        assert combine_severity_with_rating(0.0, 5.0) >= 0.0


class TestAnalyzeBatchAndFallback:
    """Test batch analysis and fallback mode."""

    def test_batch_processing(self):
        texts = [
            "Great battery life and awesome screen",
            "Device overheats and battery drains very fast",
            "Average quality nothing special",
        ]
        ratings = [5.0, 1.0, 3.0]
        results = analyze_batch(texts, ratings, use_transformer=False)
        assert len(results) == 3
        assert "combined_severity" in results[0]
        assert results[0]["combined_severity"] < results[1]["combined_severity"]

    def test_analyze_sentiment_fallback_flag(self):
        res = analyze_sentiment("Great device!", use_transformer=False)
        assert res["method"] == "fallback"
        assert res["label"] == "POSITIVE"
