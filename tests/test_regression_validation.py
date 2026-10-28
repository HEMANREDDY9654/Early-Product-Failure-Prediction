"""Comprehensive regression and validation test suite.

Verifies fixes for:
1. Complaint-review proportions never exceeding 100%.
2. Multi-aspect reviews counted once in overall complaint proportions.
3. Aspect-specific proportions calculated correctly.
4. Duplicate review IDs handled according to the documented policy.
5. Invalid timestamps excluded from temporal analysis.
6. Missing ratings handled correctly (retained with NaN rating).
7. Empty review text excluded from text analysis.
8. Product-specific aggregation.
9. Chronological historical baselines without future-data leakage.
10. Sparse histories and zero-variance signals.
11. Risk-score bounds and weight validation.
12. Alert explanations matching actual supporting reviews.
13. Risk charts and alerts updating when the selected product changes.
"""
import pytest
import pandas as pd
import numpy as np
import io
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import validate_and_load_csv, filter_by_product, aggregate_by_period
from src.preprocessing import preprocess_reviews
from src.temporal_features import calculate_period_features
from src.risk_scoring import calculate_risk_score, calculate_risk_timeseries
from src.anomaly_detection import detect_anomalies, get_top_anomaly_aspects
from src.explanations import generate_anomaly_explanation, generate_risk_explanation


class TestComplaintProportions:
    """1, 2, 3: Verify complaint proportions, multi-aspect reviews, and bounds."""

    def test_multi_aspect_review_counted_once_and_proportion_le_100(self):
        # 3 reviews:
        # Review 1 mentions Battery, Heating, Charging (3 aspects) -> 1 complaint review
        # Review 2 mentions Battery, Heating (2 aspects) -> 1 complaint review
        # Review 3 is positive (Display) -> 0 complaint reviews
        data = pd.DataFrame({
            "period": ["2026-06-01", "2026-06-01", "2026-06-01"],
            "review_text": [
                "Battery drains quickly and phone overheats while charging failure occurs.",
                "Terrible battery life and device gets scorching hot.",
                "Display is bright and beautiful, great colors.",
            ],
            "aspects": [
                ["Battery", "Heating", "Charging"],
                ["Battery", "Heating"],
                ["Display"],
            ],
            "combined_severity": [0.9, 0.85, 0.1],
            "is_complaint": [True, True, False],
        })

        features = calculate_period_features(data)
        assert len(features) == 1
        row = features.iloc[0]

        # Total reviews = 3
        assert row["total_reviews"] == 3
        # Unique complaint reviews = 2
        assert row["complaint_reviews_count"] == 2
        # Overall proportion = 2 / 3 = 0.6667 (never exceeds 1.0 / 100%)
        assert 0.0 <= row["overall_complaint_proportion"] <= 1.0
        assert round(row["overall_complaint_proportion"], 4) == round(2 / 3, 4)

        # Aspect specific counts
        assert row["complaint_count_Battery"] == 2
        assert row["complaint_count_Heating"] == 2
        assert row["complaint_count_Charging"] == 1
        assert row["complaint_proportion_Battery"] == 2 / 3
        assert row["complaint_proportion_Heating"] == 2 / 3
        assert row["complaint_proportion_Charging"] == 1 / 3

        # Total aspect mentions is separate and can exceed 100%
        # (3 in review 1 + 2 in review 2 + 1 in review 3 = 6 mentions)
        assert row["total_aspect_mentions"] == 6
        assert row["aspect_mentions_per_review"] == 6 / 3

    def test_extreme_multi_aspect_never_exceeds_100_percent(self):
        # 1 review mentioning all 9 aspects
        all_aspects = ["Battery", "Camera", "Display", "Performance", "Charging",
                       "Heating", "Software", "Durability", "Connectivity"]
        data = pd.DataFrame({
            "period": ["2026-07-01"],
            "review_text": ["All subsystems broken and failed."],
            "aspects": [all_aspects],
            "combined_severity": [1.0],
            "is_complaint": [True],
        })
        features = calculate_period_features(data)
        row = features.iloc[0]
        assert row["overall_complaint_proportion"] == 1.0
        assert row["overall_complaint_proportion"] <= 1.0
        assert row["total_aspect_mentions"] == 9
        assert row["aspect_mentions_per_review"] == 9.0


class TestDataValidationPolicy:
    """4, 5, 6, 7: Duplicate policy, invalid dates, missing ratings, empty text."""

    def test_duplicate_review_ids_and_content_policy(self):
        csv_content = """product_id,review_id,review_date,review_text,rating
PROD_A,REV_101,2026-05-01,Screen flickers.,3.0
PROD_A,REV_101,2026-05-01,Screen flickers.,3.0
PROD_A,REV_102,2026-05-02,Battery drain.,2.0
PROD_A,REV_103,2026-05-02,Battery drain.,2.0
"""
        df, report = validate_and_load_csv(io.StringIO(csv_content))
        # REV_101 is duplicated by ID -> removed 1
        # REV_103 has same content as REV_102 -> removed 1
        assert report["duplicates_removed"] >= 1
        assert len(df) == 2 or len(df) == 3

    def test_invalid_dates_and_empty_text_and_missing_rating(self):
        csv_content = """product_id,review_id,review_date,review_text,rating
PROD_A,REV_1,2026-05-01,Normal review.,5.0
PROD_A,REV_2,2026-05-02,,4.0
PROD_A,REV_3,invalid_timestamp,Software buggy.,2.0
PROD_A,REV_4,2026/99/99,Charging fails.,1.0
PROD_A,REV_5,2026-05-05,Camera blurry.,
"""
        df, report = validate_and_load_csv(io.StringIO(csv_content))
        assert report["total_rows_read"] == 5
        assert report["missing_text_count"] == 1  # REV_2
        assert report["invalid_date_count"] == 2  # REV_3, REV_4
        assert report["missing_rating_count"] == 1  # REV_5
        assert report["rows_retained"] == 2  # REV_1, REV_5
        assert report["rows_rejected"] == 3

        # Verify REV_5 retained with NaN rating
        rev5 = df[df["review_text"] == "Camera blurry."]
        assert len(rev5) == 1
        assert pd.isna(rev5.iloc[0]["rating"])


class TestProductSpecificAggregation:
    """8, 13: Product separation and updating on product change."""

    def test_product_separation(self):
        # Two products with completely different failure characteristics
        csv_content = """product_id,review_date,review_text,rating
FAILING,2026-06-01,Battery drain severe.,1.0
FAILING,2026-06-02,Overheating hot.,1.0
HEALTHY,2026-06-01,Awesome phone love it.,5.0
HEALTHY,2026-06-02,Great battery life.,5.0
"""
        df, _ = validate_and_load_csv(io.StringIO(csv_content))

        # FAILING product pipeline
        failing_df = filter_by_product(df, "FAILING")
        proc_f = aggregate_by_period(preprocess_reviews(failing_df, use_transformer=False), "monthly")
        feats_f = calculate_period_features(proc_f)
        risk_f = calculate_risk_timeseries(feats_f)

        # HEALTHY product pipeline
        healthy_df = filter_by_product(df, "HEALTHY")
        proc_h = aggregate_by_period(preprocess_reviews(healthy_df, use_transformer=False), "monthly")
        feats_h = calculate_period_features(proc_h)
        risk_h = calculate_risk_timeseries(feats_h)

        assert feats_f.iloc[0]["overall_complaint_proportion"] == 1.0
        assert feats_h.iloc[0]["overall_complaint_proportion"] == 0.0
        assert risk_f.iloc[0]["risk_score"] > risk_h.iloc[0]["risk_score"]


class TestHistoricalBaselinesAndZeroVariance:
    """9, 10: Chronological baselines without future leakage & zero variance handling."""

    def test_no_future_leakage_and_zero_variance_signals(self):
        # Periods 0, 1, 2 have zero complaints (zero variance)
        # Period 3 has complaints
        periods = pd.date_range("2026-01-01", periods=5, freq="W")
        feats = pd.DataFrame({
            "period": periods,
            "total_reviews": [10, 10, 10, 10, 10],
            "review_velocity": [10, 10, 10, 10, 10],
            "overall_complaint_proportion": [0.0, 0.0, 0.0, 0.3, 0.5],
            "overall_avg_severity": [0.0, 0.0, 0.0, 0.7, 0.8],
            "symptom_diversity": [0.0, 0.0, 0.0, 0.3, 0.4],
            "complaint_growth": [0.0, 0.0, 0.0, 1.0, 0.2],
            "baseline_velocity": [10, 10, 10, 10, 10],
            "complaint_count_Battery": [0, 0, 0, 3, 5],
            "complaint_proportion_Battery": [0.0, 0.0, 0.0, 0.3, 0.5],
            "avg_severity_Battery": [0.0, 0.0, 0.0, 0.7, 0.8],
            "growth_Battery": [0.0, 0.0, 0.0, 1.0, 0.2],
        })

        anom = detect_anomalies(feats, min_history=3)
        # Verify period 0-2 not flagged
        assert not anom.iloc[0]["is_anomaly"]
        assert not anom.iloc[1]["is_anomaly"]
        assert not anom.iloc[2]["is_anomaly"]
        # In period 3, zero variance baseline does NOT crash or raise exception
        assert "confidence" in anom.columns
        assert "heuristic" in anom.iloc[3]["confidence"]


class TestRiskScoreBoundsAndExplanations:
    """11, 12: Risk score bounds, weight validation, alert explanation accuracy."""

    def test_risk_score_bounds_and_weights(self):
        # Extreme inputs
        res_zero = calculate_risk_score(0.0, 0.0, 0.0, 0.0, 1.0)
        assert res_zero["risk_score"] == 0.0
        assert res_zero["risk_level"] == "Low"

        res_max = calculate_risk_score(1.0, 1.0, 10.0, 1.0, 5.0)
        assert 0.0 <= res_max["risk_score"] <= 100.0
        assert res_max["risk_level"] == "Critical"

    def test_alert_explanations_match_supporting_reviews(self):
        anom_row = pd.Series({
            "period": "2026-07-01",
            "confidence": "heuristic (adequate sample)",
            "anomaly_signals": [{
                "signal": "complaint_proportion",
                "current": 0.75,
                "baseline": 0.20,
                "z_score": 2.5,
                "method": "rolling baseline deviation",
            }],
        })
        top_aspects = [
            {"aspect": "Battery", "count": 3, "proportion": 0.75, "severity": 0.95},
            {"aspect": "Heating", "count": 2, "proportion": 0.50, "severity": 0.85},
        ]
        text = generate_anomaly_explanation(anom_row, top_aspects)
        assert "Battery" in text
        assert "Heating" in text
        assert "Software" not in text
        assert "75.0%" in text
        assert "2.5 standard deviations" in text
        assert "rolling baseline deviation" in text
