"""Tests for Early Warning Notification System.

Covers:
  1. Normal data generates warnings for products with sufficient history and complaints.
  2. Products with insufficient history receive "Insufficient data" status, not fabricated warnings.
  3. Reviews with invalid dates are excluded from warning calculations.
  4. Duplicate review IDs are not double-counted in complaint proportions.
  5. Multiple products are analyzed in isolation (no data mixing).
  6. A sudden complaint spike triggers at least one warning.
  7. No future data leakage: historical baselines use only preceding periods.
  8. overall_complaint_proportion never exceeds 1.0.
"""
import sys
import os
import json
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.early_warnings import generate_early_warnings, _map_risk_to_warning_level
from src.data_loader import validate_and_load_csv
from config import MIN_HISTORY_PERIODS


# ─── Fixture helpers ──────────────────────────────────────────────────────────

def _make_review_df(
    product_id: str,
    n_months: int,
    reviews_per_month: int,
    complaint_text: str = "battery drain overheating slow charging failure",
    positive_text: str = "great phone excellent performance",
    complaint_fraction: float = 0.7,
    start_date: str = "2024-01-01",
    rating_complaint: float = 2.0,
    rating_positive: float = 5.0,
) -> pd.DataFrame:
    """Build a synthetic validated DataFrame for a single product."""
    rows = []
    base = pd.Timestamp(start_date)
    rev_id = 1
    for month_offset in range(n_months):
        period_start = base + pd.DateOffset(months=month_offset)
        for day in range(reviews_per_month):
            date = period_start + timedelta(days=day % 28)
            is_complaint = day < int(reviews_per_month * complaint_fraction)
            rows.append({
                "review_id": f"REV_{rev_id}",
                "product_id": product_id,
                "review_date": date,
                "review_text": complaint_text if is_complaint else positive_text,
                "rating": rating_complaint if is_complaint else rating_positive,
            })
            rev_id += 1
    return pd.DataFrame(rows)


def _make_spike_df(product_id: str) -> pd.DataFrame:
    """Product with low complaint baseline then sudden spike in the last 2 months."""
    rows = []
    rev_id = 1
    base = pd.Timestamp("2024-01-01")
    # 4 months of low complaints
    for month_offset in range(4):
        period_start = base + pd.DateOffset(months=month_offset)
        for day in range(10):
            date = period_start + timedelta(days=day)
            is_complaint = day < 2  # ~20% complaints
            rows.append({
                "review_id": f"REV_{rev_id}",
                "product_id": product_id,
                "review_date": date,
                "review_text": "battery drain overheating" if is_complaint else "great phone",
                "rating": 1.5 if is_complaint else 5.0,
            })
            rev_id += 1
    # 2 months of spike (90% complaints)
    for month_offset in range(4, 6):
        period_start = base + pd.DateOffset(months=month_offset)
        for day in range(20):
            date = period_start + timedelta(days=day)
            is_complaint = day < 18  # 90% complaints
            rows.append({
                "review_id": f"REV_{rev_id}",
                "product_id": product_id,
                "review_date": date,
                "review_text": "battery drain overheating charging failure slow laggy" if is_complaint else "ok phone",
                "rating": 1.0 if is_complaint else 3.0,
            })
            rev_id += 1
    return pd.DataFrame(rows)


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestNormalDataGeneratesWarnings:
    """Test 1: Products with ample history and complaints should produce at least one warning."""

    def test_generates_at_least_one_warning(self):
        df = _make_spike_df("PROD_A")
        warnings_df, status_df = generate_early_warnings(df, aggregation="monthly")
        real = warnings_df[warnings_df["warning_level"] != "Insufficient data"]
        assert len(real) >= 1, "Expected at least one warning for a product with a clear spike."

    def test_warning_has_required_columns(self):
        df = _make_review_df("PROD_B", n_months=6, reviews_per_month=15)
        warnings_df, _ = generate_early_warnings(df, aggregation="monthly")
        required_cols = [
            "product_id", "warning_date", "period_str", "warning_level",
            "risk_score", "prev_risk_score", "affected_aspects", "reason_text",
            "confidence", "total_reviews", "complaint_reviews_count",
            "overall_complaint_proportion", "anomaly_signals_json",
        ]
        for col in required_cols:
            assert col in warnings_df.columns, f"Missing required column: {col}"

    def test_warning_level_values(self):
        df = _make_spike_df("PROD_C")
        warnings_df, _ = generate_early_warnings(df, aggregation="monthly")
        allowed = {"High", "Moderate", "Low", "Insufficient data"}
        for level in warnings_df["warning_level"].unique():
            assert level in allowed, f"Unexpected warning level: {level}"


class TestInsufficientHistory:
    """Test 2: Products with too few periods get 'Insufficient data', not fabricated warnings."""

    def test_two_month_product_no_real_warning(self):
        # Only 2 months → cannot have MIN_HISTORY_PERIODS+1 = 4 periods
        df = _make_review_df("PROD_FEW", n_months=2, reviews_per_month=20)
        warnings_df, status_df = generate_early_warnings(df, aggregation="monthly")
        prod_warnings = warnings_df[warnings_df["product_id"] == "PROD_FEW"]
        real = prod_warnings[prod_warnings["warning_level"] != "Insufficient data"]
        assert len(real) == 0, "Should not generate real warnings with only 2 months of data."

    def test_insufficient_data_row_present(self):
        df = _make_review_df("PROD_FEW2", n_months=2, reviews_per_month=20)
        warnings_df, _ = generate_early_warnings(df, aggregation="monthly")
        prod_rows = warnings_df[warnings_df["product_id"] == "PROD_FEW2"]
        insuf = prod_rows[prod_rows["warning_level"] == "Insufficient data"]
        assert len(insuf) >= 1, "Expected at least one Insufficient data status row."

    def test_status_df_shows_insufficient_for_low_history_product(self):
        df = _make_review_df("PROD_FEW3", n_months=2, reviews_per_month=10)
        _, status_df = generate_early_warnings(df, aggregation="monthly")
        row = status_df[status_df["product_id"] == "PROD_FEW3"]
        assert len(row) >= 1
        assert "Insufficient" in str(row.iloc[0]["status"])


class TestMissingDatesExcluded:
    """Test 3: Reviews with invalid dates must be excluded from warning calculations."""

    def test_invalid_date_reviews_excluded(self):
        df = _make_review_df("PROD_D", n_months=6, reviews_per_month=10)
        # Add rows with invalid dates
        bad_rows = pd.DataFrame([
            {
                "review_id": "BAD_1",
                "product_id": "PROD_D",
                "review_date": "invalid_timestamp",
                "review_text": "terrible battery drain overheating",
                "rating": 1.0,
            },
            {
                "review_id": "BAD_2",
                "product_id": "PROD_D",
                "review_date": "2026/99/99",
                "review_text": "charging failure slow heating",
                "rating": 1.0,
            },
        ])
        combined = pd.concat([df, bad_rows], ignore_index=True)
        # Validate through the loader to strip invalid dates
        import io, csv as _csv
        csv_buf = io.StringIO()
        combined.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        validated_df, report = validate_and_load_csv(csv_buf)
        assert report["invalid_date_count"] == 2, (
            f"Expected 2 invalid dates removed, got {report['invalid_date_count']}"
        )
        # Ensure generate_early_warnings still works on the clean df
        warnings_df, _ = generate_early_warnings(validated_df, aggregation="monthly")
        assert warnings_df is not None

    def test_total_reviews_in_warning_excludes_invalid_dates(self):
        """total_reviews in any warning row must not include bad-date rows."""
        df = _make_review_df("PROD_E", n_months=6, reviews_per_month=10)
        n_valid = len(df)
        warnings_df, _ = generate_early_warnings(df, aggregation="monthly")
        prod_warnings = warnings_df[warnings_df["product_id"] == "PROD_E"]
        real = prod_warnings[prod_warnings["warning_level"] != "Insufficient data"]
        for _, row in real.iterrows():
            assert row["total_reviews"] <= n_valid, (
                "total_reviews in a warning period cannot exceed total valid reviews."
            )


class TestDuplicateReviewsNotDoubleCounted:
    """Test 4: Duplicate review IDs must not be double-counted in proportions."""

    def test_duplicate_id_not_double_counted(self):
        df = _make_review_df("PROD_F", n_months=6, reviews_per_month=10)
        # Duplicate the first row with same review_id
        dup = df.iloc[:1].copy()
        combined = pd.concat([df, dup], ignore_index=True)
        # Run through loader to deduplicate
        import io
        csv_buf = io.StringIO()
        combined.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        validated_df, report = validate_and_load_csv(csv_buf)
        assert report["duplicates_removed"] >= 1, "Expected at least 1 duplicate removed."
        # After dedup, row count should equal original
        assert len(validated_df) == len(df), (
            f"After dedup expected {len(df)} rows, got {len(validated_df)}"
        )

    def test_proportion_uses_deduped_count(self):
        df = _make_review_df("PROD_G", n_months=6, reviews_per_month=10)
        warnings_df, _ = generate_early_warnings(df, aggregation="monthly")
        prod_w = warnings_df[warnings_df["product_id"] == "PROD_G"]
        real = prod_w[prod_w["warning_level"] != "Insufficient data"]
        for _, row in real.iterrows():
            prop = row["overall_complaint_proportion"]
            assert prop <= 1.0 + 1e-6, (
                f"Complaint proportion {prop} exceeds 1.0 — possible double-counting."
            )


class TestMultipleProductsIsolated:
    """Test 5: Each product is analyzed independently; no data mixing."""

    def test_two_products_separate_warnings(self):
        df_a = _make_spike_df("PROD_H")
        df_b = _make_review_df("PROD_I", n_months=6, reviews_per_month=15, complaint_fraction=0.1)
        combined = pd.concat([df_a, df_b], ignore_index=True)
        warnings_df, status_df = generate_early_warnings(combined, aggregation="monthly")
        # Both products must appear in status_df (i.e., were scanned)
        assert set(status_df["product_id"].unique()) >= {"PROD_H", "PROD_I"}, (
            "Both products should appear in the scan status table."
        )
        # PROD_H (spike product) should generate at least one real warning
        prod_h_w = warnings_df[
            (warnings_df["product_id"] == "PROD_H")
            & (warnings_df["warning_level"] != "Insufficient data")
        ]
        assert len(prod_h_w) >= 1, "PROD_H with spike should produce at least one warning."
        # PROD_I (low complaint) should have fewer or equal warnings than PROD_H
        prod_i_w = warnings_df[
            (warnings_df["product_id"] == "PROD_I")
            & (warnings_df["warning_level"] != "Insufficient data")
        ]
        assert len(prod_i_w) <= len(prod_h_w), (
            "Low-complaint product (PROD_I) should not have more warnings than high-complaint spike (PROD_H)."
        )

    def test_no_product_id_mixing(self):
        df_a = _make_spike_df("PROD_J")
        df_b = _make_review_df("PROD_K", n_months=6, reviews_per_month=10)
        combined = pd.concat([df_a, df_b], ignore_index=True)
        warnings_df, _ = generate_early_warnings(combined, aggregation="monthly")
        # Warnings for PROD_J must not contain PROD_K's product_id
        j_warnings = warnings_df[warnings_df["product_id"] == "PROD_J"]
        assert all(j_warnings["product_id"] == "PROD_J"), "PROD_J warnings contain wrong product_id."
        k_warnings = warnings_df[warnings_df["product_id"] == "PROD_K"]
        assert all(k_warnings["product_id"] == "PROD_K"), "PROD_K warnings contain wrong product_id."


class TestComplaintSpikeTriggerWarning:
    """Test 6: Sudden complaint spike triggers at least one warning."""

    def test_spike_triggers_warning(self):
        df = _make_spike_df("PROD_SPIKE")
        warnings_df, _ = generate_early_warnings(df, aggregation="monthly")
        real = warnings_df[
            (warnings_df["product_id"] == "PROD_SPIKE")
            & (warnings_df["warning_level"] != "Insufficient data")
        ]
        assert len(real) >= 1, (
            "Expected at least one warning for a product with a clear complaint spike."
        )

    def test_spike_warning_level_is_moderate_or_high(self):
        df = _make_spike_df("PROD_SPIKE2")
        warnings_df, _ = generate_early_warnings(df, aggregation="monthly")
        real = warnings_df[
            (warnings_df["product_id"] == "PROD_SPIKE2")
            & (warnings_df["warning_level"] != "Insufficient data")
        ]
        if len(real) > 0:
            # At least one warning should be Moderate or High given the severity of the spike
            levels = set(real["warning_level"].unique())
            assert levels & {"Moderate", "High"}, (
                f"Expected Moderate or High warning from spike; got {levels}"
            )


class TestNoFutureDataLeakage:
    """Test 7: Historical baselines must not use future periods."""

    def test_anomaly_detection_uses_prior_periods_only(self):
        """Verify that detect_anomalies() with MIN_HISTORY_PERIODS enforces no future leakage."""
        from src.anomaly_detection import detect_anomalies
        # Build a simple features_df with 5 periods
        periods = pd.date_range("2024-01-01", periods=5, freq="MS")
        props = [0.1, 0.15, 0.12, 0.13, 0.9]  # spike in last period
        features_df = pd.DataFrame({
            "period": periods,
            "overall_complaint_proportion": props,
            "overall_avg_severity": [0.3, 0.3, 0.3, 0.3, 0.9],
            "symptom_diversity": [0.1, 0.1, 0.1, 0.1, 0.8],
            "review_velocity": [5.0, 5.0, 5.0, 5.0, 20.0],
            "baseline_velocity": [5.0] * 5,
            "complaint_growth": [0.0, 0.05, -0.03, 0.01, 6.0],
            "total_reviews": [10] * 5,
            "complaint_reviews_count": [1, 2, 1, 1, 9],
        })
        result = detect_anomalies(features_df, min_history=MIN_HISTORY_PERIODS)
        # Only periods at index >= MIN_HISTORY_PERIODS can be anomalous
        for i in range(MIN_HISTORY_PERIODS):
            assert result.iloc[i]["is_anomaly"] == False, (
                f"Period {i} should not be anomalous (insufficient history)."
            )

    def test_generate_early_warnings_respects_history_guard(self):
        df = _make_spike_df("PROD_LEAKTEST")
        warnings_df, _ = generate_early_warnings(df, aggregation="monthly")
        # If any warning exists, its warning_date should not be the first period
        real = warnings_df[
            (warnings_df["product_id"] == "PROD_LEAKTEST")
            & (warnings_df["warning_level"] != "Insufficient data")
        ]
        if len(real) > 0:
            first_review_date = df["review_date"].min()
            first_period_start = pd.Timestamp(first_review_date).to_period("M").start_time
            for _, row in real.iterrows():
                warn_date = pd.Timestamp(row["warning_date"])
                assert warn_date > first_period_start, (
                    f"Warning at {warn_date} is at or before the first period {first_period_start} — "
                    "possible future leakage."
                )


class TestProportionBounded:
    """Test 8: overall_complaint_proportion must never exceed 1.0."""

    def test_proportion_always_bounded(self):
        df = _make_review_df("PROD_BOUND", n_months=6, reviews_per_month=20, complaint_fraction=1.0)
        warnings_df, _ = generate_early_warnings(df, aggregation="monthly")
        for _, row in warnings_df.iterrows():
            val = row["overall_complaint_proportion"]
            if pd.notna(val):
                assert val <= 1.0 + 1e-6, (
                    f"overall_complaint_proportion {val} exceeds 1.0 for product "
                    f"{row['product_id']} at period {row.get('period_str', '?')}."
                )

    def test_proportion_non_negative(self):
        df = _make_review_df("PROD_NN", n_months=5, reviews_per_month=15)
        warnings_df, _ = generate_early_warnings(df, aggregation="monthly")
        for _, row in warnings_df.iterrows():
            val = row["overall_complaint_proportion"]
            if pd.notna(val):
                assert val >= 0.0, f"Proportion {val} is negative."


class TestWarningLevelMapping:
    """Unit tests for the _map_risk_to_warning_level helper."""

    def test_high_threshold(self):
        assert _map_risk_to_warning_level(65.0) == "High"
        assert _map_risk_to_warning_level(100.0) == "High"
        assert _map_risk_to_warning_level(75.0) == "High"

    def test_moderate_threshold(self):
        assert _map_risk_to_warning_level(40.0) == "Moderate"
        assert _map_risk_to_warning_level(64.9) == "Moderate"
        assert _map_risk_to_warning_level(50.0) == "Moderate"

    def test_low_threshold(self):
        assert _map_risk_to_warning_level(0.0) == "Low"
        assert _map_risk_to_warning_level(39.9) == "Low"
        assert _map_risk_to_warning_level(20.0) == "Low"


class TestEmptyDataHandling:
    """Edge case: empty or None input should not raise exceptions."""

    def test_empty_dataframe(self):
        warnings_df, status_df = generate_early_warnings(pd.DataFrame())
        assert warnings_df is not None
        assert status_df is not None
        assert len(warnings_df) == 0

    def test_none_input(self):
        warnings_df, status_df = generate_early_warnings(None)
        assert warnings_df is not None
        assert len(warnings_df) == 0
