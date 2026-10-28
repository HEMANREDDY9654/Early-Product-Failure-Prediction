"""Tests for data_loader module."""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from io import StringIO
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import (
    auto_map_columns,
    validate_and_load_csv,
    get_products,
    get_date_range,
    filter_by_product,
    aggregate_by_period,
)


def _make_csv(data: str) -> StringIO:
    """Helper to create a CSV StringIO from text."""
    return StringIO(data)


class TestAutoMapColumns:
    """Test column name auto-mapping."""

    def test_standard_names(self):
        df = pd.DataFrame(columns=["product_id", "review_date", "review_text", "rating"])
        mapping = auto_map_columns(df)
        assert mapping["product_id"] == "product_id"
        assert mapping["review_date"] == "review_date"
        assert mapping["review_text"] == "review_text"
        assert mapping["rating"] == "rating"

    def test_alternative_names(self):
        df = pd.DataFrame(columns=["asin", "timestamp", "body", "overall"])
        mapping = auto_map_columns(df)
        assert mapping["product_id"] == "asin"
        assert mapping["review_date"] == "timestamp"
        assert mapping["review_text"] == "body"
        assert mapping["rating"] == "overall"

    def test_missing_columns(self):
        df = pd.DataFrame(columns=["some_col", "another_col"])
        mapping = auto_map_columns(df)
        assert mapping["product_id"] is None
        assert mapping["review_text"] is None


class TestValidateAndLoadCSV:
    """Test CSV loading and validation."""

    def test_valid_csv(self):
        csv = _make_csv(
            "product_id,review_date,review_text,rating\n"
            "P1,2024-01-15,Great product,5\n"
            "P1,2024-02-20,Not good battery,2\n"
        )
        df, report = validate_and_load_csv(csv)
        assert len(df) == 2
        assert report["rows_retained"] == 2
        assert report["rows_rejected"] == 0

    def test_missing_text_rows(self):
        csv = _make_csv(
            "product_id,review_date,review_text,rating\n"
            "P1,2024-01-15,Great product,5\n"
            "P1,2024-02-20,,2\n"
            "P1,2024-03-10,   ,3\n"
        )
        df, report = validate_and_load_csv(csv)
        assert len(df) == 1
        assert report["missing_text_count"] == 2

    def test_invalid_dates(self):
        csv = _make_csv(
            "product_id,review_date,review_text,rating\n"
            "P1,2024-01-15,Great product,5\n"
            "P1,not_a_date,Bad product,1\n"
        )
        df, report = validate_and_load_csv(csv)
        assert len(df) == 1
        assert report["invalid_date_count"] == 1

    def test_duplicate_removal(self):
        csv = _make_csv(
            "product_id,review_date,review_text,rating\n"
            "P1,2024-01-15,Same review,5\n"
            "P1,2024-01-15,Same review,5\n"
            "P1,2024-01-16,Different review,4\n"
        )
        df, report = validate_and_load_csv(csv)
        assert len(df) == 2
        assert report["duplicates_removed"] == 1

    def test_empty_csv(self):
        csv = _make_csv("product_id,review_date,review_text,rating\n")
        df, report = validate_and_load_csv(csv)
        assert len(df) == 0

    def test_missing_required_columns(self):
        csv = _make_csv("col1,col2\na,b\n")
        df, report = validate_and_load_csv(csv)
        assert len(df) == 0
        assert len(report["issues"]) > 0

    def test_chronological_sorting(self):
        csv = _make_csv(
            "product_id,review_date,review_text,rating\n"
            "P1,2024-03-15,Third,5\n"
            "P1,2024-01-15,First,4\n"
            "P1,2024-02-15,Second,3\n"
        )
        df, report = validate_and_load_csv(csv)
        dates = df["review_date"].tolist()
        assert dates == sorted(dates)

    def test_rating_clipping(self):
        csv = _make_csv(
            "product_id,review_date,review_text,rating\n"
            "P1,2024-01-15,Good,10\n"
            "P1,2024-01-16,Bad,0\n"
        )
        df, report = validate_and_load_csv(csv)
        assert df["rating"].max() <= 5
        assert df["rating"].min() >= 1

    def test_unix_timestamp_parsing(self):
        csv = _make_csv(
            "product_id,review_date,review_text,rating\n"
            "P1,1705276800,Good product,5\n"
        )
        df, report = validate_and_load_csv(csv)
        assert len(df) == 1
        assert pd.notna(df["review_date"].iloc[0])

    def test_alternative_column_names_mapping(self):
        csv = _make_csv(
            "asin,timestamp,review_body,overall\n"
            "B001,2024-01-15,Great phone,5\n"
        )
        df, report = validate_and_load_csv(csv)
        assert len(df) == 1
        assert "product_id" in df.columns
        assert "review_text" in df.columns


class TestHelperFunctions:
    """Test helper utility functions."""

    def test_get_products(self):
        df = pd.DataFrame({"product_id": ["P2", "P1", "P1", "P3"]})
        products = get_products(df)
        assert products == ["P1", "P2", "P3"]

    def test_get_products_empty(self):
        assert get_products(pd.DataFrame()) == []

    def test_filter_by_product(self):
        df = pd.DataFrame({
            "product_id": ["P1", "P2", "P1"],
            "review_text": ["a", "b", "c"],
        })
        filtered = filter_by_product(df, "P1")
        assert len(filtered) == 2

    def test_aggregate_weekly(self):
        df = pd.DataFrame({
            "review_date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-10"]),
            "review_text": ["a", "b", "c"],
        })
        result = aggregate_by_period(df, "weekly")
        assert "period" in result.columns

    def test_aggregate_monthly(self):
        df = pd.DataFrame({
            "review_date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-10"]),
            "review_text": ["a", "b", "c"],
        })
        result = aggregate_by_period(df, "monthly")
        assert "period" in result.columns
