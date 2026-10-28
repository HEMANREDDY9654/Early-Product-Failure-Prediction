"""Tests for evaluation module."""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation import (
    frequency_baseline_predict,
    multisignal_predict,
    compare_approaches,
    evaluate_synthetic_scenario,
    ablation_summary,
)
from src.risk_scoring import calculate_risk_timeseries, ablation_analysis
from src.temporal_features import calculate_period_features


def _make_dummy_features_and_risk(n_periods=10):
    periods = pd.date_range("2024-01-01", periods=n_periods, freq="W")
    features_df = pd.DataFrame({
        "period": periods,
        "total_reviews": [15] * n_periods,
        "review_velocity": [15] * n_periods,
        "overall_complaint_proportion": np.linspace(0.1, 0.8, n_periods),
        "overall_avg_severity": np.linspace(0.3, 0.9, n_periods),
        "complaint_growth": np.linspace(0.0, 0.3, n_periods),
        "symptom_diversity": np.linspace(0.1, 0.8, n_periods),
        "baseline_velocity": [15] * n_periods,
    })
    risk_df = calculate_risk_timeseries(features_df)
    return features_df, risk_df


class TestFrequencyBaseline:
    """Test frequency baseline logic."""

    def test_frequency_baseline_predictions(self):
        features_df, _ = _make_dummy_features_and_risk()
        preds = frequency_baseline_predict(features_df, threshold=0.3, min_history=3)
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(features_df)
        # First 3 periods should not be flagged (min_history=3)
        assert not preds.iloc[:3].any()
        # High complaint proportion periods (>0.3) should be flagged
        assert preds.iloc[4]  # 0.35 >= 0.30

    def test_empty_features(self):
        preds = frequency_baseline_predict(pd.DataFrame())
        assert len(preds) == 0


class TestMultisignalPredict:
    """Test multi-signal risk prediction."""

    def test_multisignal_predictions(self):
        features_df, risk_df = _make_dummy_features_and_risk()
        preds = multisignal_predict(risk_df, risk_threshold=50.0, min_history=3)
        assert len(preds) == len(risk_df)
        assert not preds.iloc[:3].any()

    def test_empty_risk_df(self):
        preds = multisignal_predict(pd.DataFrame())
        assert len(preds) == 0


class TestCompareApproaches:
    """Test comparison between baseline and multi-signal approaches."""

    def test_compare_without_true_labels(self):
        features_df, risk_df = _make_dummy_features_and_risk()
        comp = compare_approaches(features_df, risk_df, baseline_threshold=0.3, risk_threshold=50.0)
        assert comp["has_true_labels"] is False
        assert "agreement" in comp
        assert "agreement_rate" in comp["agreement"]
        assert "note" in comp

    def test_compare_with_true_labels(self):
        features_df, risk_df = _make_dummy_features_and_risk(10)
        # Ground truth: last 4 periods have failure
        true_labels = pd.Series([False] * 6 + [True] * 4)
        comp = compare_approaches(features_df, risk_df, true_labels=true_labels)
        assert comp["has_true_labels"] is True
        assert "baseline_metrics" in comp
        assert "multisignal_metrics" in comp
        assert "f1" in comp["baseline_metrics"]
        assert "f1" in comp["multisignal_metrics"]


class TestSyntheticScenarioAndAblation:
    """Test evaluation on synthetic scenarios and ablation summaries."""

    def test_synthetic_scenario_evaluation(self):
        features_df, risk_df = _make_dummy_features_and_risk(10)
        anomaly_df = risk_df.copy()
        anomaly_df["is_anomaly"] = [False] * 5 + [True] * 5

        res = evaluate_synthetic_scenario(risk_df, anomaly_df, injection_start_period=5)
        assert res["scenario_type"] == "synthetic_test"
        assert "disclaimer" in res
        assert "risk_score_metrics" in res
        assert "anomaly_detection_metrics" in res

    def test_ablation_summary(self):
        features_df, _ = _make_dummy_features_and_risk(8)
        ablation_dict = ablation_analysis(features_df)
        summary = ablation_summary(ablation_dict)
        assert isinstance(summary, pd.DataFrame)
        assert "configuration" in summary.columns
        assert "mean_risk_score" in summary.columns
        assert len(summary) == len(ablation_dict)
