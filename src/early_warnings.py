"""Early Warning Notification System module.

Scans ALL products in a validated DataFrame and generates warning records
whenever anomaly detection flags an unusual period.

Design principles
-----------------
- No future data leakage: baselines use strictly prior periods.
- Each product is analyzed in isolation; no data mixing between products.
- Complaint proportions are already bounded to [0, 1] by temporal_features.
- If a product has fewer than MIN_HISTORY_PERIODS + 1 periods after preprocessing,
  a single informational status row (warning_level = "Insufficient data") is returned
  for that product rather than a fabricated warning.
- use_transformer is always False here for responsive multi-product scanning.

Output columns
--------------
product_id          : str
warning_date        : pd.Timestamp  (start of the flagged period)
period_str          : str           (human-readable period label)
warning_level       : str           ("Low" | "Moderate" | "High" | "Insufficient data")
risk_score          : float         (0–100) or NaN when insufficient data
prev_risk_score     : float         (risk score of the immediately prior period, or NaN)
affected_aspects    : str           (comma-separated list of top aspects, or empty)
reason_text         : str           (plain-language explanation)
confidence          : str           (from detect_anomalies, or "N/A")
total_reviews       : int           (reviews in the flagged period)
complaint_reviews_count : int       (complaint reviews in the flagged period)
overall_complaint_proportion : float (0–1, max 1.0)
anomaly_signals_json : str          (JSON-encoded list of signal dicts, or "[]")
"""
import json
import sys
import os
from typing import Optional

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MIN_HISTORY_PERIODS,
    RISK_BANDS,
    ASPECT_LEXICON,
)
from src.data_loader import filter_by_product, aggregate_by_period, get_products
from src.preprocessing import preprocess_reviews
from src.temporal_features import calculate_period_features
from src.risk_scoring import calculate_risk_timeseries
from src.anomaly_detection import detect_anomalies, get_top_anomaly_aspects


# ─── Warning level mapping ────────────────────────────────────────────────────
# Simplified three-level scheme for the Early Warnings tab.
# Note: the full Risk Monitoring tab uses four bands (Low/Moderate/High/Critical).
_WARN_LEVEL_THRESHOLDS = [
    (65, "High"),
    (40, "Moderate"),
    (0,  "Low"),
]


def _map_risk_to_warning_level(risk_score: float) -> str:
    """Map a 0–100 risk score to a three-level warning label."""
    for threshold, level in _WARN_LEVEL_THRESHOLDS:
        if risk_score >= threshold:
            return level
    return "Low"


def _build_reason_text(
    product_id: str,
    signals: list,
    top_aspects: list,
    risk_score: float,
    prev_risk_score: Optional[float],
) -> str:
    """Build a plain-language reason string from anomaly signals."""
    parts = []

    # Describe triggered signals
    signal_descriptions = {
        "complaint_proportion": "complaint frequency rose above its historical baseline",
        "severity":             "average complaint severity increased above its historical baseline",
        "symptom_diversity":    "the number of distinct complaint types exceeded its historical baseline",
        "review_velocity":      "review arrival rate spiked above its historical baseline",
    }
    triggered_descs = [
        signal_descriptions.get(s.get("signal", ""), s.get("signal", ""))
        for s in signals
    ]
    if triggered_descs:
        parts.append("Unusual pattern detected: " + "; ".join(triggered_descs) + ".")

    # Mention top affected aspects
    if top_aspects:
        aspect_names = [a["aspect"] for a in top_aspects]
        parts.append(f"Most affected aspects: {', '.join(aspect_names)}.")

    # Mention risk trend
    if prev_risk_score is not None and not np.isnan(prev_risk_score):
        delta = risk_score - prev_risk_score
        if delta > 0:
            parts.append(f"Risk score increased by {delta:.1f} points (from {prev_risk_score:.1f} to {risk_score:.1f}/100).")
        else:
            parts.append(f"Risk score is {risk_score:.1f}/100 (previously {prev_risk_score:.1f}/100).")

    if not parts:
        parts.append(f"Risk score of {risk_score:.1f}/100 was flagged as anomalous for {product_id}.")

    parts.append(
        "Note: This warning indicates an unusual complaint pattern, "
        "not a confirmed product failure."
    )
    return " ".join(parts)


def generate_early_warnings(
    df_validated: pd.DataFrame,
    aggregation: str = "monthly",
) -> tuple:
    """Analyze all products and return (warnings_df, status_df).

    Parameters
    ----------
    df_validated : pd.DataFrame
        Output of validate_and_load_csv — already deduplicated and date-cleaned.
    aggregation : str
        'weekly' or 'monthly'.

    Returns
    -------
    warnings_df : pd.DataFrame
        One row per anomalous period across all products.
        Empty if no anomalies were found.
    status_df : pd.DataFrame
        One row per product explaining its analysis status (e.g., insufficient
        data, no anomalies, or count of warnings generated).
    """
    if df_validated is None or df_validated.empty:
        empty_warn = _empty_warnings_df()
        empty_status = pd.DataFrame(columns=["product_id", "status", "periods", "warnings"])
        return empty_warn, empty_status

    products = get_products(df_validated)
    all_warnings = []
    status_rows = []

    for product_id in products:
        product_df = filter_by_product(df_validated, product_id)

        # ── 1. Preprocess (no transformer — multi-product scan must be fast) ──
        try:
            processed = preprocess_reviews(product_df, use_transformer=False)
        except Exception as exc:
            status_rows.append({
                "product_id": product_id,
                "status": f"Preprocessing error: {exc}",
                "periods": 0,
                "warnings": 0,
            })
            continue

        # ── 2. Aggregate by period ─────────────────────────────────────────
        processed = aggregate_by_period(processed, aggregation)

        # ── 3. Temporal features ───────────────────────────────────────────
        try:
            features_df = calculate_period_features(processed)
        except Exception as exc:
            status_rows.append({
                "product_id": product_id,
                "status": f"Feature calculation error: {exc}",
                "periods": 0,
                "warnings": 0,
            })
            continue

        n_periods = len(features_df)

        # ── 4. Insufficient history guard ──────────────────────────────────
        # Need at least MIN_HISTORY_PERIODS + 1 periods so that detect_anomalies
        # has at least one period to evaluate against MIN_HISTORY_PERIODS of baseline.
        if n_periods < MIN_HISTORY_PERIODS + 1:
            # Emit one informational row per product instead of inventing a warning
            warn_row = _insufficient_data_row(product_id, features_df, n_periods)
            all_warnings.append(warn_row)
            status_rows.append({
                "product_id": product_id,
                "status": "Insufficient historical data",
                "periods": n_periods,
                "warnings": 0,
            })
            continue

        # ── 5. Risk scoring ────────────────────────────────────────────────
        try:
            risk_df = calculate_risk_timeseries(features_df)
        except Exception as exc:
            status_rows.append({
                "product_id": product_id,
                "status": f"Risk scoring error: {exc}",
                "periods": n_periods,
                "warnings": 0,
            })
            continue

        # ── 6. Anomaly detection ───────────────────────────────────────────
        try:
            anomaly_df = detect_anomalies(risk_df)
        except Exception as exc:
            status_rows.append({
                "product_id": product_id,
                "status": f"Anomaly detection error: {exc}",
                "periods": n_periods,
                "warnings": 0,
            })
            continue

        # ── 7. Emit warning rows for flagged periods ───────────────────────
        product_warnings = 0
        for i, row in anomaly_df.iterrows():
            if not row.get("is_anomaly", False):
                continue

            period_idx = anomaly_df.index.get_loc(i)
            risk_score = float(row.get("risk_score", 0.0))
            warning_level = _map_risk_to_warning_level(risk_score)

            # Previous period risk score (no future leakage — use period_idx - 1)
            prev_risk_score = float(np.nan)
            if period_idx > 0:
                prev_risk_score = float(anomaly_df.iloc[period_idx - 1].get("risk_score", np.nan))

            # Top affected aspects
            top_aspects = get_top_anomaly_aspects(
                anomaly_df,
                period_idx=period_idx,
                aspects=list(ASPECT_LEXICON.keys()),
                top_n=3,
            )
            affected_aspects = ", ".join([a["aspect"] for a in top_aspects])

            # Anomaly signals
            raw_signals = row.get("anomaly_signals", [])
            if isinstance(raw_signals, str):
                try:
                    raw_signals = json.loads(raw_signals)
                except (json.JSONDecodeError, ValueError):
                    raw_signals = []
            signals_list = raw_signals if isinstance(raw_signals, list) else []

            # Reason text
            reason = _build_reason_text(
                product_id, signals_list, top_aspects, risk_score, prev_risk_score
            )

            # Period label
            period_val = row.get("period", pd.NaT)
            if pd.isna(period_val):
                period_str = "Unknown"
            else:
                period_str = (
                    pd.Timestamp(period_val).strftime("%Y-%m") 
                    if aggregation == "monthly"
                    else pd.Timestamp(period_val).strftime("%Y-W%W")
                )

            all_warnings.append({
                "product_id": product_id,
                "warning_date": period_val,
                "period_str": period_str,
                "warning_level": warning_level,
                "risk_score": risk_score,
                "prev_risk_score": prev_risk_score,
                "affected_aspects": affected_aspects,
                "reason_text": reason,
                "confidence": str(row.get("confidence", "N/A")),
                "total_reviews": int(row.get("total_reviews", 0)),
                "complaint_reviews_count": int(row.get("complaint_reviews_count", 0)),
                "overall_complaint_proportion": float(row.get("overall_complaint_proportion", 0.0)),
                "anomaly_signals_json": json.dumps(signals_list),
            })
            product_warnings += 1

        if product_warnings == 0:
            status_rows.append({
                "product_id": product_id,
                "status": "No anomalies detected",
                "periods": n_periods,
                "warnings": 0,
            })
        else:
            status_rows.append({
                "product_id": product_id,
                "status": "Warnings generated",
                "periods": n_periods,
                "warnings": product_warnings,
            })

    warnings_df = (
        pd.DataFrame(all_warnings) if all_warnings else _empty_warnings_df()
    )
    status_df = (
        pd.DataFrame(status_rows)
        if status_rows
        else pd.DataFrame(columns=["product_id", "status", "periods", "warnings"])
    )
    return warnings_df, status_df


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _empty_warnings_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "product_id", "warning_date", "period_str", "warning_level",
        "risk_score", "prev_risk_score", "affected_aspects", "reason_text",
        "confidence", "total_reviews", "complaint_reviews_count",
        "overall_complaint_proportion", "anomaly_signals_json",
    ])


def _insufficient_data_row(
    product_id: str,
    features_df: pd.DataFrame,
    n_periods: int,
) -> dict:
    """Build a status-only row for products with too little history."""
    # Use the last available period date if possible, else NaT
    if not features_df.empty and "period" in features_df.columns:
        last_period = features_df["period"].iloc[-1]
    else:
        last_period = pd.NaT

    return {
        "product_id": product_id,
        "warning_date": last_period,
        "period_str": "N/A",
        "warning_level": "Insufficient data",
        "risk_score": float("nan"),
        "prev_risk_score": float("nan"),
        "affected_aspects": "",
        "reason_text": (
            f"Insufficient historical data for {product_id}: only {n_periods} "
            f"period(s) found (minimum required: {MIN_HISTORY_PERIODS + 1}). "
            "Expand the date range or upload more reviews to enable warning generation."
        ),
        "confidence": "N/A",
        "total_reviews": int(features_df["total_reviews"].sum()) if not features_df.empty and "total_reviews" in features_df.columns else 0,
        "complaint_reviews_count": 0,
        "overall_complaint_proportion": float("nan"),
        "anomaly_signals_json": "[]",
    }
