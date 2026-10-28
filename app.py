"""
Early Product Failure Prediction from Review Evolution
=======================================================
AI-Based Early Warning System for Emerging Product Problems

Streamlit Dashboard Application

Run with: streamlit run app.py
"""
import json as _json_mod
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ASPECT_LEXICON, DEFAULT_RISK_WEIGHTS, RISK_BANDS,
    COLUMN_ALIASES, DATA_DIR, AGGREGATION_OPTIONS,
    ANOMALY_THRESHOLD, MIN_HISTORY_PERIODS, MIN_REVIEWS_FOR_WARNING,
    BASELINE_COMPLAINT_THRESHOLD,
)
from src.data_loader import (
    validate_and_load_csv, auto_map_columns, get_products,
    get_date_range, filter_by_product, aggregate_by_period,
)
from src.preprocessing import preprocess_reviews
from src.aspect_extraction import get_supported_aspects
from src.temporal_features import calculate_period_features
from src.risk_scoring import (
    calculate_risk_timeseries, ablation_analysis, classify_risk,
)
from src.anomaly_detection import (
    detect_anomalies, get_anomaly_periods, get_top_anomaly_aspects,
)
from src.evaluation import (
    frequency_baseline_predict, multisignal_predict,
    compare_approaches, ablation_summary,
)
from src.explanations import (
    generate_risk_explanation, generate_anomaly_explanation,
    generate_comparison_explanation,
)
from src.early_warnings import generate_early_warnings

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Product Failure Prediction",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Design System CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
[data-testid="stAppViewContainer"] { background-color: #0f1117; color: #e0e0e0; }
[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #252d3f;
    min-width: 230px !important;
}
[data-testid="stSidebar"] .stRadio > div { gap: 2px; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.85rem; color: #9aa3b5; padding: 5px 8px;
    border-radius: 5px; cursor: pointer;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #f0a500; background: #1e253a; }
/* Main content */
.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2rem !important;
    max-width: 1180px !important;
}
/* Page header */
.page-header {
    margin-bottom: 1.2rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid #252d3f;
}
.page-title { font-size: 1.45rem; font-weight: 700; color: #eaecf0; margin: 0; line-height: 1.3; }
.page-subtitle { font-size: 0.82rem; color: #7b859c; margin: 0.15rem 0 0 0; }
/* Section headings */
.section-heading {
    font-size: 0.88rem; font-weight: 600; color: #c0c8d8;
    margin: 1.4rem 0 0.55rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #252d3f;
    letter-spacing: 0.02em;
}
/* Metric cards */
.dash-card {
    background: #191f30; border: 1px solid #252d3f;
    border-radius: 8px; padding: 0.9rem 1.1rem; height: 100%;
}
.dash-card-title {
    font-size: 0.68rem; font-weight: 600; color: #7b859c;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.3rem;
}
.dash-card-value { font-size: 1.55rem; font-weight: 700; color: #eaecf0; line-height: 1.2; }
.dash-card-sub { font-size: 0.72rem; color: #5a6375; margin-top: 0.12rem; }
/* Sidebar brand */
.sidebar-brand {
    font-size: 0.78rem; font-weight: 700; color: #f0a500;
    letter-spacing: 0.06em; text-transform: uppercase; padding: 0.4rem 0;
}
.sidebar-tagline { font-size: 0.69rem; color: #4b5568; margin-bottom: 0.8rem; }
.sidebar-divider { border-top: 1px solid #252d3f; margin: 0.6rem 0; }
/* Status pills */
.status-pill { display: inline-block; border-radius: 10px; padding: 2px 10px; font-size: 0.74rem; font-weight: 600; }
.status-loaded { background: #0c2a1a; color: #4ade80; border: 1px solid #166534; }
.status-nodata { background: #2a1e0a; color: #fb923c; border: 1px solid #7c2d12; }
/* Notice boxes */
.notice-box {
    background: #191f30; border-left: 3px solid #f0a500;
    border-radius: 0 6px 6px 0; padding: 0.6rem 0.9rem;
    font-size: 0.82rem; color: #9aa3b5; margin: 0.65rem 0;
}
.notice-box-info {
    background: #0f1e30; border-left: 3px solid #3b82f6;
    border-radius: 0 6px 6px 0; padding: 0.6rem 0.9rem;
    font-size: 0.82rem; color: #9aa3b5; margin: 0.65rem 0;
}
.notice-box-warn {
    background: #2a1a0a; border-left: 3px solid #f59e0b;
    border-radius: 0 6px 6px 0; padding: 0.6rem 0.9rem;
    font-size: 0.82rem; color: #9aa3b5; margin: 0.65rem 0;
}
/* Warning badges */
.badge-high { background:#3d1515; color:#f87171; border:1px solid #7f1d1d; border-radius:4px; padding:2px 9px; font-size:0.72rem; font-weight:700; }
.badge-mod  { background:#3d2e0a; color:#fbbf24; border:1px solid #78350f; border-radius:4px; padding:2px 9px; font-size:0.72rem; font-weight:700; }
.badge-low  { background:#0f2e1a; color:#34d399; border:1px solid #064e3b; border-radius:4px; padding:2px 9px; font-size:0.72rem; font-weight:700; }
/* Empty state */
.empty-state {
    text-align: center; padding: 2.5rem 1rem;
    color: #5a6375; border: 1px dashed #252d3f;
    border-radius: 8px; margin: 1rem 0;
}
.empty-state-icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
.empty-state-msg { font-size: 0.9rem; }
.empty-state-hint { font-size: 0.77rem; color: #4b5568; margin-top: 0.3rem; }
/* Metric override */
[data-testid="stMetricValue"] { color: #eaecf0 !important; font-size: 1.35rem !important; }
[data-testid="stMetricLabel"] { color: #7b859c !important; font-size: 0.7rem !important; }
/* Tabs */
[data-testid="stTabs"] button { color: #7b859c !important; font-size: 0.84rem !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #f0a500 !important; border-bottom-color: #f0a500 !important; }
/* Expander */
[data-testid="stExpander"] { background: #191f30; border: 1px solid #252d3f; border-radius: 6px; }
/* Dataframe */
.stDataFrame { border-radius: 6px; overflow: hidden; }
/* Footer note */
.footer-note { font-size: 0.68rem; color: #3d4556; text-align: center; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────
def init_session_state():
    defaults = {
        "df_raw": None,
        "df_validated": None,
        "validation_report": None,
        "df_processed": None,
        "features_df": None,
        "risk_df": None,
        "anomaly_df": None,
        "selected_product": None,
        "aggregation": "weekly",
        "data_loaded": False,
        "processing_done": False,
        "analysis_done": False,
        "early_warnings_df": None,
        "early_warnings_status_df": None,
        "early_warnings_agg": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


def run_pipeline(product_id: str, agg: str):
    """Execute analysis pipeline for a specific product and aggregation interval."""
    if st.session_state.df_validated is None or len(st.session_state.df_validated) == 0:
        return
    product_df = filter_by_product(st.session_state.df_validated, product_id)
    if len(product_df) == 0:
        return
    processed = preprocess_reviews(product_df, use_transformer=True)
    processed = aggregate_by_period(processed, agg)
    features = calculate_period_features(processed)
    risk_df = calculate_risk_timeseries(features)
    anomaly_df = detect_anomalies(risk_df)
    st.session_state.selected_product = product_id
    st.session_state.aggregation = agg
    st.session_state.df_processed = processed
    st.session_state.features_df = features
    st.session_state.risk_df = risk_df
    st.session_state.anomaly_df = anomaly_df
    st.session_state.processing_done = True
    st.session_state.analysis_done = True


# ──────────────────────────────────────────────
# Reusable UI helpers
# ──────────────────────────────────────────────
def page_header(title: str, subtitle: str = ""):
    sub = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="page-header"><h1 class="page-title">{title}</h1>{sub}</div>',
        unsafe_allow_html=True,
    )


def section_heading(text: str):
    st.markdown(f'<p class="section-heading">{text}</p>', unsafe_allow_html=True)


def metric_card(col, title: str, value: str, subtitle: str = ""):
    sub = f'<div class="dash-card-sub">{subtitle}</div>' if subtitle else ""
    col.markdown(
        f'<div class="dash-card"><div class="dash-card-title">{title}</div>'
        f'<div class="dash-card-value">{value}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def notice(text: str, kind: str = "amber"):
    cls = {"amber": "notice-box", "blue": "notice-box-info", "warn": "notice-box-warn"}.get(kind, "notice-box")
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def empty_state(msg: str, hint: str = ""):
    hint_html = f'<div class="empty-state-hint">{hint}</div>' if hint else ""
    st.markdown(
        f'<div class="empty-state"><div class="empty-state-icon">📭</div>'
        f'<div class="empty-state-msg">{msg}</div>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def dataset_status_html() -> str:
    if st.session_state.data_loaded and st.session_state.df_validated is not None:
        return '<span class="status-pill status-loaded">● Data Loaded</span>'
    return '<span class="status-pill status-nodata">● No Data Loaded</span>'


def warning_badge_html(level: str) -> str:
    cls = {"High": "badge-high", "Moderate": "badge-mod", "Low": "badge-low"}.get(level, "badge-low")
    return f'<span class="{cls}">{level}</span>'


def chart_layout_dark(title: str = "", height: int = 380, **extra) -> dict:
    """Return consistent dark-theme Plotly layout overrides."""
    layout = dict(
        paper_bgcolor="#191f30",
        plot_bgcolor="#191f30",
        font=dict(color="#8892a4", size=11),
        height=height,
        margin=dict(l=52, r=24, t=44 if title else 20, b=40),
        xaxis=dict(gridcolor="#252d3f", linecolor="#252d3f", zeroline=False),
        yaxis=dict(gridcolor="#252d3f", linecolor="#252d3f", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#9aa3b5")),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#252d3f", font_color="#e0e0e0"),
    )
    if title:
        layout["title"] = dict(text=title, font=dict(size=12, color="#c0c8d8"), x=0)
    layout.update(extra)
    return layout


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
st.sidebar.markdown('<div class="sidebar-brand">Product Failure<br>Prediction</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-tagline">AI-Based Early Warning System</div>', unsafe_allow_html=True)
st.sidebar.markdown(dataset_status_html(), unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Data Explorer",
        "Early Warnings",
        "Complaint Analysis",
        "Temporal Trends",
        "Risk Monitoring",
        "Model Evaluation",
        "Methodology",
    ],
    label_visibility="collapsed",
)

# Sidebar Active Product Selector
if st.session_state.data_loaded and st.session_state.df_validated is not None:
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div style="font-size:0.74rem;font-weight:600;color:#7b859c;margin-bottom:4px;">ACTIVE PRODUCT</div>', unsafe_allow_html=True)
    avail_prods = get_products(st.session_state.df_validated)
    if avail_prods:
        curr_prod = st.session_state.selected_product if st.session_state.selected_product in avail_prods else avail_prods[0]
        sb_prod = st.sidebar.selectbox("Product", avail_prods, index=avail_prods.index(curr_prod), key="sb_product_choice")
        curr_agg = st.session_state.aggregation if st.session_state.aggregation in AGGREGATION_OPTIONS else AGGREGATION_OPTIONS[0]
        sb_agg = st.sidebar.selectbox("Interval", AGGREGATION_OPTIONS, index=AGGREGATION_OPTIONS.index(curr_agg), key="sb_agg_choice")
        if (sb_prod != st.session_state.selected_product or sb_agg != st.session_state.aggregation or not st.session_state.processing_done):
            if st.sidebar.button("Analyze Product", type="primary", use_container_width=True):
                with st.spinner(f"Analyzing {sb_prod}..."):
                    run_pipeline(sb_prod, sb_agg)
                    st.rerun()

st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="footer-note">v1.0 · Heuristic analysis only<br>Not validated failure probabilities</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════
if page == "Overview":
    page_header(
        "Early Product Failure Prediction",
        "AI-Based Early Warning System for Emerging Product Problems",
    )

    notice(
        '<strong>Research Question:</strong> "Can the temporal co-evolution of multiple complaint signals '
        'provide an early warning of a potential product failure?"',
        kind="blue",
    )

    # ── Summary metrics ────────────────────────────────────────
    section_heading("Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)

    if st.session_state.data_loaded and st.session_state.df_validated is not None:
        df = st.session_state.df_validated
        d_min, d_max = get_date_range(df)
        date_str = f"{d_min.strftime('%b %Y')} – {d_max.strftime('%b %Y')}" if d_min and d_max else "—"

        # Active warnings count
        ew_df = st.session_state.early_warnings_df
        if ew_df is not None and not ew_df.empty:
            real_w = ew_df[ew_df["warning_level"] != "Insufficient data"]
            active_warn = len(real_w)
            high_warn = int((real_w["warning_level"] == "High").sum())
            warn_sub = f"{high_warn} high-risk" if high_warn else "none high-risk"
        else:
            active_warn = "—"
            warn_sub = "run Early Warnings scan"

        metric_card(c1, "Valid Reviews", f"{len(df):,}", f"{df['product_id'].nunique()} product(s)")
        metric_card(c2, "Products", f"{df['product_id'].nunique()}", date_str)
        metric_card(c3, "Date Range", date_str, f"{len(df):,} reviews")

        if isinstance(active_warn, int):
            metric_card(c4, "Active Warnings", str(active_warn), warn_sub)
        else:
            metric_card(c4, "Active Warnings", "—", warn_sub)
    else:
        metric_card(c1, "Valid Reviews", "—", "no data loaded")
        metric_card(c2, "Products", "—", "—")
        metric_card(c3, "Date Range", "—", "—")
        metric_card(c4, "Active Warnings", "—", "—")

    # ── Warning summary ────────────────────────────────────────
    ew_df = st.session_state.early_warnings_df
    if ew_df is not None and not ew_df.empty:
        real_w = ew_df[ew_df["warning_level"] != "Insufficient data"]
        if len(real_w) > 0:
            section_heading("Warning Summary")
            wb1, wb2, wb3 = st.columns(3)
            metric_card(wb1, "High Risk", str(int((real_w["warning_level"] == "High").sum())), "immediate attention")
            metric_card(wb2, "Moderate Risk", str(int((real_w["warning_level"] == "Moderate").sum())), "monitor closely")
            metric_card(wb3, "Low Risk", str(int((real_w["warning_level"] == "Low").sum())), "routine monitoring")

            section_heading("Most Recent Warnings")
            recent = real_w.sort_values("warning_date", ascending=False).head(5)
            for _, wr in recent.iterrows():
                lvl = wr.get("warning_level", "Low")
                badge = warning_badge_html(lvl)
                prod = wr.get("product_id", "Unknown")
                period = wr.get("period_str", "N/A")
                score = wr.get("risk_score", 0)
                aspects = wr.get("affected_aspects", "")
                aspect_str = f" · {aspects}" if aspects else ""
                st.markdown(
                    f'{badge} &nbsp; <span style="color:#c0c8d8;font-size:0.85rem;"><strong>{prod}</strong> — {period} · Risk {score:.0f}/100{aspect_str}</span>',
                    unsafe_allow_html=True,
                )

    # ── Complaint trend (if analysis done) ────────────────────
    if st.session_state.analysis_done and st.session_state.features_df is not None:
        section_heading(f"Complaint Trend — {st.session_state.selected_product}")
        features = st.session_state.features_df
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=features["period"],
            y=(features["overall_complaint_proportion"] * 100).round(1),
            mode="lines+markers", name="Complaint %",
            line=dict(color="#f0a500", width=2), marker=dict(size=5),
        ))
        if "baseline_complaint_proportion" in features.columns:
            fig.add_trace(go.Scatter(
                x=features["period"],
                y=(features["baseline_complaint_proportion"] * 100).round(1),
                mode="lines", name="Baseline (EWMA)",
                line=dict(color="#4b5568", dash="dash"),
            ))
        fig.update_yaxes(title_text="Complaint Review %", range=[0, 105])
        fig.update_layout(**chart_layout_dark("", height=280))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Complaint % = reviews with ≥1 complaint / total valid reviews × 100. Bounded ≤ 100%.")
    elif not st.session_state.data_loaded:
        section_heading("Getting Started")
        empty_state(
            "No data loaded yet.",
            "Go to Data Explorer to upload a CSV or load the synthetic demo dataset.",
        )

    # ── How to interpret ───────────────────────────────────────
    with st.expander("How to interpret risk indicators", expanded=False):
        st.markdown("""
**Risk Score (0–100)** — A heuristic score combining complaint frequency, severity, growth rate,
symptom diversity, and review velocity. It is **not** a calibrated failure probability.

| Band | Score | Interpretation |
|------|-------|----------------|
| Low | 0–25 | No significant concern |
| Moderate | 25–50 | Some signals elevated — monitor |
| High | 50–75 | Multiple signals suggest potential issues |
| Critical | 75–100 | Strong multi-signal evidence of problems |

**Warnings** flag periods where complaint patterns differ significantly from a product's own historical baseline
(rolling z-score > 2.0 standard deviations). A minimum of 3 prior periods is required before any warning is issued.

⚠️ Warnings indicate *unusual complaint patterns*, not confirmed product failures.
""")

    # ── Capabilities ───────────────────────────────────────────
    with st.expander("System capabilities", expanded=False):
        st.markdown("""
- **Aspect-Based Complaint Extraction** — keyword lexicon identifies specific product issues (battery, heating, charging, etc.)
- **Sentiment & Severity Analysis** — measures complaint intensity using lexicon scoring blended with star ratings
- **Temporal Signal Engineering** — tracks how complaints evolve week-by-week or month-by-month
- **Multi-Signal Risk Scoring** — combines 5 signals into a transparent, auditable risk score
- **Anomaly Detection & Early Warning** — rolling baseline deviation with heuristic confidence labels
- **Baseline Comparison** — compares multi-signal approach against simple frequency-based detection
""")


# ═══════════════════════════════════════════════════════════════
# PAGE: Data Explorer
# ═══════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    page_header("Data Explorer", "Upload and validate your review dataset")

    tab_upload, tab_demo = st.tabs(["Upload CSV", "Load Demo Data"])

    # ── Upload Tab ─────────────────────────────────────────────
    with tab_upload:
        notice(
            "<strong>Expected columns:</strong> <code>product_id</code>, <code>review_date</code>, "
            "<code>review_text</code>, <code>rating</code> (optional: <code>review_id</code>). "
            "Alternative names are auto-detected — e.g. <code>asin</code>, <code>timestamp</code>, "
            "<code>body</code>, <code>overall</code>.",
            kind="blue",
        )

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

        if uploaded_file is not None:
            temp_df = pd.read_csv(uploaded_file, nrows=5)
            uploaded_file.seek(0)
            auto_mapping = auto_map_columns(temp_df)

            section_heading("Column Mapping")
            st.caption("Auto-detection results are pre-filled. Override if needed.")
            col1, col2 = st.columns(2)
            manual_mapping = {}
            cols_list = ["(auto)"] + list(temp_df.columns)

            with col1:
                for field in ["product_id", "review_id", "review_date"]:
                    detected = auto_mapping.get(field)
                    default_idx = cols_list.index(detected) if detected in cols_list else 0
                    sel = st.selectbox(f"`{field}`", cols_list, index=default_idx, key=f"map_{field}")
                    manual_mapping[field] = None if sel == "(auto)" else sel

            with col2:
                for field in ["review_text", "rating"]:
                    detected = auto_mapping.get(field)
                    default_idx = cols_list.index(detected) if detected in cols_list else 0
                    sel = st.selectbox(f"`{field}`", cols_list, index=default_idx, key=f"map_{field}")
                    manual_mapping[field] = None if sel == "(auto)" else sel

            final_mapping = {k: (v if v is not None else auto_mapping.get(k)) for k, v in manual_mapping.items()}

            if st.button("Load and Validate", type="primary"):
                with st.spinner("Loading and validating data..."):
                    df, report = validate_and_load_csv(uploaded_file, final_mapping)
                    st.session_state.df_validated = df
                    st.session_state.validation_report = report
                    st.session_state.data_loaded = len(df) > 0
                    st.session_state.processing_done = False
                    st.session_state.analysis_done = False
                    st.session_state.early_warnings_df = None

    # ── Demo Tab ───────────────────────────────────────────────
    with tab_demo:
        notice(
            "<strong>Synthetic Demo Data.</strong> Pre-generated for software testing only. "
            "Do <em>not</em> present synthetic results as real-world findings.",
            kind="warn",
        )
        demo_path = DATA_DIR / "synthetic_reviews.csv"
        if not demo_path.exists():
            st.warning("Synthetic demo data not found.")
            if st.button("Generate Synthetic Data"):
                with st.spinner("Generating synthetic dataset..."):
                    os.system(f'python "{PROJECT_ROOT / "scripts" / "generate_demo_data.py"}"')
                    st.rerun()
        else:
            if st.button("Load Synthetic Demo Dataset", type="primary"):
                with st.spinner("Loading synthetic demo data..."):
                    df, report = validate_and_load_csv(str(demo_path))
                    st.session_state.df_validated = df
                    st.session_state.validation_report = report
                    st.session_state.data_loaded = len(df) > 0
                    st.session_state.processing_done = False
                    st.session_state.analysis_done = False
                    st.session_state.early_warnings_df = None

    # ── Validation Results ─────────────────────────────────────
    if st.session_state.validation_report is not None:
        report = st.session_state.validation_report
        section_heading("Validation Summary")
        v1, v2, v3, v4 = st.columns(4)
        metric_card(v1, "Rows Read", str(report["total_rows_read"]))
        metric_card(v2, "Rows Retained", str(report["rows_retained"]), "usable records")
        metric_card(v3, "Rows Rejected", str(report["rows_rejected"]), "invalid/missing fields")
        metric_card(v4, "Duplicates Removed", str(report["duplicates_removed"]), "by ID + content")

        if report["issues"]:
            with st.expander(f"Validation Details ({len(report['issues'])} issue(s))", expanded=False):
                for issue in report["issues"]:
                    st.warning(issue)
        else:
            st.success("All records passed validation.")

    # ── Data Preview ───────────────────────────────────────────
    if st.session_state.data_loaded and st.session_state.df_validated is not None:
        df = st.session_state.df_validated
        section_heading("Data Preview")
        st.caption(f"Showing first 10 of {len(df):,} valid records.")
        st.dataframe(df.head(10), use_container_width=True)

        # Missing data
        missing = df.isnull().sum()
        if missing.sum() > 0:
            with st.expander("Missing Value Report", expanded=False):
                missing_pct = (missing / len(df) * 100).round(2)
                missing_report = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
                st.dataframe(missing_report[missing_report["Missing Count"] > 0], use_container_width=True)
        else:
            st.caption("No missing values in the retained dataset.")

        # Analysis configuration
        section_heading("Analysis Configuration")
        st.caption("Select a product and interval, then run the pipeline to enable all analysis pages.")
        ac1, ac2 = st.columns(2)
        products = get_products(df)
        with ac1:
            selected_product = st.selectbox("Product", products, key="product_select")
            st.session_state.selected_product = selected_product
        with ac2:
            agg = st.selectbox("Aggregation Interval", AGGREGATION_OPTIONS, key="agg_select")
            st.session_state.aggregation = agg

        product_df = filter_by_product(df, selected_product)
        st.caption(f"Product **{selected_product}** — {len(product_df)} reviews")

        if st.button("Run Analysis Pipeline", type="primary"):
            with st.spinner(f"Running pipeline for {selected_product} ({agg})..."):
                run_pipeline(selected_product, agg)
            st.success(f"Pipeline complete for **{selected_product}**. Navigate to other pages to explore results.")


# ═══════════════════════════════════════════════════════════════
# PAGE: Early Warnings
# ═══════════════════════════════════════════════════════════════
elif page == "Early Warnings":
    page_header("Early Warning Notification System", "Unusual complaint patterns across all products")

    notice(
        "<strong>Disclaimer:</strong> Warnings indicate unusual complaint patterns in the uploaded dataset. "
        "They do <strong>not</strong> confirm physical product failure, establish liability, or constitute "
        "validated predictions. Risk scores are heuristic indicators, not calibrated failure probabilities.",
        kind="warn",
    )

    if not st.session_state.data_loaded or st.session_state.df_validated is None:
        empty_state(
            "No data loaded.",
            "Go to Data Explorer to upload a CSV dataset first.",
        )
        st.stop()

    df_validated = st.session_state.df_validated
    all_products = get_products(df_validated)
    n_products = len(all_products)

    # Controls
    ec1, ec2 = st.columns([2, 1])
    with ec1:
        ew_agg = st.selectbox(
            "Aggregation Interval",
            AGGREGATION_OPTIONS, index=0, key="ew_aggregation",
            help="Group reviews by weekly or monthly periods before analysing trends.",
        )
    with ec2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("Generate Early Warnings", type="primary", use_container_width=True)

    need_generate = generate_btn or (
        st.session_state.early_warnings_df is not None
        and st.session_state.early_warnings_agg == ew_agg
    )

    if need_generate:
        if generate_btn or st.session_state.early_warnings_df is None:
            with st.spinner(f"Scanning {n_products} product(s) for unusual complaint patterns..."):
                ew_df, ew_status = generate_early_warnings(df_validated, ew_agg)
                st.session_state.early_warnings_df = ew_df
                st.session_state.early_warnings_status_df = ew_status
                st.session_state.early_warnings_agg = ew_agg

        ew_df = st.session_state.early_warnings_df
        ew_status = st.session_state.early_warnings_status_df

        # Summary metrics
        section_heading("Warning Summary")
        real_warnings = ew_df[ew_df["warning_level"] != "Insufficient data"] if not ew_df.empty else ew_df
        total_warnings = len(real_warnings)
        high_risk_count = int((real_warnings["warning_level"] == "High").sum()) if not real_warnings.empty else 0
        products_with_warnings = int(real_warnings["product_id"].nunique()) if not real_warnings.empty else 0

        wm1, wm2, wm3, wm4 = st.columns(4)
        metric_card(wm1, "Total Warnings", str(total_warnings), "anomalous periods detected")
        metric_card(wm2, "High-Risk Warnings", str(high_risk_count), "immediate attention")
        metric_card(wm3, "Products Affected", str(products_with_warnings), f"of {n_products} scanned")
        metric_card(wm4, "Products Scanned", str(n_products), f"{ew_agg} aggregation")

        with st.expander("Per-Product Scan Status", expanded=False):
            if ew_status is not None and not ew_status.empty:
                st.dataframe(ew_status, use_container_width=True)
            else:
                st.caption("No status information available.")

        # Filters & table
        section_heading("Warning History")

        if total_warnings == 0:
            st.success(
                "No unusual complaint patterns detected. "
                "Try switching to weekly aggregation or upload a dataset with more reviews per product."
            )
        else:
            f1, f2, f3 = st.columns(3)
            with f1:
                product_filter = st.multiselect(
                    "Product", options=sorted(real_warnings["product_id"].unique().tolist()),
                    default=[], key="ew_product_filter", placeholder="All products",
                )
            with f2:
                level_filter = st.multiselect(
                    "Warning Level", options=["High", "Moderate", "Low"],
                    default=[], key="ew_level_filter", placeholder="All levels",
                )
            with f3:
                valid_dates = real_warnings["warning_date"].dropna()
                date_range = None
                if len(valid_dates) > 0:
                    min_date = pd.Timestamp(valid_dates.min()).date()
                    max_date = pd.Timestamp(valid_dates.max()).date()
                    date_range = st.date_input(
                        "Date Range", value=(min_date, max_date),
                        min_value=min_date, max_value=max_date, key="ew_date_filter",
                    )

            filtered = real_warnings.copy()
            if product_filter:
                filtered = filtered[filtered["product_id"].isin(product_filter)]
            if level_filter:
                filtered = filtered[filtered["warning_level"].isin(level_filter)]
            if date_range and len(date_range) == 2:
                try:
                    filtered = filtered[
                        (filtered["warning_date"] >= pd.Timestamp(date_range[0]))
                        & (filtered["warning_date"] <= pd.Timestamp(date_range[1]))
                    ]
                except Exception:
                    pass

            st.caption(f"{len(filtered)} warning(s) shown")

            display_cols = [
                "product_id", "period_str", "warning_level",
                "risk_score", "prev_risk_score", "affected_aspects",
                "confidence", "total_reviews", "complaint_reviews_count",
                "overall_complaint_proportion",
            ]
            display_cols = [c for c in display_cols if c in filtered.columns]

            def _style_row(row):
                colour = {"High": "#3d1515", "Moderate": "#3d2e0a", "Low": "#0f2e1a"}.get(
                    row.get("warning_level", ""), ""
                )
                return [f"background-color: {colour}" if colour else "" for _ in row]

            if not filtered.empty:
                styled = filtered[display_cols].style.apply(_style_row, axis=1)
                st.dataframe(styled, use_container_width=True, hide_index=True)

                csv_bytes = filtered.drop(columns=["warning_date"], errors="ignore").to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Warning History (CSV)",
                    data=csv_bytes, file_name="early_warnings.csv", mime="text/csv",
                )
            else:
                st.info("No warnings match the selected filters.")

            # Complaint trend chart
            section_heading("Complaint Trend Chart")

            chart_products = sorted(real_warnings["product_id"].unique().tolist())
            selected_chart_prod = st.selectbox("Product", options=chart_products, key="ew_chart_product")

            if selected_chart_prod:
                with st.spinner(f"Building trend chart for {selected_chart_prod}..."):
                    try:
                        prod_df = filter_by_product(df_validated, selected_chart_prod)
                        prod_processed = preprocess_reviews(prod_df, use_transformer=False)
                        prod_processed = aggregate_by_period(prod_processed, ew_agg)
                        prod_features = calculate_period_features(prod_processed)
                        prod_risk = calculate_risk_timeseries(prod_features)
                        prod_anomaly = detect_anomalies(prod_risk)

                        if not prod_anomaly.empty:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=prod_anomaly["period"],
                                y=(prod_anomaly["overall_complaint_proportion"] * 100).round(1),
                                mode="lines+markers", name="Complaint Proportion (%)",
                                line=dict(color="#f0a500", width=2), marker=dict(size=5),
                            ))
                            fig.add_trace(go.Scatter(
                                x=prod_anomaly["period"],
                                y=prod_anomaly["risk_score"].round(1),
                                mode="lines+markers", name="Risk Score (0–100)",
                                line=dict(color="#60a5fa", width=2, dash="dot"),
                                marker=dict(size=5), yaxis="y2",
                            ))
                            anomaly_rows = prod_anomaly[prod_anomaly["is_anomaly"] == True]
                            if not anomaly_rows.empty:
                                fig.add_trace(go.Scatter(
                                    x=anomaly_rows["period"],
                                    y=(anomaly_rows["overall_complaint_proportion"] * 100).round(1),
                                    mode="markers", name="Anomalous Period",
                                    marker=dict(color="#f87171", size=12, symbol="star"),
                                ))
                            layout = chart_layout_dark(f"Complaint Trend: {selected_chart_prod}", height=400)
                            layout["yaxis"]["title"] = "Complaint %"
                            layout["yaxis"]["range"] = [0, 105]
                            layout["yaxis2"] = dict(
                                title="Risk Score", overlaying="y", side="right",
                                range=[0, 105], gridcolor="#252d3f", linecolor="#252d3f",
                                zeroline=False,
                            )
                            layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#9aa3b5"))
                            fig.update_layout(**layout)
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(
                                "Complaint % = reviews with ≥1 complaint / total valid reviews × 100 (bounded ≤ 100%). "
                                "Stars (★) mark anomalous periods."
                            )
                    except Exception as exc:
                        st.error(f"Could not build trend chart: {exc}")

            # Warning detail expanders
            section_heading("Warning Explanations")
            if not filtered.empty:
                for _, warn_row in filtered.iterrows():
                    lvl = warn_row.get("warning_level", "Low")
                    badge = warning_badge_html(lvl)
                    prod_id = warn_row.get("product_id", "Unknown")
                    period_s = warn_row.get("period_str", "N/A")
                    score = warn_row.get("risk_score", 0)
                    with st.expander(f"{prod_id} — {period_s} · Risk {score:.0f}/100", expanded=False):
                        st.markdown(f'{badge}', unsafe_allow_html=True)
                        d1, d2 = st.columns(2)
                        prev_val = warn_row.get("prev_risk_score", float("nan"))
                        prev_nan = isinstance(prev_val, float) and np.isnan(prev_val)
                        with d1:
                            st.metric("Risk Score", f"{score:.1f}/100")
                            st.metric("Previous Risk Score", f"{prev_val:.1f}" if not prev_nan else "N/A")
                            st.metric("Confidence", str(warn_row.get("confidence", "N/A")))
                        with d2:
                            prop_val = warn_row.get("overall_complaint_proportion", 0)
                            prop_nan = isinstance(prop_val, float) and np.isnan(prop_val)
                            st.metric("Complaint Proportion", f"{prop_val*100:.1f}%" if not prop_nan else "N/A",
                                      help="Reviews with ≥1 complaint / total valid reviews × 100. Max 100%.")
                            st.metric("Total Reviews", int(warn_row.get("total_reviews", 0)))
                            st.metric("Complaint Reviews", int(warn_row.get("complaint_reviews_count", 0)))
                        aspects = warn_row.get("affected_aspects", "")
                        if aspects:
                            st.markdown(f"**Affected Aspects:** {aspects}")
                        st.markdown(f"**Explanation:** {warn_row.get('reason_text', 'N/A')}")
                        raw = warn_row.get("anomaly_signals_json", "[]")
                        try:
                            signals_parsed = _json_mod.loads(raw) if isinstance(raw, str) else raw
                        except Exception:
                            signals_parsed = []
                        if signals_parsed:
                            with st.expander("Raw Anomaly Signals", expanded=False):
                                st.json(signals_parsed)
            else:
                st.caption("Adjust filters above to see warning explanations.")

        # Insufficient data notices
        insuf_rows = ew_df[ew_df["warning_level"] == "Insufficient data"] if not ew_df.empty else pd.DataFrame()
        if not insuf_rows.empty:
            section_heading("Products With Insufficient History")
            for _, r in insuf_rows.iterrows():
                notice(f"<strong>{r['product_id']}:</strong> {r['reason_text']}", kind="blue")

        # Limitations
        with st.expander("System Limitations & Interpretation Guide", expanded=False):
            st.markdown("""
**What this system does:**
- Identifies periods where complaint patterns differ significantly from a product's own historical baseline.
- Uses rolling baseline deviation (z-score, threshold 2.0 SD) with a minimum of 3 prior periods.
- No future data leakage: each warning uses only reviews up to and including that period.

**What this system does NOT do:**
- Does **not** prove physical product failure.
- Does **not** provide calibrated failure probabilities.
- Does **not** validate predictions against real-world failure labels.

**Complaint Proportion denominator:** Unique valid reviews. A review mentioning Battery, Heating,
and Charging counts as 1 complaint review (not 3). Proportion is always ≤ 100%.
""")
    else:
        empty_state(
            "Click 'Generate Early Warnings' to scan all products for unusual complaint patterns.",
            f"{n_products} product(s) available to scan.",
        )


# ═══════════════════════════════════════════════════════════════
# PAGE: Complaint Analysis
# ═══════════════════════════════════════════════════════════════
elif page == "Complaint Analysis":
    page_header(
        "Complaint Analysis",
        f"Aspect distribution and severity — {st.session_state.selected_product or 'no product selected'}",
    )

    if not st.session_state.processing_done or st.session_state.df_processed is None:
        empty_state(
            "No analysis results yet.",
            "Go to Data Explorer, select a product, and run the analysis pipeline.",
        )
    else:
        df = st.session_state.df_processed

        notice(
            "<strong>Complaint Proportion</strong> (≤ 100%) counts each review at most once, "
            "regardless of how many aspects it mentions. "
            "<strong>Aspect Mention Density</strong> counts total mentions and may exceed 100%.",
            kind="blue",
        )

        # Aspect distribution (horizontal bar)
        section_heading("Most Frequent Complaint Aspects")
        all_aspects = []
        for aspects in df["aspects"]:
            if isinstance(aspects, list):
                all_aspects.extend(aspects)

        if all_aspects:
            aspect_counts = pd.Series(all_aspects).value_counts().sort_values(ascending=True)
            fig = px.bar(
                x=aspect_counts.values, y=aspect_counts.index,
                orientation="h",
                labels={"x": "Mention Count", "y": "Aspect"},
                color=aspect_counts.values,
                color_continuous_scale=[[0, "#2d3748"], [1, "#f0a500"]],
            )
            fig.update_layout(**chart_layout_dark("Complaint Aspect Distribution (total mentions)", height=360))
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Note: 'Mention Count' counts total aspect mentions across all reviews. "
                "A single review can mention multiple aspects. This metric may exceed total review count."
            )
        else:
            st.info("No specific complaint aspects detected in the reviews.")

        # Sentiment + Severity
        section_heading("Sentiment & Severity Distribution")
        col1, col2 = st.columns(2)
        with col1:
            sent_counts = df["sentiment_label"].value_counts()
            fig = px.pie(
                values=sent_counts.values, names=sent_counts.index,
                color_discrete_map={"negative": "#f87171", "positive": "#4ade80", "neutral": "#fbbf24"},
            )
            fig.update_layout(**chart_layout_dark("Sentiment Label Distribution", height=320))
            fig.update_traces(textfont_color="#e0e0e0")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(
                df, x="combined_severity", nbins=20,
                labels={"combined_severity": "Combined Severity (0=positive, 1=negative)"},
                color_discrete_sequence=["#f0a500"],
            )
            fig.update_layout(**chart_layout_dark("Severity Score Distribution", height=320))
            st.plotly_chart(fig, use_container_width=True)

        # Severity by aspect
        section_heading("Average Severity by Aspect")
        aspect_severity_data = []
        for _, row in df.iterrows():
            if isinstance(row["aspects"], list):
                for asp in row["aspects"]:
                    aspect_severity_data.append({"Aspect": asp, "Severity": row["combined_severity"]})
        if aspect_severity_data:
            asp_sev_df = pd.DataFrame(aspect_severity_data)
            avg_sev = asp_sev_df.groupby("Aspect")["Severity"].mean().sort_values(ascending=True)
            fig = px.bar(
                x=avg_sev.values, y=avg_sev.index, orientation="h",
                labels={"x": "Average Severity (0–1)", "y": "Aspect"},
                color=avg_sev.values,
                color_continuous_scale=[[0, "#2d3748"], [1, "#f87171"]],
            )
            fig.update_layout(**chart_layout_dark("Average Complaint Severity by Aspect", height=340))
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Example reviews
        section_heading("Review Explorer")
        aspect_filter = st.selectbox("Filter by aspect", ["All"] + get_supported_aspects())
        display_df = df.copy()
        if aspect_filter != "All":
            display_df = display_df[
                display_df["aspects"].apply(lambda x: aspect_filter in x if isinstance(x, list) else False)
            ]
        if len(display_df) > 0:
            show_cols = ["review_text", "aspects", "sentiment_label", "combined_severity", "rating"]
            show_cols = [c for c in show_cols if c in display_df.columns]
            st.dataframe(display_df[show_cols].head(20), use_container_width=True)
        else:
            st.info(f"No reviews found for aspect: {aspect_filter}")


# ═══════════════════════════════════════════════════════════════
# PAGE: Temporal Trends
# ═══════════════════════════════════════════════════════════════
elif page == "Temporal Trends":
    agg_label = (st.session_state.aggregation or "weekly").capitalize()
    prod_label = st.session_state.selected_product or "—"
    page_header("Temporal Trends", f"{agg_label} analysis · {prod_label}")

    if not st.session_state.analysis_done or st.session_state.features_df is None:
        empty_state(
            "No temporal data available.",
            "Go to Data Explorer and run the analysis pipeline first.",
        )
    else:
        features = st.session_state.features_df

        notice(
            "<strong>Complaint-Review Proportion</strong> = reviews with ≥1 complaint / total valid reviews. "
            "Bounded strictly ≤ 100%. Denominator: total valid reviews per period.",
            kind="blue",
        )

        # Main complaint trend (2-row subplot)
        section_heading(f"{agg_label} Complaint Trends")
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("Complaint-Review Proportion (≤ 100%)", "Valid Reviews per Period"),
            vertical_spacing=0.12,
        )
        fig.add_trace(go.Scatter(
            x=features["period"], y=features["overall_complaint_proportion"],
            mode="lines+markers", name="Complaint Proportion",
            line=dict(color="#f0a500", width=2), marker=dict(size=5),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=features["period"], y=features["baseline_complaint_proportion"],
            mode="lines", name="Baseline (EWMA)",
            line=dict(color="#4b5568", dash="dash"),
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=features["period"], y=features["total_reviews"],
            name="Total Valid Reviews", marker_color="#3b82f6", opacity=0.7,
        ), row=2, col=1)
        fig.update_yaxes(title_text="Complaint %", tickformat=".0%", range=[0, 1.05], row=1, col=1,
                         gridcolor="#252d3f", linecolor="#252d3f")
        fig.update_yaxes(title_text="Review Count", row=2, col=1,
                         gridcolor="#252d3f", linecolor="#252d3f")
        fig.update_xaxes(gridcolor="#252d3f", linecolor="#252d3f")
        base_layout = chart_layout_dark("", height=480)
        fig.update_layout(
            paper_bgcolor=base_layout["paper_bgcolor"],
            plot_bgcolor=base_layout["plot_bgcolor"],
            font=base_layout["font"],
            height=480,
            margin=base_layout["margin"],
            legend=base_layout["legend"],
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Aspect Mention Density
        if "aspect_mentions_per_review" in features.columns:
            section_heading("Aspect Mention Density — Multi-Complaint Co-occurrence")
            st.caption("Average Aspect Mentions per Review: may exceed 1.0 when customers report multiple co-occurring failures.")
            fig_density = make_subplots(specs=[[{"secondary_y": True}]])
            fig_density.add_trace(go.Bar(
                x=features["period"], y=features["total_aspect_mentions"],
                name="Total Aspect Mentions", marker_color="#6366f1", opacity=0.6,
            ), secondary_y=False)
            fig_density.add_trace(go.Scatter(
                x=features["period"], y=features["aspect_mentions_per_review"],
                mode="lines+markers", name="Avg Mentions / Review",
                line=dict(color="#a78bfa", width=2),
            ), secondary_y=True)
            fig_density.update_yaxes(title_text="Total Mentions", secondary_y=False,
                                     gridcolor="#252d3f", linecolor="#252d3f")
            fig_density.update_yaxes(title_text="Mentions / Review", secondary_y=True)
            base_layout2 = chart_layout_dark("", height=340)
            fig_density.update_layout(
                paper_bgcolor=base_layout2["paper_bgcolor"],
                plot_bgcolor=base_layout2["plot_bgcolor"],
                font=base_layout2["font"], height=340, margin=base_layout2["margin"],
                legend=base_layout2["legend"], hovermode="x unified",
            )
            st.plotly_chart(fig_density, use_container_width=True)

        # Per-aspect trends
        section_heading("Per-Aspect Complaint Proportions")
        st.caption("Aspect-Specific Proportion: % of valid reviews mentioning a particular aspect. Bounded ≤ 100%.")
        aspects = get_supported_aspects()
        active_aspects = [a for a in aspects if f"complaint_count_{a}" in features.columns and features[f"complaint_count_{a}"].sum() > 0]
        default_aspects = active_aspects[:5] if active_aspects else [a for a in ["Battery", "Heating", "Charging", "Display"] if a in aspects]
        selected_aspects = st.multiselect("Aspects to display", aspects, default=default_aspects)
        if selected_aspects:
            fig = go.Figure()
            colors = ["#f0a500", "#f87171", "#60a5fa", "#4ade80", "#a78bfa", "#fb923c", "#34d399", "#818cf8", "#fbbf24"]
            for i, asp in enumerate(selected_aspects):
                col = f"complaint_proportion_{asp}"
                if col in features.columns:
                    fig.add_trace(go.Scatter(
                        x=features["period"], y=features[col],
                        mode="lines+markers", name=asp,
                        line=dict(color=colors[i % len(colors)]),
                        marker=dict(size=4),
                    ))
            fig.update_yaxes(title_text="Aspect Proportion", tickformat=".0%", range=[0, 1.05])
            fig.update_layout(**chart_layout_dark("Aspect Complaint Proportions Over Time", height=380))
            st.plotly_chart(fig, use_container_width=True)

        # Growth rate
        section_heading("Complaint Growth Rate")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=features["period"], y=features["complaint_growth"],
            marker_color=features["complaint_growth"].apply(lambda x: "#f87171" if x > 0 else "#4ade80"),
            name="Growth Rate",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="#4b5568")
        fig.update_yaxes(title_text="Growth Rate")
        fig.update_layout(**chart_layout_dark("Overall Complaint Growth Rate", height=320))
        st.plotly_chart(fig, use_container_width=True)

        # Symptom diversity
        section_heading("Symptom Diversity Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=features["period"], y=features["symptom_diversity"],
            mode="lines+markers", fill="tozeroy",
            line=dict(color="#a78bfa"),
            fillcolor="rgba(167,139,250,0.12)",
            name="Symptom Diversity",
        ))
        fig.update_yaxes(title_text="Diversity (0–1)", range=[0, 1.05])
        fig.update_layout(**chart_layout_dark("Symptom Diversity (0=no aspects, 1=all aspects)", height=300))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: Risk Monitoring
# ═══════════════════════════════════════════════════════════════
elif page == "Risk Monitoring":
    page_header(
        "Risk Monitoring",
        f"Heuristic risk assessment — {st.session_state.selected_product or '—'}",
    )

    if not st.session_state.analysis_done or st.session_state.risk_df is None:
        empty_state(
            "No risk data available.",
            "Go to Data Explorer and run the analysis pipeline first.",
        )
    else:
        risk_df = st.session_state.risk_df
        anomaly_df = st.session_state.anomaly_df
        features_df = st.session_state.features_df

        notice(
            "Risk scores are <strong>heuristic indicators</strong>, not calibrated failure probabilities. "
            "They combine complaint frequency, severity, growth, symptom diversity, and review velocity "
            "into a single 0–100 score for monitoring purposes only.",
            kind="amber",
        )

        latest = risk_df.iloc[-1] if len(risk_df) > 0 else None

        if latest is not None:
            section_heading("Current Risk Assessment (Latest Period)")
            risk_score = latest["risk_score"]
            risk_level = latest["risk_level"]
            level_colors_text = {"Low": "#4ade80", "Moderate": "#fbbf24", "High": "#fb923c", "Critical": "#f87171"}

            rm1, rm2, rm3 = st.columns([1, 1, 2])
            with rm1:
                st.metric("Risk Score", f"{risk_score:.1f} / 100")
                st.metric("Risk Level", risk_level)
            with rm2:
                # Key feature metrics
                prop_val = float(latest.get("overall_complaint_proportion", 0.0))
                crevs = int(latest.get("complaint_reviews_count", 0))
                trevs = int(latest.get("total_reviews", 0))
                st.metric("Complaint Reviews", f"{crevs}/{trevs}", help="Reviews with ≥1 complaint / total valid reviews")
                st.metric("Complaint %", f"{prop_val*100:.1f}%")
            with rm3:
                contrib_cols = [c for c in risk_df.columns if c.startswith("contrib_")]
                if contrib_cols:
                    contrib_data = {c.replace("contrib_", "").replace("_", " ").title(): latest[c] for c in contrib_cols}
                    fig = px.bar(
                        x=list(contrib_data.values()), y=list(contrib_data.keys()),
                        orientation="h",
                        labels={"x": "Contribution (points)", "y": "Signal"},
                        color=list(contrib_data.values()),
                        color_continuous_scale=[[0, "#2d3748"], [0.5, "#f0a500"], [1, "#f87171"]],
                    )
                    fig.update_layout(**chart_layout_dark("Signal Contributions to Risk Score", height=230))
                    fig.update_coloraxes(showscale=False)
                    st.plotly_chart(fig, use_container_width=True)

            # Risk explanation
            top_aspects = get_top_anomaly_aspects(features_df, len(features_df) - 1)
            try:
                explanation = generate_risk_explanation({}, latest.to_dict(), top_aspects)
            except Exception:
                explanation = f"Risk score: {risk_score:.1f}/100. Level: {risk_level}."
            notice(f"<strong>Assessment:</strong> {explanation}")

            # Full breakdown table
            section_heading("Multi-Signal Risk Score Breakdown")
            st.latex(r"\text{Risk Score} = 100 \times \left( 0.25\hat{p} + 0.20\hat{s} + 0.25\hat{g} + 0.15\hat{d} + 0.15\hat{v} \right)")

            sev_val = float(latest.get("overall_avg_severity", 0.0))
            growth_val = float(latest.get("complaint_growth", 0.0))
            div_val = float(latest.get("symptom_diversity", 0.0))
            aspects_n = int(latest.get("num_distinct_aspects", 0))
            vel_val = float(latest.get("review_velocity", 0.0))
            base_vel = float(latest.get("baseline_velocity", 0.0))
            vel_ratio = vel_val / base_vel if base_vel > 0 else 1.0
            weights = DEFAULT_RISK_WEIGHTS

            breakdown_rows = [
                {"Signal": "Complaint Frequency", "Weight": f"{weights['frequency']*100:.0f}%",
                 "Raw Value": f"{prop_val:.1%} ({crevs}/{trevs})",
                 "Normalized [0,1]": f"{prop_val:.3f}",
                 "Points": f"{latest.get('contrib_frequency', 0.0):.2f}",
                 "Description": "Proportion of valid reviews with ≥1 complaint"},
                {"Signal": "Complaint Severity", "Weight": f"{weights['severity']*100:.0f}%",
                 "Raw Value": f"{sev_val:.2f} / 1.0",
                 "Normalized [0,1]": f"{sev_val:.3f}",
                 "Points": f"{latest.get('contrib_severity', 0.0):.2f}",
                 "Description": "Average severity of complaint reviews"},
                {"Signal": "Complaint Growth", "Weight": f"{weights['growth']*100:.0f}%",
                 "Raw Value": f"{growth_val:.1%}",
                 "Normalized [0,1]": f"{np.tanh(max(0, growth_val)):.3f}",
                 "Points": f"{latest.get('contrib_growth', 0.0):.2f}",
                 "Description": "Period-over-period change in complaint proportion"},
                {"Signal": "Symptom Diversity", "Weight": f"{weights['symptom_diversity']*100:.0f}%",
                 "Raw Value": f"{aspects_n}/9 aspects ({div_val:.1%})",
                 "Normalized [0,1]": f"{div_val:.3f}",
                 "Points": f"{latest.get('contrib_symptom_diversity', 0.0):.2f}",
                 "Description": "Breadth of distinct complaint categories"},
                {"Signal": "Review Velocity", "Weight": f"{weights['review_velocity']*100:.0f}%",
                 "Raw Value": f"{vel_val:.0f} (baseline {base_vel:.1f})",
                 "Normalized [0,1]": f"{np.tanh(max(0, vel_ratio - 1.0)):.3f}",
                 "Points": f"{latest.get('contrib_review_velocity', 0.0):.2f}",
                 "Description": "Review volume vs historical baseline"},
            ]
            st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True)
            st.caption(f"Total Risk Score: {risk_score:.2f}/100 — Bands: Low <25, Moderate <50, High <75, Critical ≥75. Not a failure probability.")

        # Historical risk chart
        section_heading("Risk Score History")
        fig = go.Figure()
        band_colors = {
            "Low": "rgba(74,222,128,0.07)", "Moderate": "rgba(251,191,36,0.08)",
            "High": "rgba(251,146,60,0.10)", "Critical": "rgba(248,113,113,0.12)",
        }
        for level, (low, high) in RISK_BANDS.items():
            fig.add_hrect(
                y0=low, y1=high, fillcolor=band_colors.get(level, "rgba(0,0,0,0)"),
                layer="below", line_width=0,
                annotation_text=level,
                annotation=dict(font_size=9, font_color="#4b5568"),
            )
        fig.add_trace(go.Scatter(
            x=risk_df["period"], y=risk_df["risk_score"],
            mode="lines+markers", name="Risk Score",
            line=dict(color="#f0a500", width=2), marker=dict(size=5),
        ))
        if anomaly_df is not None and "is_anomaly" in anomaly_df.columns:
            anomaly_points = anomaly_df[anomaly_df["is_anomaly"] == True]
            if len(anomaly_points) > 0:
                fig.add_trace(go.Scatter(
                    x=anomaly_points["period"],
                    y=anomaly_points["risk_score"] if "risk_score" in anomaly_points.columns else [0]*len(anomaly_points),
                    mode="markers", name="Anomaly Flagged",
                    marker=dict(color="#f87171", size=11, symbol="triangle-up",
                                line=dict(width=1.5, color="#7f1d1d")),
                ))
        fig.update_yaxes(title_text="Risk Score (0–100)", range=[0, 105])
        fig.update_layout(**chart_layout_dark("Product Risk Score Over Time", height=420))
        st.plotly_chart(fig, use_container_width=True)

        # Anomaly alerts
        section_heading("Anomaly Alerts")
        if anomaly_df is not None and "is_anomaly" in anomaly_df.columns:
            anomalies = anomaly_df[anomaly_df["is_anomaly"] == True]
            if len(anomalies) > 0:
                for idx, (row_idx, anom_row) in enumerate(anomalies.iterrows()):
                    period_idx = anomaly_df.index.get_loc(row_idx)
                    top_asp = get_top_anomaly_aspects(features_df, period_idx)
                    explanation = generate_anomaly_explanation(anom_row, top_asp)
                    period_str = anom_row["period"].strftime("%Y-%m-%d") if hasattr(anom_row["period"], "strftime") else str(anom_row["period"])
                    with st.expander(
                        f"Alert — {period_str} · Confidence: {anom_row.get('confidence', 'heuristic')}",
                        expanded=(idx < 3),
                    ):
                        st.markdown(explanation)
                        if top_asp:
                            st.dataframe(pd.DataFrame(top_asp), use_container_width=True)
            else:
                st.success("No anomalies detected in the analyzed periods.")
        else:
            st.info("Anomaly detection results not available.")

        # Supporting evidence
        section_heading("Supporting Review Evidence (Highest Severity Complaints)")
        if st.session_state.df_processed is not None:
            processed = st.session_state.df_processed
            complaints = processed[processed["is_complaint"] == True].sort_values("combined_severity", ascending=False)
            if len(complaints) > 0:
                show_cols = ["review_text", "aspects", "sentiment_label", "combined_severity", "rating", "period"]
                show_cols = [c for c in show_cols if c in complaints.columns]
                st.dataframe(complaints[show_cols].head(15), use_container_width=True)
            else:
                st.info("No complaint reviews to display.")


# ═══════════════════════════════════════════════════════════════
# PAGE: Model Evaluation
# ═══════════════════════════════════════════════════════════════
elif page == "Model Evaluation":
    page_header("Model Evaluation", "Approach comparison and signal contribution analysis")

    if not st.session_state.analysis_done or st.session_state.risk_df is None:
        empty_state(
            "No analysis results available.",
            "Go to Data Explorer and run the analysis pipeline first.",
        )
    else:
        features_df = st.session_state.features_df
        risk_df = st.session_state.risk_df

        notice(
            "<strong>Important:</strong> Without independently verified product-failure labels, "
            "standard failure-prediction precision, recall, and F1 <strong>cannot be validly calculated</strong>. "
            "What is shown here is approach-comparison and ablation analysis — not validated predictive performance.",
            kind="warn",
        )

        section_heading("Approach Comparison — Baseline vs Multi-Signal")
        col1, col2 = st.columns(2)
        with col1:
            baseline_thresh = st.slider(
                "Baseline complaint proportion threshold",
                0.0, 1.0, BASELINE_COMPLAINT_THRESHOLD, 0.05,
            )
        with col2:
            risk_thresh = st.slider("Multi-signal risk score threshold", 0.0, 100.0, 50.0, 5.0)

        comparison = compare_approaches(features_df, risk_df,
                                        baseline_threshold=baseline_thresh,
                                        risk_threshold=risk_thresh)

        cm1, cm2, cm3 = st.columns(3)
        metric_card(cm1, "Baseline Flags", str(comparison["baseline_flags_count"]), "frequency > threshold")
        metric_card(cm2, "Multi-Signal Flags", str(comparison["multisignal_flags_count"]), "risk score > threshold")
        metric_card(cm3, "Total Periods", str(comparison["total_periods"]), "analyzed")

        if "agreement" in comparison:
            ag = comparison["agreement"]
            st.caption(f"Agreement Rate: {ag['agreement_rate']:.1%}")
            fig = go.Figure(data=[go.Bar(
                x=["Both Flagged", "Only Baseline", "Only Multi-Signal", "Neither"],
                y=[ag["both_flagged"], ag["only_baseline"], ag["only_multisignal"], ag["neither_flagged"]],
                marker_color=["#f87171", "#fbbf24", "#60a5fa", "#4ade80"],
            )])
            fig.update_yaxes(title_text="Period Count")
            fig.update_layout(**chart_layout_dark("Approach Agreement Analysis", height=320))
            st.plotly_chart(fig, use_container_width=True)

        explanation = generate_comparison_explanation(comparison)
        notice(f"{explanation}")
        if "note" in comparison:
            notice(comparison["note"], kind="warn")

        # Visual comparison
        section_heading("Risk Score vs Baseline Over Time")
        baseline_flags = frequency_baseline_predict(features_df, baseline_thresh)
        multi_flags = multisignal_predict(risk_df, risk_thresh)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=risk_df["period"], y=risk_df["risk_score"],
            mode="lines+markers", name="Multi-Signal Risk Score",
            line=dict(color="#f0a500"),
        ))
        fig.add_hline(y=risk_thresh, line_dash="dash", line_color="#f87171",
                      annotation_text=f"Risk Threshold ({risk_thresh})",
                      annotation_font_color="#f87171")
        fig.add_trace(go.Scatter(
            x=features_df["period"],
            y=features_df["overall_complaint_proportion"] * 100,
            mode="lines+markers", name="Complaint Proportion × 100",
            line=dict(color="#60a5fa"), yaxis="y2",
        ))
        layout = chart_layout_dark("Multi-Signal Risk vs Complaint Proportion", height=380)
        layout["yaxis"]["title"] = "Risk Score (0–100)"
        layout["yaxis"]["range"] = [0, 105]
        layout["yaxis2"] = dict(
            title="Complaint Review % (≤ 100%)", overlaying="y", side="right",
            range=[0, 105], gridcolor="#252d3f", linecolor="#252d3f", zeroline=False,
        )
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

        # Ablation
        section_heading("Ablation Analysis — Signal Contribution")
        notice(
            "Ablation removes one signal at a time and recalculates risk scores to quantify each signal's contribution.",
            kind="blue",
        )
        ablation_results = ablation_analysis(features_df)
        summary_df = ablation_summary(ablation_results)
        if len(summary_df) > 0:
            st.dataframe(summary_df, use_container_width=True)
            fig = go.Figure()
            abl_colors = {"full_model": "#f0a500"}
            for config_name, abl_df in ablation_results.items():
                if "risk_score" in abl_df.columns:
                    is_full = config_name == "full_model"
                    fig.add_trace(go.Scatter(
                        x=abl_df["period"], y=abl_df["risk_score"],
                        mode="lines", name=config_name.replace("_", " ").title(),
                        opacity=1.0 if is_full else 0.65,
                        line=dict(width=2.5 if is_full else 1.5,
                                  color="#f0a500" if is_full else None),
                    ))
            fig.update_yaxes(title_text="Risk Score (0–100)")
            fig.update_layout(**chart_layout_dark("Risk Score Under Ablation Configurations", height=420))
            st.plotly_chart(fig, use_container_width=True)

        # Evaluation status
        section_heading("Evaluation Status")
        with st.expander("What can and cannot be evaluated with this data", expanded=True):
            st.markdown("""
**Available without ground-truth labels:**
- ✅ Agreement analysis between baseline and multi-signal approaches
- ✅ Ablation analysis showing each signal's contribution
- ✅ Temporal plots of risk score evolution
- ✅ Descriptive anomaly detection results

**Requires independently verified failure labels:**
- ❌ Validated precision, recall, and F1 for failure prediction
- ❌ Calibrated lead-time analysis
- ❌ Confusion matrix against real failure events

To enable validated evaluation, provide a dataset with a `failure_label` column (boolean)
indicating independently verified product failure periods.
""")


# ═══════════════════════════════════════════════════════════════
# PAGE: Methodology
# ═══════════════════════════════════════════════════════════════
elif page == "Methodology":
    page_header("Methodology", "System architecture, feature definitions, and limitations")

    section_heading("1. Problem Statement")
    st.markdown(
        "E-commerce platforms contain large volumes of customer reviews. Ordinary sentiment analysis "
        "classifies reviews as positive or negative but does not identify whether several product complaints "
        "are *increasing together* over time. This system addresses the research question:\n\n"
        "> *\"Can the temporal co-evolution of multiple complaint signals provide an early warning of a "
        "potential product failure?\"*"
    )

    section_heading("2. System Architecture")
    st.code("""Customer Reviews (CSV)
→ Data Validation & Deduplication
→ Product and Date Identification
→ Aspect/Complaint Extraction  (keyword lexicon — not deep learning)
→ Sentiment & Severity Scoring (lexicon + star-rating blend)
→ Weekly or Monthly Temporal Aggregation
→ Multi-Signal Feature Engineering
→ Abnormal Trend Detection      (rolling z-score, EWMA baseline)
→ Product Risk Score (0–100)    (heuristic, not a failure probability)
→ Risk-Level Classification    (Low / Moderate / High / Critical)
→ Explainable Warning Generation
→ Interactive Dashboard & Evaluation""", language="text")

    section_heading("3. Data Ingestion & Validation")
    st.markdown("""
- **Deduplication** uses two passes: (1) duplicate `review_id` per product, (2) identical `(product_id, review_date, review_text)`.
- **Invalid dates** (unparseable strings, impossible values) are removed and reported. No replacement timestamps are invented.
- **Missing review text** rows are excluded from text analysis and counted in the validation report.
- **Missing ratings** are retained; rating-dependent features fall back to text-only severity.
- The validation report shows: rows read, rows retained, rows rejected, duplicates removed, and individual issues.
""")

    section_heading("4. Complaint Aspect Identification")
    st.markdown("""
Aspect extraction uses a **keyword lexicon** (not a transformer model). Each review text is scanned
for keywords matching 9 complaint aspects: Battery, Camera, Display, Performance, Charging, Heating,
Software, Durability, Connectivity.

A review is classified as a complaint (`is_complaint = True`) when it contains aspect keywords
**and** has a low star rating (≤ 2.0) or negative lexicon-based sentiment.
""")

    section_heading("5. Feature Definitions")
    st.markdown(r"""
**A. Overall Complaint-Review Proportion** — bounded strictly ≤ 100%:

$$\text{proportion}(t) = \frac{|\{r \in R_t : \text{is\_complaint}(r) = \text{True}\}|}{|R_t|}$$

**B. Aspect-Specific Complaint Proportion** — bounded strictly ≤ 100%:

$$\text{aspect\_proportion}(a, t) = \frac{|\{r \in R_t : a \in \text{aspects}(r)\}|}{|R_t|}$$

**C. Aspect Mention Density** — may exceed 100% (separate metric, not complaint proportion):

$$\text{mentions\_per\_review}(t) = \frac{\sum_{a \in A} \text{frequency}(a, t)}{|R_t|}$$

**D. Safe Complaint Growth Rate** (floor prevents zero-division):

$$\text{growth}(t) = \frac{\text{proportion}(t) - \text{proportion}(t-1)}{\max(\text{proportion}(t-1),\; 0.05)}$$

**E. Symptom Diversity** (0–1):

$$\text{diversity}(t) = \frac{|\{a : \text{frequency}(a, t) > 0\}|}{|A|}$$
""")

    section_heading("6. Risk Score Formula")
    st.latex(r"\text{RiskScore} = 100 \times \sum_{s \in S} w_s \cdot \hat{s}")
    st.markdown("""
Where $\\hat{s}$ is the normalized signal in [0,1] and $w_s$ is its weight.

| Signal | Weight | Normalization |
|--------|--------|---------------|
| Complaint Frequency | 25% | Already in [0,1] |
| Complaint Severity | 20% | Already in [0,1] |
| Complaint Growth | 25% | tanh(growth) clipped to [0,1] |
| Symptom Diversity | 15% | Already in [0,1] |
| Review Velocity | 15% | tanh(velocity_ratio - 1) |

**Risk Bands:** Low 0–25 · Moderate 25–50 · High 50–75 · Critical 75–100.
⚠️ These are heuristic categories, not calibrated failure probabilities.
""")

    section_heading("7. Anomaly Detection")
    st.markdown(r"""
Uses a **rolling baseline deviation** method. For each period $t$ (after sufficient history):

$$z_s(t) = \frac{s(t) - \mu_s(1\ldots t-1)}{\sigma_s(1\ldots t-1)}$$

An anomaly is flagged if $z_s(t) > \theta$ for any monitored signal $s$.

- Default threshold: $\theta = 2.0$ standard deviations  
- Minimum floor on $\sigma$ to prevent instability from zero variance  
- Requires ≥ 3 prior periods — no warnings are issued before this  
- Historical baseline uses **only past data** (no future leakage)  
- Confidence is labeled explicitly as *heuristic*, not statistically calibrated
""")

    section_heading("8. Evaluation Approach")
    st.markdown("""
- **Approach comparison:** Multi-signal flags are compared with a simple complaint-frequency baseline.
- **Ablation analysis:** Each signal is removed in turn to measure its contribution.
- **No ground-truth validation:** Without independently verified product-failure labels, precision, recall, F1, and lead time cannot be validly calculated.
""")

    section_heading("9. Limitations & Future Improvements")
    st.markdown("""
1. **Keyword-based extraction** may miss indirect references, misspellings, or contextual negation
   (e.g., "battery is great" matches Battery — polarity is handled by sentiment scoring separately).
2. **Risk weights are engineering estimates**, not empirically optimized.
3. **The risk score is a heuristic indicator**, not a calibrated failure probability.
4. **Without verified failure labels**, predictive performance cannot be validly evaluated.
5. **The system estimates emerging complaint patterns**, not proven physical product failures.
6. **Synthetic demo data** is for software testing only and must not be presented as real-world results.

*Future improvements could include: transformer-based aspect extraction, labelled failure datasets for calibration, real-time data ingestion, and multi-product portfolio dashboards.*
""")
