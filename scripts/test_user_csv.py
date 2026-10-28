"""Test script to verify pipeline on reviews_dataset.csv."""
import sys
import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import validate_and_load_csv, filter_by_product, aggregate_by_period
from src.preprocessing import preprocess_reviews
from src.temporal_features import calculate_period_features
from src.risk_scoring import calculate_risk_timeseries
from src.anomaly_detection import detect_anomalies, get_top_anomaly_aspects
from src.explanations import generate_risk_explanation, generate_anomaly_explanation

def run():
    csv_path = r"C:\Users\PRANAV REDDY\Downloads\reviews_dataset.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    print("=" * 60)
    print("1. LOADING & VALIDATING CSV")
    print("=" * 60)
    df, report = validate_and_load_csv(csv_path)
    for k, v in report.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("2. PRODUCT SEPARATION & TIME SERIES")
    print("=" * 60)
    for pid in df["product_id"].unique():
        print(f"\n>>> Analyzing Product: {pid}")
        pdf = filter_by_product(df, pid)
        proc = aggregate_by_period(preprocess_reviews(pdf, use_transformer=False), "monthly")
        feats = calculate_period_features(proc)
        risk = calculate_risk_timeseries(feats)
        anom = detect_anomalies(risk)
        
        print(f"Total valid reviews: {len(pdf)}")
        print(f"Monthly periods: {len(feats)}")
        for idx, r in risk.iterrows():
            prop = r["overall_complaint_proportion"]
            revs = r["total_reviews"]
            crevs = r["complaint_reviews_count"]
            mentions = r["total_aspect_mentions"]
            print(f"  Period: {r['period'].strftime('%Y-%m-%d')} | Total Revs: {revs} | Complaints: {crevs} | Prop: {prop:.1%} ({crevs}/{revs}) | Aspect Mentions: {mentions} | Risk: {r['risk_score']:.1f} ({r['risk_level']})")

        anom_periods = anom[anom["is_anomaly"] == True]
        print(f"Anomalies detected: {len(anom_periods)}")
        for idx, ar in anom_periods.iterrows():
            pidx = anom.index.get_loc(idx)
            top = get_top_anomaly_aspects(feats, pidx)
            expl = generate_anomaly_explanation(ar, top)
            print(f"  ALERT [{ar['period'].strftime('%Y-%m-%d')}]: {expl}")

if __name__ == "__main__":
    run()
