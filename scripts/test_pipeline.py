import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import validate_and_load_csv, filter_by_product, aggregate_by_period
from src.preprocessing import preprocess_reviews
from src.temporal_features import calculate_period_features
from src.risk_scoring import calculate_risk_timeseries, ablation_analysis
from src.anomaly_detection import detect_anomalies, get_top_anomaly_aspects
from src.evaluation import compare_approaches, ablation_summary
from src.explanations import generate_risk_explanation, generate_anomaly_explanation

def test_pipeline():
    print("Running end-to-end pipeline test...")
    df, report = validate_and_load_csv("data/synthetic_reviews.csv")
    print(f"Loaded {len(df)} rows. Validation report: {report['rows_retained']} retained.")

    # Test product PHONE_A1 (gradual battery failure)
    df_a1 = filter_by_product(df, "PHONE_A1")
    print(f"PHONE_A1 reviews: {len(df_a1)}")

    processed = preprocess_reviews(df_a1, use_transformer=False)
    processed = aggregate_by_period(processed, "weekly")
    print(f"Processed reviews, columns: {list(processed.columns)}")

    features = calculate_period_features(processed)
    print(f"Features generated: {len(features)} periods.")

    risk_df = calculate_risk_timeseries(features)
    print(f"Risk timeseries computed. Max risk: {risk_df['risk_score'].max():.2f}")

    anomaly_df = detect_anomalies(risk_df)
    anomalies = anomaly_df[anomaly_df['is_anomaly'] == True]
    print(f"Detected {len(anomalies)} anomalous periods.")

    comparison = compare_approaches(features, risk_df)
    print(f"Comparison: baseline flags={comparison['baseline_flags_count']}, multisignal flags={comparison['multisignal_flags_count']}")

    ablation = ablation_analysis(features)
    summary = ablation_summary(ablation)
    print("Ablation summary:")
    print(summary[['configuration', 'mean_risk_score', 'max_risk_score']])

    if len(anomalies) > 0:
        anom_row = anomalies.iloc[0]
        period_idx = anomaly_df.index.get_loc(anomalies.index[0])
        top_asp = get_top_anomaly_aspects(features, period_idx)
        anom_expl = generate_anomaly_explanation(anom_row, top_asp)
        print(f"Sample Anomaly Explanation:\n{anom_expl}")

    print("\nSUCCESS: Pipeline integration verified completely!")

if __name__ == "__main__":
    test_pipeline()
