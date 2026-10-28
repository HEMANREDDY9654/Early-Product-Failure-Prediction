"""Preprocessing module - bridges data loading with analysis pipeline.

Applies aspect extraction and sentiment analysis to the validated DataFrame,
preparing it for temporal feature engineering.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.aspect_extraction import extract_aspects, has_complaint_indicators
from src.sentiment_analysis import analyze_sentiment, combine_severity_with_rating


def preprocess_reviews(
    df: pd.DataFrame,
    use_transformer: bool = True,
    progress_callback=None,
) -> pd.DataFrame:
    """Apply aspect extraction and sentiment analysis to all reviews.
    
    Args:
        df: Validated DataFrame with review_text, rating columns.
        use_transformer: Whether to attempt transformer sentiment model.
        progress_callback: Optional callable(current, total) for progress.
        
    Returns:
        DataFrame with added columns: aspects, sentiment_label, sentiment_score,
        severity, combined_severity, is_complaint.
    """
    if df.empty:
        df["aspects"] = []
        df["sentiment_label"] = ""
        df["sentiment_score"] = 0.0
        df["severity"] = 0.0
        df["combined_severity"] = 0.0
        df["is_complaint"] = False
        return df
    
    df = df.copy()
    n = len(df)
    
    aspects_list = []
    labels = []
    scores = []
    severities = []
    combined_sevs = []
    is_complaints = []
    
    for i, (idx, row) in enumerate(df.iterrows()):
        text = str(row.get("review_text", ""))
        rating = row.get("rating", None)
        
        # Aspect extraction
        aspects = extract_aspects(text)
        
        # If no specific aspect matched but text has complaint indicators,
        # classify as "Other"
        if not aspects and has_complaint_indicators(text):
            aspects = ["Other"]
        
        aspects_list.append(aspects)
        
        # Sentiment analysis
        result = analyze_sentiment(text, use_transformer)
        labels.append(result["label"])
        scores.append(result["score"])
        severities.append(result["severity"])
        
        # Combine with rating
        r = rating if pd.notna(rating) else None
        combined = combine_severity_with_rating(result["severity"], r)
        combined_sevs.append(combined)
        
        # Is complaint: mentions an aspect AND has negative sentiment, elevated severity, or low star rating (<= 2)
        has_negative_rating = (r is not None and r <= 2.0)
        is_complaint = (
            len(aspects) > 0
            and (result["label"] == "NEGATIVE" or combined >= 0.5 or has_negative_rating)
        )
        is_complaints.append(is_complaint)
        
        if progress_callback and (i + 1) % 50 == 0:
            progress_callback(i + 1, n)
    
    df["aspects"] = aspects_list
    df["sentiment_label"] = labels
    df["sentiment_score"] = scores
    df["severity"] = severities
    df["combined_severity"] = combined_sevs
    df["is_complaint"] = is_complaints
    
    if progress_callback:
        progress_callback(n, n)
    
    return df
