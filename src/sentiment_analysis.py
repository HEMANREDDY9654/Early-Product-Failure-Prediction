"""Sentiment analysis and severity scoring module.

Provides sentiment analysis using:
1. Primary: HuggingFace Transformers pretrained model (distilbert-base-uncased-finetuned-sst-2-english)
2. Fallback: VADER-style rule-based sentiment (no external downloads required)

The module caches loaded models and handles download failures gracefully.
"""
import re
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import warnings

# Model cache
_transformer_pipeline = None
_transformer_available = None


def _load_transformer_model():
    """Attempt to load the transformer sentiment model.
    
    Returns:
        Pipeline object or None if unavailable.
    """
    global _transformer_pipeline, _transformer_available
    
    if _transformer_available is not None:
        return _transformer_pipeline
    
    try:
        from transformers import pipeline
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import SENTIMENT_MODEL_NAME, SENTIMENT_MAX_LENGTH
        
        _transformer_pipeline = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL_NAME,
            truncation=True,
            max_length=SENTIMENT_MAX_LENGTH,
        )
        _transformer_available = True
        return _transformer_pipeline
    except Exception as e:
        warnings.warn(
            f"Transformer sentiment model unavailable: {e}. "
            f"Falling back to rule-based sentiment analysis."
        )
        _transformer_available = False
        _transformer_pipeline = None
        return None


def is_transformer_available() -> bool:
    """Check if transformer model is available."""
    global _transformer_available
    if _transformer_available is None:
        _load_transformer_model()
    return _transformer_available or False


def analyze_sentiment_transformer(text: str) -> Dict:
    """Analyze sentiment using transformer model.
    
    Args:
        text: Review text.
        
    Returns:
        Dict with 'label' (POSITIVE/NEGATIVE), 'score' (confidence 0-1),
        and 'severity' (0-1, higher = more negative).
    """
    pipe = _load_transformer_model()
    if pipe is None:
        return analyze_sentiment_fallback(text)
    
    try:
        # Truncate very long texts
        truncated = text[:2000] if len(text) > 2000 else text
        result = pipe(truncated)[0]
        
        label = result["label"].upper()
        score = result["score"]
        
        # Map to severity: NEGATIVE with high confidence -> high severity
        if label == "NEGATIVE":
            severity = 0.5 + (score * 0.5)  # Range: 0.5 to 1.0
        else:
            severity = 0.5 - (score * 0.5)  # Range: 0.0 to 0.5
        
        severity = max(0.0, min(1.0, severity))
        
        return {
            "label": label,
            "score": score,
            "severity": severity,
            "method": "transformer",
        }
    except Exception:
        return analyze_sentiment_fallback(text)


def analyze_sentiment_fallback(text: str) -> Dict:
    """Rule-based fallback sentiment analysis.
    
    Uses a simple lexicon approach when transformers are unavailable.
    This is a lightweight approximation - not as accurate as model-based approaches.
    
    Args:
        text: Review text.
        
    Returns:
        Dict with 'label', 'score', 'severity', and 'method'.
    """
    if not text or not isinstance(text, str):
        return {"label": "NEUTRAL", "score": 0.5, "severity": 0.5, "method": "fallback"}
    
    text_lower = text.lower()
    
    positive_words = {
        "good", "great", "excellent", "amazing", "awesome", "fantastic",
        "wonderful", "perfect", "love", "loved", "best", "happy",
        "satisfied", "recommend", "impressive", "beautiful", "nice",
        "superb", "outstanding", "brilliant", "solid", "reliable",
        "smooth", "fast", "quality", "worth", "pleased",
    }
    
    negative_words = {
        "bad", "terrible", "horrible", "awful", "worst", "poor",
        "disappointing", "disappointed", "frustrating", "frustrated",
        "broken", "defective", "useless", "waste", "hate", "hated",
        "regret", "slow", "laggy", "overheating", "overheat",
        "crash", "crashes", "freeze", "freezing", "dead", "fail",
        "failed", "failure", "problem", "issue", "bug", "buggy",
        "cheap", "flimsy", "junk", "garbage", "rubbish", "annoying",
        "uncomfortable", "unreliable", "ugly", "burning", "scorching",
        "drain", "drains", "draining", "discharge", "discharging",
        "refuses", "refused", "disconnect", "disconnects", "disconnecting",
        "flicker", "flickers", "flickering", "dangerous", "stutter",
        "stuttering", "stutters",
    }
    
    intensifiers = {"very", "extremely", "really", "highly", "absolutely", "totally", "completely"}
    negators = {"not", "no", "never", "neither", "nor", "don't", "doesn't", "didn't", "won't", "can't", "couldn't"}
    
    words = re.findall(r'\b\w+\b', text_lower)
    
    pos_count = 0
    neg_count = 0
    intensity_boost = 0
    
    for i, word in enumerate(words):
        is_negated = (i > 0 and words[i-1] in negators)
        is_intensified = (i > 0 and words[i-1] in intensifiers)
        boost = 0.5 if is_intensified else 0
        
        if word in positive_words:
            if is_negated:
                neg_count += 1 + boost
            else:
                pos_count += 1 + boost
        elif word in negative_words:
            if is_negated:
                pos_count += 1 + boost
            else:
                neg_count += 1 + boost
    
    total = pos_count + neg_count
    if total == 0:
        return {"label": "NEUTRAL", "score": 0.5, "severity": 0.5, "method": "fallback"}
    
    neg_ratio = neg_count / total
    pos_ratio = pos_count / total
    
    # Determine label
    if neg_ratio > 0.6:
        label = "NEGATIVE"
        confidence = min(0.95, 0.5 + neg_ratio * 0.45)
    elif pos_ratio > 0.6:
        label = "POSITIVE"
        confidence = min(0.95, 0.5 + pos_ratio * 0.45)
    else:
        label = "NEUTRAL"
        confidence = 0.5 + abs(neg_ratio - pos_ratio) * 0.3
    
    # Calculate severity (0-1, higher = more negative)
    severity = neg_ratio
    severity = max(0.0, min(1.0, severity))
    
    return {
        "label": label,
        "score": confidence,
        "severity": severity,
        "method": "fallback",
    }


def analyze_sentiment(text: str, use_transformer: bool = True) -> Dict:
    """Analyze sentiment of a review text.
    
    Tries transformer model first (if requested and available),
    falls back to rule-based analysis.
    
    Args:
        text: Review text.
        use_transformer: Whether to attempt transformer model.
        
    Returns:
        Dict with 'label', 'score', 'severity', and 'method'.
    """
    if use_transformer:
        return analyze_sentiment_transformer(text)
    return analyze_sentiment_fallback(text)


def combine_severity_with_rating(
    text_severity: float,
    rating: Optional[float],
    text_weight: float = 0.7,
    rating_weight: float = 0.3,
) -> float:
    """Combine text-based severity with star rating.
    
    The star rating provides supplementary evidence. A low rating
    increases severity; a high rating may moderate it.
    
    We avoid double-counting by weighting text sentiment more heavily,
    since the text usually provides more specific information.
    
    Args:
        text_severity: Severity from text analysis (0-1).
        rating: Star rating (1-5) or None.
        text_weight: Weight for text severity (default 0.7).
        rating_weight: Weight for rating severity (default 0.3).
        
    Returns:
        Combined severity score (0-1).
    """
    if rating is None or np.isnan(rating):
        return text_severity
    
    # Convert rating to severity: rating 1 -> severity 1.0, rating 5 -> severity 0.0
    rating_severity = (5.0 - float(rating)) / 4.0
    rating_severity = max(0.0, min(1.0, rating_severity))
    
    combined = text_weight * text_severity + rating_weight * rating_severity
    return max(0.0, min(1.0, combined))


def analyze_batch(
    texts: List[str],
    ratings: Optional[List[Optional[float]]] = None,
    use_transformer: bool = True,
) -> List[Dict]:
    """Analyze sentiment for a batch of reviews.
    
    Args:
        texts: List of review texts.
        ratings: Optional list of ratings (parallel to texts).
        use_transformer: Whether to attempt transformer model.
        
    Returns:
        List of sentiment result dicts, each with 'combined_severity'.
    """
    results = []
    for i, text in enumerate(texts):
        result = analyze_sentiment(text, use_transformer)
        rating = None
        if ratings is not None and i < len(ratings):
            rating = ratings[i]
        result["combined_severity"] = combine_severity_with_rating(
            result["severity"], rating
        )
        results.append(result)
    return results


def reset_model_cache():
    """Reset the cached transformer model. Useful for testing."""
    global _transformer_pipeline, _transformer_available
    _transformer_pipeline = None
    _transformer_available = None
