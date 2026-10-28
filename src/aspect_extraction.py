"""Aspect and complaint extraction module.

Extracts product complaint aspects from review text using a configurable
keyword lexicon. Designed to be modular so semantic similarity approaches
can be added later.

Limitations:
- Keyword matching may miss indirect references or misspellings.
- Context is not considered (e.g., 'battery is great' still matches 'Battery').
  The sentiment module handles polarity separately.
- Domain-specific jargon may not be covered by default lexicon.
"""
from typing import Dict, List, Set, Optional
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ASPECT_LEXICON


def extract_aspects(
    text: str,
    lexicon: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """Extract complaint aspects from review text.
    
    A review may mention multiple aspects. All matching aspects are returned.
    If no specific aspect matches, and the review appears to contain a complaint
    (based on negative indicators), it is classified as 'Other'.
    
    Args:
        text: Review text string.
        lexicon: Optional custom lexicon. Uses default if None.
        
    Returns:
        List of matched aspect names (may be empty or contain multiple entries).
    """
    if lexicon is None:
        lexicon = ASPECT_LEXICON
    
    if not text or not isinstance(text, str):
        return []
    
    text_lower = text.lower().strip()
    if not text_lower:
        return []
    
    matched_aspects = []
    
    for aspect, keywords in lexicon.items():
        for keyword in keywords:
            # Use word boundary matching for short keywords,
            # substring matching for multi-word phrases
            if len(keyword.split()) > 1:
                if keyword.lower() in text_lower:
                    matched_aspects.append(aspect)
                    break
            else:
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    matched_aspects.append(aspect)
                    break
    
    return matched_aspects


def extract_aspects_batch(
    texts: List[str],
    lexicon: Optional[Dict[str, List[str]]] = None,
) -> List[List[str]]:
    """Extract aspects from a batch of review texts.
    
    Args:
        texts: List of review text strings.
        lexicon: Optional custom lexicon.
        
    Returns:
        List of aspect lists, one per input text.
    """
    return [extract_aspects(text, lexicon) for text in texts]


def get_supported_aspects(lexicon: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """Return list of supported aspect names."""
    if lexicon is None:
        lexicon = ASPECT_LEXICON
    return list(lexicon.keys())


def get_aspect_keywords(aspect: str, lexicon: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """Return keywords for a specific aspect."""
    if lexicon is None:
        lexicon = ASPECT_LEXICON
    return lexicon.get(aspect, [])


def has_complaint_indicators(text: str) -> bool:
    """Check if text contains general complaint/negative indicators.
    
    Used to classify reviews as 'Other' complaints when no specific
    aspect is matched but the review appears negative.
    
    Args:
        text: Review text.
        
    Returns:
        True if complaint indicators are found.
    """
    if not text or not isinstance(text, str):
        return False
    
    indicators = [
        "problem", "issue", "defect", "defective", "broken", "fail",
        "failed", "failure", "worst", "terrible", "horrible", "awful",
        "unusable", "waste", "refund", "return", "disappointed",
        "disappointing", "frustrating", "frustrated", "regret",
        "not working", "stopped working", "doesn't work", "does not work",
        "don't buy", "do not buy", "poor quality", "bad quality",
        "very bad", "very poor",
    ]
    
    text_lower = text.lower()
    for indicator in indicators:
        if len(indicator.split()) > 1:
            if indicator in text_lower:
                return True
        else:
            if re.search(r'\b' + re.escape(indicator) + r'\b', text_lower):
                return True
    return False
