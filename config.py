"""Configuration for Early Product Failure Prediction system."""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT / "data"

# Aggregation options
AGGREGATION_OPTIONS = ["weekly", "monthly"]
DEFAULT_AGGREGATION = "weekly"

# Risk score weights (must sum to 1.0)
DEFAULT_RISK_WEIGHTS = {
    "frequency": 0.25,
    "severity": 0.20,
    "growth": 0.25,
    "symptom_diversity": 0.15,
    "review_velocity": 0.15,
}

# Risk bands
RISK_BANDS = {
    "Low": (0, 25),
    "Moderate": (25, 50),
    "High": (50, 75),
    "Critical": (75, 100),
}

# Supported aspects and their lexicons
ASPECT_LEXICON = {
    "Battery": [
        "battery", "battery life", "battery drain", "battery drains",
        "battery drains quickly", "power drain", "battery dies",
        "battery dead", "battery percentage", "battery health",
        "battery swelling", "battery swollen", "mah", "power consumption",
        "battery backup", "standby time", "battery discharge",
    ],
    "Camera": [
        "camera", "photo", "photos", "picture", "pictures", "lens",
        "selfie", "zoom", "focus", "autofocus", "blurry", "blurred",
        "image quality", "camera quality", "megapixel", "video recording",
        "night mode", "portrait mode", "camera app", "shutter",
    ],
    "Display": [
        "display", "screen", "touchscreen", "touch screen",
        "screen flickers", "flickering", "dead pixel", "dead pixels",
        "brightness", "dim", "oled", "lcd", "amoled", "screen crack",
        "screen broken", "ghost touch", "screen burn", "burn in",
        "screen protector", "resolution", "refresh rate",
    ],
    "Performance": [
        "slow", "lag", "lagging", "laggy", "hang", "hanging", "freeze",
        "freezing", "frozen", "sluggish", "performance", "speed",
        "ram", "memory", "processor", "cpu", "gpu", "benchmark",
        "stuttering", "stutter", "jank", "janky", "responsive",
        "unresponsive", "app crash", "apps crash",
    ],
    "Charging": [
        "charging", "charger", "charge", "slow charging",
        "fast charging", "wireless charging", "charging port",
        "usb", "cable", "adapter", "not charging", "charging speed",
        "charging time", "quick charge", "watt", "stop charging",
        "charging failure", "refuses to charge", "refuses to fast charge",
    ],
    "Heating": [
        "heat", "heating", "overheat", "overheating", "overheats",
        "hot", "warm", "thermal", "temperature", "burn", "burning",
        "burning hot", "scorching", "scorching hot", "gets hot", "too hot",
        "heats up", "heat up", "thermal throttling", "uncomfortably hot",
        "dangerous heating",
    ],
    "Software": [
        "software", "bug", "bugs", "buggy", "glitch",
        "glitchy", "crash", "crashes", "crashing", "os", "android",
        "ios", "firmware", "frequent crashes", "system update",
        "software update", "firmware update", "update broke",
        "apps crash", "app crash", "system crash",
    ],
    "Durability": [
        "durability", "durable", "build quality", "build", "flimsy",
        "fragile", "break", "broke", "broken", "crack", "cracked",
        "scratch", "scratched", "dent", "dented", "water damage",
        "waterproof", "water resistant", "drop", "dropped",
        "shatter", "shattered", "bend", "bent",
    ],
    "Connectivity": [
        "wifi", "wi-fi", "bluetooth", "network", "signal",
        "connection", "disconnect", "disconnecting", "cellular",
        "mobile data", "4g", "5g", "lte", "gps", "nfc",
        "hotspot", "tethering", "sim", "sim card", "call drop",
        "call drops", "no signal", "weak signal", "reception",
    ],
}

# Supported column name mappings
COLUMN_ALIASES = {
    "review_id": ["review_id", "reviewid", "id", "rev_id", "review_num"],
    "product_id": ["product_id", "product", "asin", "product_name", "item_id", "sku", "productid"],
    "review_date": ["review_date", "date", "timestamp", "review_time", "created_at", "reviewtime", "unixreviewtime"],
    "review_text": ["review_text", "text", "review", "body", "comment", "reviewtext", "review_body", "content"],
    "rating": ["rating", "score", "overall", "stars", "star_rating", "review_rating"],
}

# Sentiment model configuration
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_MAX_LENGTH = 512
SENTIMENT_BATCH_SIZE = 32

# Anomaly detection
EWMA_SPAN = 4  # Number of periods for EWMA
ANOMALY_THRESHOLD = 2.0  # Standard deviations above mean
MIN_HISTORY_PERIODS = 3  # Minimum periods before issuing warnings
MIN_REVIEWS_FOR_WARNING = 5  # Minimum reviews for warning confidence assessment
MIN_STD_FLOOR = 0.05  # Floor for standard deviation to avoid tiny-sample division instability
MIN_ABSOLUTE_INCREASE = 0.05  # Minimum absolute signal increase required for anomaly

# Baseline
BASELINE_COMPLAINT_THRESHOLD = 0.3  # Proportion threshold for baseline

# Random seed for reproducibility
RANDOM_SEED = 42

# Synthetic data settings
SYNTHETIC_NUM_PRODUCTS = 5
SYNTHETIC_REVIEWS_PER_PRODUCT = 200
SYNTHETIC_DATE_START = "2024-01-01"
SYNTHETIC_DATE_END = "2024-12-31"
