"""Generate synthetic demonstration dataset for Early Product Failure Prediction.

This script creates a clearly labelled synthetic dataset for testing and
demonstration purposes. The generated data should NOT be presented as
real customer reviews.

Usage:
    python scripts/generate_demo_data.py
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED, DATA_DIR

np.random.seed(RANDOM_SEED)

# Product definitions with their "failure" patterns
PRODUCTS = {
    "PHONE_A1": {
        "name": "SmartPhone Alpha",
        "description": "Product with gradual battery and heating issues emerging around week 20",
        "normal_aspects": ["Camera", "Display", "Software"],
        "failure_aspects": ["Battery", "Heating", "Charging"],
        "failure_start_week": 20,
        "failure_ramp_weeks": 8,
    },
    "PHONE_B2": {
        "name": "SmartPhone Beta",
        "description": "Product with sudden software and performance crash around week 30",
        "normal_aspects": ["Battery", "Camera", "Display"],
        "failure_aspects": ["Software", "Performance"],
        "failure_start_week": 30,
        "failure_ramp_weeks": 4,
    },
    "PHONE_C3": {
        "name": "SmartPhone Gamma",
        "description": "Product with display and durability issues from week 15",
        "normal_aspects": ["Battery", "Camera", "Software"],
        "failure_aspects": ["Display", "Durability"],
        "failure_start_week": 15,
        "failure_ramp_weeks": 10,
    },
    "PHONE_D4": {
        "name": "SmartPhone Delta",
        "description": "Stable product with no emerging failure pattern (control)",
        "normal_aspects": ["Camera", "Display", "Software", "Battery"],
        "failure_aspects": [],
        "failure_start_week": 999,
        "failure_ramp_weeks": 0,
    },
    "PHONE_E5": {
        "name": "SmartPhone Epsilon",
        "description": "Product with connectivity issues emerging around week 25",
        "normal_aspects": ["Camera", "Battery", "Display"],
        "failure_aspects": ["Connectivity", "Software"],
        "failure_start_week": 25,
        "failure_ramp_weeks": 6,
    },
}

# Review templates for different scenarios
POSITIVE_TEMPLATES = [
    "Really happy with this phone. The {aspect} works perfectly.",
    "Great product! The {aspect} is excellent and I'm very satisfied.",
    "Love the {aspect} on this device. Best purchase I've made.",
    "Impressive {aspect}. Works smoothly and looks great.",
    "Five stars! The {aspect} exceeded my expectations.",
    "Very good phone overall. The {aspect} is a standout feature.",
    "No complaints about the {aspect}. Works as advertised.",
    "Solid {aspect} performance. Would recommend to others.",
    "Happy customer here. The {aspect} is top notch.",
    "Excellent value. The {aspect} quality is amazing for this price.",
]

NEGATIVE_TEMPLATES = {
    "Battery": [
        "Battery drains so quickly, barely lasts half a day.",
        "The battery life is terrible. Battery dies before noon.",
        "Battery drains quickly even on standby. Very disappointed.",
        "Battery health dropped to 80% in just two months.",
        "Power consumption is way too high. Battery drain is a real problem.",
        "Battery backup is awful. Need to charge three times a day.",
        "The battery percentage drops from 100 to 50 in two hours.",
    ],
    "Camera": [
        "Camera quality is disappointing. Photos come out blurry.",
        "The camera app keeps crashing. Blurred photos every time.",
        "Autofocus doesn't work properly. Pictures are blurry.",
        "Image quality has degraded after the update. Photos look washed out.",
        "Night mode is useless. Photos are grainy and dark.",
    ],
    "Display": [
        "Screen flickers randomly. Very annoying to look at.",
        "Dead pixels appeared after a month. Screen has bright spots.",
        "The touchscreen is unresponsive in certain areas.",
        "Ghost touch issues make the phone unusable. Screen keeps flickering.",
        "Screen burn-in after just a few weeks. Display quality is poor.",
    ],
    "Performance": [
        "Phone is extremely slow and laggy. Apps take forever to load.",
        "Constant freezing and hanging. Phone becomes unresponsive.",
        "Performance has degraded significantly. Everything feels sluggish.",
        "Apps crash frequently. The phone stutters during basic tasks.",
        "RAM management is terrible. Apps keep closing in background.",
    ],
    "Charging": [
        "Slow charging takes over 3 hours. Not charging properly.",
        "Charging port is loose. Cable keeps disconnecting.",
        "Fast charging stopped working after update.",
        "Phone not charging with original cable. Adapter seems defective.",
        "Wireless charging is unreliable. Takes forever to charge.",
    ],
    "Heating": [
        "Phone overheats during normal use. Gets too hot to hold.",
        "Severe overheating problem. The phone heats up while charging.",
        "Thermal throttling makes the phone slow when it gets hot.",
        "Phone gets extremely warm even during light usage.",
        "Overheats during video calls. Temperature is dangerously high.",
    ],
    "Software": [
        "Software is buggy and full of glitches. Frequent crashes.",
        "System update broke several features. Bugs everywhere.",
        "Notifications don't work properly after firmware update.",
        "UI is laggy and the interface feels unpolished. Crashes daily.",
        "Software update caused battery drain and crashes.",
    ],
    "Durability": [
        "Build quality is poor. Phone feels flimsy and cheap.",
        "Back panel cracked from a small drop. Not durable at all.",
        "Screen shattered despite having a protector. Fragile build.",
        "Phone scratches very easily. The paint is already chipping.",
        "Water damage despite claimed water resistance. Broke after light rain.",
    ],
    "Connectivity": [
        "WiFi keeps disconnecting randomly. Very frustrating.",
        "Bluetooth is unreliable. Connection drops frequently.",
        "Weak signal reception. Call drops in areas where other phones work fine.",
        "GPS is inaccurate. Mobile data keeps dropping out.",
        "5G doesn't work properly. Network connection is unstable.",
    ],
}

NEUTRAL_TEMPLATES = [
    "It's an okay phone. Does what it's supposed to do.",
    "Average product. Nothing special but nothing terrible.",
    "Decent phone for the price. Meets basic expectations.",
    "It works fine for everyday use. No major complaints.",
    "Mediocre at best. Expected a bit more for the price.",
]


def generate_review(product_id: str, product_info: dict, week_number: int, date: datetime) -> dict:
    """Generate a single synthetic review."""
    is_failure_period = week_number >= product_info["failure_start_week"]
    
    if is_failure_period and product_info["failure_aspects"]:
        weeks_into_failure = week_number - product_info["failure_start_week"]
        ramp = product_info["failure_ramp_weeks"]
        # Probability of negative review increases over the ramp period
        failure_prob = min(0.7, 0.2 + (weeks_into_failure / max(ramp, 1)) * 0.5)
        
        if np.random.random() < failure_prob:
            # Generate negative review about failure aspect
            aspect = np.random.choice(product_info["failure_aspects"])
            templates = NEGATIVE_TEMPLATES.get(aspect, NEGATIVE_TEMPLATES["Software"])
            text = np.random.choice(templates)
            rating = np.random.choice([1, 2, 2, 1, 3], p=[0.3, 0.35, 0.2, 0.1, 0.05])
            return {
                "product_id": product_id,
                "review_date": date.strftime("%Y-%m-%d"),
                "review_text": f"[SYNTHETIC] {text}",
                "rating": int(rating),
            }
    
    # Normal review (can be positive, neutral, or mildly negative)
    review_type = np.random.choice(["positive", "neutral", "mild_negative"], p=[0.5, 0.25, 0.25])
    
    if review_type == "positive":
        if product_info["normal_aspects"]:
            aspect = np.random.choice(product_info["normal_aspects"])
        else:
            aspect = "product"
        text = np.random.choice(POSITIVE_TEMPLATES).format(aspect=aspect.lower())
        rating = np.random.choice([4, 5, 5, 4, 3], p=[0.25, 0.4, 0.2, 0.1, 0.05])
    elif review_type == "neutral":
        text = np.random.choice(NEUTRAL_TEMPLATES)
        rating = np.random.choice([3, 3, 4, 2], p=[0.4, 0.3, 0.2, 0.1])
    else:
        # Mild negative (baseline noise)
        random_aspect = np.random.choice(list(NEGATIVE_TEMPLATES.keys()))
        templates = NEGATIVE_TEMPLATES[random_aspect]
        text = np.random.choice(templates)
        rating = np.random.choice([2, 3, 2, 1], p=[0.4, 0.3, 0.2, 0.1])
    
    return {
        "product_id": product_id,
        "review_date": date.strftime("%Y-%m-%d"),
        "review_text": f"[SYNTHETIC] {text}",
        "rating": int(rating),
    }


def generate_dataset() -> pd.DataFrame:
    """Generate the complete synthetic demonstration dataset."""
    all_reviews = []
    start_date = datetime(2024, 1, 1)
    total_weeks = 48  # ~11 months of data
    
    for product_id, product_info in PRODUCTS.items():
        # Base reviews per week (varies slightly)
        base_reviews_per_week = np.random.randint(3, 8)
        
        for week in range(total_weeks):
            week_start = start_date + timedelta(weeks=week)
            
            # Slightly vary review count per week
            n_reviews = max(1, int(np.random.normal(base_reviews_per_week, 1.5)))
            
            # Increase reviews during failure period (more people complaining)
            is_failure = week >= product_info.get("failure_start_week", 999)
            if is_failure and product_info["failure_aspects"]:
                n_reviews = int(n_reviews * 1.3)
            
            for j in range(n_reviews):
                day_offset = np.random.randint(0, 7)
                review_date = week_start + timedelta(days=day_offset)
                review = generate_review(product_id, product_info, week, review_date)
                all_reviews.append(review)
    
    df = pd.DataFrame(all_reviews)
    df = df.sort_values("review_date").reset_index(drop=True)
    
    return df


def main():
    """Generate and save synthetic dataset."""
    print("Generating synthetic demonstration dataset...")
    print("NOTE: This data is synthetic and clearly labelled. Do NOT present as real customer data.")
    print()
    
    df = generate_dataset()
    
    # Create data directory if needed
    os.makedirs(DATA_DIR, exist_ok=True)
    
    output_path = DATA_DIR / "synthetic_reviews.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} synthetic reviews for {df['product_id'].nunique()} products")
    print(f"Date range: {df['review_date'].min()} to {df['review_date'].max()}")
    print(f"Saved to: {output_path}")
    print()
    print("Product descriptions:")
    for pid, info in PRODUCTS.items():
        print(f"  {pid}: {info['description']}")
    print()
    print("Rating distribution:")
    print(df['rating'].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
