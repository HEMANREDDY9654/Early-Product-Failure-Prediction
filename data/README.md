# Data Directory

## Expected CSV Schema

The application expects a CSV file with the following columns:

| Column | Required | Description | Alternative Names |
|--------|----------|-------------|-------------------|
| product_id | Yes | Unique product identifier | product, asin, product_name, item_id |
| review_date | Yes | Date of the review | date, timestamp, review_time, created_at |
| review_text | Yes | Full text of the review | text, review, body, comment, review_body |
| rating | Optional | Star rating (1-5) | score, overall, stars, star_rating |

## Synthetic Demo Dataset

Run `python scripts/generate_demo_data.py` to generate `data/synthetic_reviews.csv`.

This synthetic dataset is clearly labelled and should NOT be presented as real customer data.

## Recommended Public Datasets

### Amazon Product Reviews
- **Source**: Available from various academic repositories
- **Required fields**: Must contain `asin` (product ID), `reviewText`, `overall` (rating), and `unixReviewTime` or `reviewTime`
- **Note**: Filter for electronics categories (e.g., Cell Phones & Accessories) for best results with the built-in aspect lexicon

## Data Preparation Guide

1. Ensure your CSV has the required columns (product_id, review_date, review_text)
2. Dates should be in a standard format (ISO 8601 preferred: YYYY-MM-DD)
3. Remove any personally identifiable information
4. The application handles missing ratings, duplicate removal, and date parsing automatically
