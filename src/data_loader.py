"""Data loading, validation, and preprocessing module."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import re
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLUMN_ALIASES


def auto_map_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Automatically map DataFrame columns to expected schema.
    
    Args:
        df: Input DataFrame with potentially non-standard column names.
        
    Returns:
        Dictionary mapping standard names to detected column names.
    """
    mapping = {}
    df_cols_lower = {col.lower().strip(): col for col in df.columns}
    
    for standard_name, aliases in COLUMN_ALIASES.items():
        matched = None
        for alias in aliases:
            if alias.lower() in df_cols_lower:
                matched = df_cols_lower[alias.lower()]
                break
        mapping[standard_name] = matched
    
    return mapping


def validate_and_load_csv(file_path_or_buffer, column_mapping: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict]:
    """Load and validate a CSV file for review analysis.
    
    Args:
        file_path_or_buffer: Path to CSV file or file-like object.
        column_mapping: Optional explicit column mapping dict.
        
    Returns:
        Tuple of (cleaned DataFrame, validation report dict).
    """
    report = {
        "total_rows_read": 0,
        "rows_retained": 0,
        "rows_rejected": 0,
        "duplicates_removed": 0,
        "missing_text_count": 0,
        "invalid_date_count": 0,
        "missing_rating_count": 0,
        "issues": [],
    }
    
    try:
        df = pd.read_csv(file_path_or_buffer, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            if hasattr(file_path_or_buffer, 'seek'):
                file_path_or_buffer.seek(0)
            df = pd.read_csv(file_path_or_buffer, encoding="latin-1")
        except Exception as e:
            report["issues"].append(f"Failed to read CSV: {str(e)}")
            return pd.DataFrame(), report
    except Exception as e:
        report["issues"].append(f"Failed to read CSV: {str(e)}")
        return pd.DataFrame(), report
    
    report["total_rows_read"] = len(df)
    
    if len(df) == 0:
        report["issues"].append("CSV file is empty.")
        return pd.DataFrame(), report
    
    # Auto-map columns if no explicit mapping provided
    if column_mapping is None:
        column_mapping = auto_map_columns(df)
    
    # Rename columns to standard names
    rename_map = {}
    for standard_name, original_name in column_mapping.items():
        if original_name is not None and original_name in df.columns:
            rename_map[original_name] = standard_name
    
    df = df.rename(columns=rename_map)
    
    # Check required columns
    if "product_id" not in df.columns:
        report["issues"].append("No product_id column found or mapped.")
        return pd.DataFrame(), report
    if "review_text" not in df.columns:
        report["issues"].append("No review_text column found or mapped.")
        return pd.DataFrame(), report
    if "review_date" not in df.columns:
        report["issues"].append("No review_date column found or mapped.")
        return pd.DataFrame(), report
    
    # Track original count
    initial_count = len(df)
    
    # Handle product_id
    df["product_id"] = df["product_id"].astype(str).str.strip()
    df = df[df["product_id"].notna() & (df["product_id"] != "") & (df["product_id"] != "nan")]
    
    # Handle review_text - remove rows with missing/empty text
    missing_text_mask = df["review_text"].isna() | (df["review_text"].astype(str).str.strip() == "")
    report["missing_text_count"] = int(missing_text_mask.sum())
    if report["missing_text_count"] > 0:
        report["issues"].append(f"{report['missing_text_count']} reviews had missing/empty text and were removed.")
    df = df[~missing_text_mask].copy()
    df["review_text"] = df["review_text"].astype(str).str.strip()
    
    # Parse dates robustly
    df["review_date"] = _parse_dates(df["review_date"])
    invalid_date_mask = df["review_date"].isna()
    report["invalid_date_count"] = int(invalid_date_mask.sum())
    if report["invalid_date_count"] > 0:
        report["issues"].append(f"{report['invalid_date_count']} reviews had invalid/unparseable dates and were removed.")
    df = df[~invalid_date_mask].copy()
    
    # Handle rating
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        missing_rating_mask = df["rating"].isna()
        report["missing_rating_count"] = int(missing_rating_mask.sum())
        if report["missing_rating_count"] > 0:
            report["issues"].append(f"{report['missing_rating_count']} reviews had missing/invalid ratings (kept with NaN rating).")
        # Clip ratings to 1-5 range if they exist
        df.loc[df["rating"].notna(), "rating"] = df.loc[df["rating"].notna(), "rating"].clip(1, 5)
    else:
        df["rating"] = np.nan
        report["issues"].append("No rating column found. Ratings will not be used.")
    
    # Deduplication policy:
    # 1. If review_id is present, remove duplicate records sharing the same (product_id, review_id), keeping the first.
    # 2. Also remove duplicate records sharing exact (product_id, review_date, review_text), keeping the first.
    dup_count = 0
    if "review_id" in df.columns:
        valid_id_mask = df["review_id"].notna() & (df["review_id"].astype(str).str.strip() != "")
        if valid_id_mask.any():
            id_dup_count = df[valid_id_mask].duplicated(subset=["product_id", "review_id"], keep="first").sum()
            if id_dup_count > 0:
                df = df[~df.index.isin(df[valid_id_mask][df[valid_id_mask].duplicated(subset=["product_id", "review_id"], keep="first")].index)]
                dup_count += int(id_dup_count)
                report["issues"].append(f"{id_dup_count} duplicate reviews removed based on review_id.")

    content_dup_count = df.duplicated(subset=["product_id", "review_date", "review_text"], keep="first").sum()
    if content_dup_count > 0:
        df = df.drop_duplicates(subset=["product_id", "review_date", "review_text"], keep="first")
        dup_count += int(content_dup_count)
        report["issues"].append(f"{content_dup_count} duplicate reviews removed based on exact content (product_id, date, text).")

    report["duplicates_removed"] = dup_count
    
    # Sort chronologically
    df = df.sort_values("review_date").reset_index(drop=True)
    
    report["rows_retained"] = len(df)
    report["rows_rejected"] = initial_count - len(df)
    
    return df, report


def _parse_dates(date_series: pd.Series) -> pd.Series:
    """Parse dates from various formats robustly.
    
    Handles: ISO format, US format, Unix timestamps, and various separators.
    
    Args:
        date_series: Series of date values in various formats.
        
    Returns:
        Series of parsed datetime values (NaT for unparseable).
    """
    result = pd.to_datetime(date_series, errors="coerce")
    
    # For remaining NaTs, try Unix timestamp (seconds)
    still_nat = result.isna() & date_series.notna()
    if still_nat.any():
        numeric_vals = pd.to_numeric(date_series[still_nat], errors="coerce")
        valid_numeric = numeric_vals.notna()
        if valid_numeric.any():
            # Heuristic: if value > 1e12, it's likely milliseconds
            millis_mask = numeric_vals > 1e12
            unix_dates = pd.Series(index=numeric_vals.index, dtype="datetime64[ns]")
            if millis_mask.any():
                unix_dates[millis_mask] = pd.to_datetime(numeric_vals[millis_mask], unit="ms", errors="coerce")
            if (~millis_mask & valid_numeric).any():
                unix_dates[~millis_mask & valid_numeric] = pd.to_datetime(
                    numeric_vals[~millis_mask & valid_numeric], unit="s", errors="coerce"
                )
            result.update(unix_dates.dropna())
    
    return result


def get_products(df: pd.DataFrame) -> List[str]:
    """Get sorted list of unique product IDs."""
    if df.empty or "product_id" not in df.columns:
        return []
    return sorted(df["product_id"].unique().tolist())


def get_date_range(df: pd.DataFrame) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Get the date range of the dataset."""
    if df.empty or "review_date" not in df.columns:
        return None, None
    return df["review_date"].min(), df["review_date"].max()


def filter_by_product(df: pd.DataFrame, product_id: str) -> pd.DataFrame:
    """Filter dataset to a specific product."""
    return df[df["product_id"] == product_id].copy()


def aggregate_by_period(df: pd.DataFrame, period: str = "weekly") -> pd.DataFrame:
    """Add a period column based on aggregation choice.
    
    Args:
        df: DataFrame with review_date column.
        period: 'weekly' or 'monthly'.
        
    Returns:
        DataFrame with added 'period' column.
    """
    df = df.copy()
    if period == "weekly":
        df["period"] = df["review_date"].dt.to_period("W").apply(lambda x: x.start_time)
    else:
        df["period"] = df["review_date"].dt.to_period("M").apply(lambda x: x.start_time)
    return df
