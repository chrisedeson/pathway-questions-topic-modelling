"""
Timezone conversion utilities for the monitoring dashboard.
Supports UTC, PST/PDT, and browser timezone conversions.
"""

import pandas as pd
from datetime import datetime, timezone
from typing import Literal, Optional
import pytz


# Timezone definitions
UTC = pytz.UTC
PST = pytz.timezone('America/Los_Angeles')  # Handles PST/PDT automatically


def convert_timestamp_to_timezone(
    timestamp: pd.Timestamp,
    target_tz: Literal['UTC', 'PST', 'Browser'] = 'PST'
) -> pd.Timestamp:
    """
    Convert a timestamp to the specified timezone.
    
    Args:
        timestamp: Input timestamp (assumed UTC if naive)
        target_tz: Target timezone ('UTC', 'PST', or 'Browser')
    
    Returns:
        Timestamp in target timezone
    """
    if pd.isna(timestamp):
        return timestamp
    
    # Ensure timestamp is timezone-aware (assume UTC if naive)
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize('UTC')
    
    if target_tz == 'UTC':
        return timestamp.tz_convert('UTC')
    elif target_tz == 'PST':
        return timestamp.tz_convert(PST)
    elif target_tz == 'Browser':
        # For browser timezone, return UTC and let JavaScript handle conversion
        # (Streamlit doesn't have access to browser timezone in Python)
        return timestamp.tz_convert('UTC')
    else:
        return timestamp


def convert_series_to_timezone(
    series: pd.Series,
    target_tz: Literal['UTC', 'PST', 'Browser'] = 'PST'
) -> pd.Series:
    """
    Convert a pandas Series of timestamps to the specified timezone.
    
    Args:
        series: Series of timestamps
        target_tz: Target timezone ('UTC', 'PST', or 'Browser')
    
    Returns:
        Series with converted timestamps
    """
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, errors='coerce')
    
    # Localize to UTC if naive
    if series.dt.tz is None:
        series = series.dt.tz_localize('UTC')
    
    if target_tz == 'UTC':
        return series.dt.tz_convert('UTC')
    elif target_tz == 'PST':
        return series.dt.tz_convert(PST)
    elif target_tz == 'Browser':
        # Return UTC for browser (will be converted client-side)
        return series.dt.tz_convert('UTC')
    else:
        return series


def get_timezone_label(tz: Literal['UTC', 'PST', 'Browser'] = 'PST') -> str:
    """
    Get a human-readable label for the timezone.
    
    Args:
        tz: Timezone identifier
    
    Returns:
        Human-readable label
    """
    if tz == 'UTC':
        return 'UTC'
    elif tz == 'PST':
        # Check if currently in PST or PDT
        now = datetime.now(PST)
        tz_name = now.tzname()  # Returns 'PST' or 'PDT'
        return tz_name
    elif tz == 'Browser':
        return 'Browser Time'
    else:
        return 'UTC'


def format_timestamp(
    timestamp: pd.Timestamp,
    target_tz: Literal['UTC', 'PST', 'Browser'] = 'PST',
    include_tz: bool = True
) -> str:
    """
    Format a timestamp for display with timezone.
    
    Args:
        timestamp: Timestamp to format
        target_tz: Target timezone
        include_tz: Whether to include timezone in output
    
    Returns:
        Formatted timestamp string
    """
    if pd.isna(timestamp):
        return "N/A"
    
    # Convert to target timezone
    converted = convert_timestamp_to_timezone(timestamp, target_tz)
    
    # Format
    if include_tz:
        tz_label = get_timezone_label(target_tz)
        return f"{converted.strftime('%Y-%m-%d %H:%M:%S')} {tz_label}"
    else:
        return converted.strftime('%Y-%m-%d %H:%M:%S')


def get_current_time(target_tz: Literal['UTC', 'PST'] = 'PST') -> datetime:
    """
    Get current time in specified timezone.
    
    Args:
        target_tz: Target timezone
    
    Returns:
        Current datetime in target timezone
    """
    if target_tz == 'UTC':
        return datetime.now(UTC)
    elif target_tz == 'PST':
        return datetime.now(PST)
    else:
        return datetime.now(UTC)
