"""
Utility functions for data loading and processing
"""

import streamlit as st
import pandas as pd
import boto3
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
from io import BytesIO
import os


@st.cache_data(ttl=3600, show_spinner="Loading data from S3...")
def load_data_from_s3() -> Dict[str, pd.DataFrame]:
    """
    Load all parquet files from S3.
    Returns a dictionary of DataFrames.
    """
    # Import config values here (after Streamlit has initialized)
    from config import (
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET, 
        AWS_S3_PREFIX, AWS_REGION, FILE_PATTERNS
    )
    
    try:
        # Debug: Check if credentials are loaded
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            st.error(f"❌ AWS credentials not loaded. Access Key: {'SET' if AWS_ACCESS_KEY_ID else 'EMPTY'}, Secret Key: {'SET' if AWS_SECRET_ACCESS_KEY else 'EMPTY'}")
            return {}
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        
        data = {}
        
        # List all objects in the prefix
        response = s3_client.list_objects_v2(
            Bucket=AWS_S3_BUCKET,
            Prefix=AWS_S3_PREFIX + "/"
        )
        
        if 'Contents' not in response:
            st.warning("No files found in S3 bucket.")
            return data
        
        # Get the most recent file for each pattern
        file_groups = {key: [] for key in FILE_PATTERNS.keys()}
        
        for obj in response['Contents']:
            key = obj['Key']
            filename = key.split('/')[-1]
            
            # Match files to patterns
            for pattern_key, pattern in FILE_PATTERNS.items():
                # Extract the base name and extension from pattern
                # e.g., "similar_questions_*.parquet" -> base="similar_questions_", ext=".parquet"
                parts = pattern.split('*')
                if len(parts) == 2:
                    prefix = parts[0]  # "similar_questions_"
                    suffix = parts[1]  # ".parquet"
                    
                    # Check if filename matches pattern
                    if filename.startswith(prefix) and filename.endswith(suffix):
                        file_groups[pattern_key].append({
                            'key': key,
                            'last_modified': obj['LastModified'],
                            'size': obj['Size']
                        })
        
        # Load the most recent file for each type
        for file_type, files in file_groups.items():
            if not files:
                continue
                
            # Sort by last_modified descending
            files.sort(key=lambda x: x['last_modified'], reverse=True)
            most_recent = files[0]
            
            # Skip error_log JSON files for now
            if file_type == "error_log":
                continue
            
            try:
                # Download and read parquet file
                # Must read into BytesIO buffer because pandas needs seekable stream
                obj = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=most_recent['key'])
                buffer = BytesIO(obj['Body'].read())
                df = pd.read_parquet(buffer)
                data[file_type] = df
                
            except Exception as e:
                st.error(f"Error loading {file_type}: {str(e)}")
        
        return data
        
    except Exception as e:
        st.error(f"Error connecting to S3: {str(e)}")
        return {}


@st.cache_data(ttl=3600)
def get_latest_file_info() -> Dict[str, any]:
    """
    Get information about the latest files in S3.
    """
    # Import config values here (after Streamlit has initialized)
    from config import (
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET, 
        AWS_S3_PREFIX, AWS_REGION
    )
    
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        
        response = s3_client.list_objects_v2(
            Bucket=AWS_S3_BUCKET,
            Prefix=AWS_S3_PREFIX + "/"
        )
        
        if 'Contents' not in response:
            return {}
        
        # Get the most recent file
        files = response['Contents']
        files.sort(key=lambda x: x['LastModified'], reverse=True)
        
        if files:
            latest = files[0]
            return {
                'last_updated': latest['LastModified'],
                'file_count': len(files),
                'total_size_mb': sum(f['Size'] for f in files) / (1024 * 1024)
            }
        
        return {}
        
    except Exception as e:
        st.error(f"Error getting file info: {str(e)}")
        return {}


def merge_data_for_dashboard(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge data for the main dashboard view.
    Primary source is pathway_questions_review which has ALL questions.
    """
    # Use pathway_questions_review as primary source (has all questions)
    if 'pathway_questions_review' in data and not data['pathway_questions_review'].empty:
        df = data['pathway_questions_review'].copy()
        
        # Rename classification values to match dashboard expectations
        classification_map = {
            'existing': 'Existing Topic',
            'new': 'New Topic',
            'uncategorized': 'Uncategorized'
        }
        df['classification'] = df['classification'].map(classification_map)
        
        # For consistency, rename topic_name to matched_topic for existing topics
        df['matched_topic'] = df['topic_name']
        
        # Add similarity score from similar_questions if available
        if 'similar_questions' in data:
            similar_df = data['similar_questions'][['question', 'similarity_score']].copy()
            df = df.merge(similar_df, on='question', how='left')
        else:
            # Use confidence as similarity for existing topics
            df['similarity_score'] = df.apply(
                lambda row: row['confidence'] if row['classification'] == 'Existing Topic' else None,
                axis=1
            )
        
        # Ensure timestamp is datetime if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        return df
    
    # Fallback to old logic if pathway_questions_review doesn't exist
    dfs = []
    
    # Process similar questions (existing topics)
    if 'similar_questions' in data and not data['similar_questions'].empty:
        df_similar = data['similar_questions'].copy()
        df_similar['classification'] = 'Existing Topic'
        df_similar['matched_topic'] = df_similar['existing_topic']
        dfs.append(df_similar)
    
    # Process new topics
    if 'new_topics' in data and not data['new_topics'].empty:
        df_new = data['new_topics'].copy()
        df_new['classification'] = 'New Topic'
        # For new topics, matched_topic and matched_subtopic should be null
        if 'matched_topic' not in df_new.columns:
            df_new['matched_topic'] = None
        dfs.append(df_new)
    
    if not dfs:
        return pd.DataFrame()
    
    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Ensure timestamp is datetime
    if 'timestamp' in merged_df.columns:
        merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], errors='coerce')
    
    # Sort by timestamp descending (newest first)
    if 'timestamp' in merged_df.columns:
        merged_df = merged_df.sort_values('timestamp', ascending=False)
    
    return merged_df


def calculate_kpis(merged_df: pd.DataFrame, data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
    """
    Calculate Key Performance Indicators for the dashboard.
    """
    kpis = {
        'total_questions': len(merged_df),
        'matched_existing': len(merged_df[merged_df['classification'] == 'Existing Topic']),
        'new_topics_discovered': 0,
        'questions_in_new_topics': len(merged_df[merged_df['classification'] == 'New Topic']),
        'countries': merged_df['country'].nunique() if 'country' in merged_df.columns else 0,
        'avg_similarity': merged_df['similarity_score'].mean() if 'similarity_score' in merged_df.columns else 0,
        'last_updated': None
    }
    
    # Count unique new topics
    # new_topics DataFrame has columns: topic_name, representative_question, question_count
    # Each row represents one topic cluster
    if 'new_topics' in data and not data['new_topics'].empty:
        kpis['new_topics_discovered'] = len(data['new_topics'])
    
    # Alternative: Count unique topic names in merged_df for new topics
    if kpis['new_topics_discovered'] == 0:
        new_topic_df = merged_df[merged_df['classification'] == 'New Topic']
        if 'matched_topic' in new_topic_df.columns:
            kpis['new_topics_discovered'] = new_topic_df['matched_topic'].nunique()
    
    # Get last updated timestamp
    file_info = get_latest_file_info()
    if file_info and 'last_updated' in file_info:
        kpis['last_updated'] = file_info['last_updated']
    
    return kpis


def filter_dataframe(
    df: pd.DataFrame,
    classification: str = "All",
    date_range: Optional[Tuple[datetime, datetime]] = None,
    countries: Optional[List[str]] = None,
    search_query: Optional[str] = None,
    min_similarity: Optional[float] = None
) -> pd.DataFrame:
    """
    Apply filters to the dataframe.
    All filtering is done in-memory on cached data for instant results.
    """
    filtered_df = df.copy()
    
    # Classification filter
    if classification != "All":
        filtered_df = filtered_df[filtered_df['classification'] == classification]
    
    # Date range filter
    if date_range and 'timestamp' in filtered_df.columns:
        start_date, end_date = date_range
        # Convert to timezone-aware timestamps if the column is timezone-aware
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        if filtered_df['timestamp'].dtype.tz is not None:
            start_ts = start_ts.tz_localize('UTC')
            end_ts = end_ts.tz_localize('UTC')
        filtered_df = filtered_df[
            (filtered_df['timestamp'] >= start_ts) &
            (filtered_df['timestamp'] <= end_ts)
        ]
    
    # Country filter
    if countries and 'country' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['country'].isin(countries)]
    
    # Search filter (searches in question text)
    if search_query and 'input' in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df['input'].str.contains(search_query, case=False, na=False)
        ]
    
    # Similarity filter
    if min_similarity is not None and 'similarity_score' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['similarity_score'] >= min_similarity]
    
    return filtered_df


def sort_dataframe(df: pd.DataFrame, sort_by: str, ascending: bool = True) -> pd.DataFrame:
    """
    Sort dataframe by specified column.
    """
    if sort_by in df.columns:
        return df.sort_values(sort_by, ascending=ascending)
    return df


def export_to_csv(df: pd.DataFrame) -> bytes:
    """
    Export dataframe to CSV format for download.
    """
    return df.to_csv(index=False).encode('utf-8')


def format_timestamp(ts: pd.Timestamp) -> str:
    """
    Format timestamp for display.
    """
    if pd.isna(ts):
        return "N/A"
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def get_column_config(columns: List[str]) -> Dict[str, any]:
    """
    Get Streamlit column configuration for dataframe display.
    """
    from config import COLUMN_DISPLAY_NAMES
    
    config = {}
    for col in columns:
        if col in COLUMN_DISPLAY_NAMES:
            config[col] = st.column_config.Column(
                label=COLUMN_DISPLAY_NAMES[col],
                width="medium"
            )
            
            # Special configurations
            if col == "timestamp":
                config[col] = st.column_config.DatetimeColumn(
                    label="Timestamp",
                    format="YYYY-MM-DD HH:mm:ss"
                )
            elif col == "similarity_score":
                config[col] = st.column_config.NumberColumn(
                    label="Similarity Score",
                    format="%.3f"
                )
            elif col == "input":
                config[col] = st.column_config.TextColumn(
                    label="Question",
                    width="large"
                )
    
    return config


def ensure_data_loaded():
    """
    Ensure data is loaded into session state.
    Call this at the start of every page to handle page refreshes.
    """
    if 'merged_df' not in st.session_state or 'raw_data' not in st.session_state or 'kpis' not in st.session_state:
        with st.spinner("🔄 Loading data from AWS S3..."):
            data = load_data_from_s3()
        
        if not data:
            st.error("❌ **No data available.** Please ensure the notebook has uploaded files to S3.")
            st.info("""
            **💡 Tip:** Run the Jupyter notebook to process questions and upload results to S3.
            The dashboard will automatically load the most recent data.
            """)
            st.stop()
        
        # Merge data for dashboard
        merged_df = merge_data_for_dashboard(data)
        
        if merged_df.empty:
            st.warning("⚠️ No question data available in the loaded files.")
            st.stop()
        
        # Calculate KPIs
        kpis = calculate_kpis(merged_df, data)
        
        # Store in session state
        st.session_state['merged_df'] = merged_df
        st.session_state['raw_data'] = data
        st.session_state['kpis'] = kpis
