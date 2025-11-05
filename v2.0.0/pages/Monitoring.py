"""
Streamlit page for Monitoring Dashboard with Crash Diagnosis
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_monitoring_data_from_s3
from utils.monitoring_visualizations import (
    calculate_health_score,
    create_health_dashboard,
    create_crash_analysis,
    create_memory_leak_detector,
    create_summary_metrics,
    create_time_series_charts,
    create_system_diagnostics
)

st.set_page_config(
    page_title="Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Pathway Chatbot Monitoring Dashboard")
st.markdown("Real-time monitoring and analytics for the Pathway Chatbot backend")

# Sidebar configuration
st.sidebar.header("Configuration")
days_back = st.sidebar.slider("Days to show", 1, 90, 30, help="Number of days of monitoring data to load. Increase if you don't see data.")

# Add "Load All" checkbox
load_all = st.sidebar.checkbox("Load ALL available data", value=False, help="Ignore date filter and load everything")
if load_all:
    days_back = 9999  # Special value to load everything

# Load data
with st.spinner("Loading monitoring data from S3..."):
    df = load_monitoring_data_from_s3(days_back)

if df is None or df.empty:
    st.warning("No monitoring data found for the selected period.")
    st.stop()

st.success(f"✅ Loaded {len(df)} records from the last {days_back} days.")

# Calculate health score
health_score = calculate_health_score(df)

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏥 Health Overview",
    "🚨 Crash Analysis", 
    "💾 Memory & Leaks",
    "⚡ Performance",
    "📊 System Diagnostics",
    "🔍 Raw Data"
])

with tab1:
    create_health_dashboard(df, health_score)

with tab2:
    create_crash_analysis(df)

with tab3:
    create_memory_leak_detector(df)
    st.markdown("---")
    create_summary_metrics(df)

with tab4:
    create_time_series_charts(df)

with tab5:
    create_system_diagnostics(df)

with tab6:
    st.header("🔍 Raw Data Explorer")
    st.markdown("Complete monitoring data for technical analysis")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        show_errors_only = st.checkbox("Show errors only (4xx/5xx)")
    with col2:
        endpoint_filter = st.selectbox(
            "Filter by endpoint",
            ["All"] + sorted(df['endpoint'].unique().tolist())
        )
    
    # Apply filters
    filtered_df = df.copy()
    if show_errors_only:
        filtered_df = filtered_df[filtered_df['status_code'] >= 400]
    if endpoint_filter != "All":
        filtered_df = filtered_df[filtered_df['endpoint'] == endpoint_filter]
    
    st.dataframe(filtered_df, use_container_width=True, height=400)
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"monitoring_data_{days_back}days.csv",
        mime="text/csv"
    )
    
    # Data summary
    st.markdown("### 📋 Data Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(filtered_df))
    with col2:
        st.metric("Columns", len(filtered_df.columns))
    with col3:
        st.metric("Date Range", f"{(filtered_df['timestamp'].max() - filtered_df['timestamp'].min()).days} days")
    with col4:
        st.metric("Data Size", f"{filtered_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
