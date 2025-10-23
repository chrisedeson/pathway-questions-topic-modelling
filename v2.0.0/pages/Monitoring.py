"""
Streamlit page for Monitoring Dashboard
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_monitoring_data_from_s3
from utils.monitoring_visualizations import create_summary_metrics, create_time_series_charts

st.set_page_config(
    page_title="Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Pathway Chatbot Monitoring Dashboard")
st.markdown("Real-time monitoring and analytics for the Pathway Chatbot backend")

# Sidebar configuration
st.sidebar.header("Configuration")
days_back = st.sidebar.slider("Days to show", 1, 90, 7)

# Load data
df = load_monitoring_data_from_s3(days_back)

if df is None or df.empty:
    st.warning("No monitoring data found for the selected period.")
    st.stop()

st.success(f"✅ Loaded {len(df)} records from the last {days_back} days.")

# Display metrics and charts
create_summary_metrics(df)
create_time_series_charts(df)

# Raw data explorer
with st.expander("🔍 Raw Data Explorer"):
    st.dataframe(df, use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"monitoring_data.csv",
        mime="text/csv"
    )