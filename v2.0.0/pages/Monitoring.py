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
    create_system_diagnostics,
    create_data_completeness_check,
    create_emergency_alert_banner,
    create_heartbeat_timeline
)

st.set_page_config(
    page_title="Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Pathway Chatbot Monitoring Dashboard")
st.markdown("Real-time monitoring and analytics for the Pathway Chatbot backend")

# Load S3 config for emergency alerts
try:
    from config import (
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, 
        MONITORING_S3_BUCKET, MONITORING_S3_PREFIX, AWS_REGION
    )
    import boto3
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    
    # Display emergency alerts if any exist
    alert_count = create_emergency_alert_banner(
        s3_client=s3_client,
        bucket=MONITORING_S3_BUCKET,
        prefix=MONITORING_S3_PREFIX
    )
    
    if alert_count > 0:
        st.markdown("---")
except Exception as e:
    # Silently skip if S3 not configured
    pass

# Sidebar configuration
st.sidebar.header("Configuration")
days_back = st.sidebar.slider("Days to show", 1, 90, 30, help="Number of days of monitoring data to load. Increase if you don't see data.")

# Add "Load All" checkbox
load_all = st.sidebar.checkbox("Load ALL available data", value=False, help="Ignore date filter and load everything")
if load_all:
    days_back = 9999  # Special value to load everything

# Timezone selector
st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Timezone")
timezone_option = st.sidebar.radio(
    "Display timestamps in:",
    options=["PST", "UTC", "Browser"],
    index=0,
    help="PST: Pacific Standard Time\nUTC: Coordinated Universal Time\nBrowser: Your local timezone"
)

# Store timezone preference in session state
if 'timezone_preference' not in st.session_state or st.session_state.timezone_preference != timezone_option:
    st.session_state.timezone_preference = timezone_option

st.sidebar.markdown("---")

# Load data
with st.spinner("Loading monitoring data from S3..."):
    df, alert_events = load_monitoring_data_from_s3(days_back)

if df is None or df.empty:
    st.warning("No monitoring data found for the selected period.")
    st.stop()

# Convert timestamps to selected timezone
if 'timestamp' in df.columns:
    from utils.timezone_utils import convert_series_to_timezone, get_timezone_label
    df['timestamp'] = convert_series_to_timezone(df['timestamp'], st.session_state.timezone_preference)
    timezone_label = get_timezone_label(st.session_state.timezone_preference)
    st.success(f"✅ Loaded {len(df)} records from the last {days_back} days. Timestamps shown in **{timezone_label}**.")
else:
    st.success(f"✅ Loaded {len(df)} records from the last {days_back} days.")

# Display emergency alerts banner
if alert_events:
    # Categorize all alert events
    emergency_alerts = [a for a in alert_events if a.get('event_type') in ['alert', 'emergency']]
    crash_boots = [a for a in alert_events if a.get('event_type') == 'boot' and a.get('restart_type') == 'crash_recovery']
    normal_boots = [a for a in alert_events if a.get('event_type') == 'boot' and a.get('restart_type') == 'clean_start']
    heartbeats = [a for a in alert_events if a.get('event_type') == 'heartbeat' or a.get('type') == 'heartbeat']
    
    # Show summary with breakdown
    total_events = len(alert_events)
    st.info(f"📋 **Found {total_events} alert events:** "
            f"{len(crash_boots)} crash recovery boots, "
            f"{len(normal_boots)} normal boots, "
            f"{len(emergency_alerts)} alerts/emergencies, "
            f"{len(heartbeats)} heartbeats")
    
    if emergency_alerts or crash_boots:
        st.error(f"🚨 **{len(emergency_alerts + crash_boots)} Critical Events Detected!**")
        
        # Show details in expanders
        for alert in sorted(emergency_alerts + crash_boots, key=lambda x: x.get('timestamp', ''), reverse=True)[:5]:
            timestamp = alert.get('timestamp', 'Unknown time')
            message = alert.get('message', 'Alert')
            severity = alert.get('severity', 'warning')
            
            severity_icon = '🔴' if severity == 'critical' else '⚠️' if severity == 'error' else '🟡'
            
            with st.expander(f"{severity_icon} {message} - {timestamp}"):
                st.json(alert)
        
        st.markdown("---")
    
    # Show normal boots in an info section (not critical)
    if normal_boots:
        with st.expander(f"ℹ️ Normal System Boots ({len(normal_boots)}) - Click to view"):
            st.info("These are normal deployments or restarts (not crashes)")
            for boot in sorted(normal_boots, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]:
                timestamp = boot.get('timestamp', 'Unknown time')
                files_found = boot.get('unsaved_files_found', 0)
                st.markdown(f"- 🔵 **{timestamp}** - Clean start ({files_found} leftover files)")

# Event Timeline moved to Crash Analysis tab for better interactivity
# (includes search, filtering, and full event history)

# Calculate health score
health_score = calculate_health_score(df)

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏥 Health Overview",
    "🚨 Crash Analysis", 
    "💾 Memory & Leaks",
    "⚡ Performance",
    "📊 System Diagnostics",
    "📋 Data Quality",
    "🔍 Raw Data"
])

with tab1:
    create_health_dashboard(df, health_score)

with tab2:
    create_crash_analysis(df, alert_events)

with tab3:
    create_memory_leak_detector(df, alert_events)
    st.markdown("---")
    create_summary_metrics(df)

with tab4:
    create_time_series_charts(df)

with tab5:
    create_system_diagnostics(df, alert_events)

with tab6:
    st.header("📋 Data Quality Checks")
    st.markdown("*Monitoring data completeness and service health validation*")
    
    # Create mini-tabs for different quality checks
    quality_tab1, quality_tab2 = st.tabs([
        "📊 Data Completeness",
        "💓 Heartbeat Monitoring"
    ])
    
    with quality_tab1:
        create_data_completeness_check(df)
    
    with quality_tab2:
        create_heartbeat_timeline(alert_events)

with tab7:
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
