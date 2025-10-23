"""
Streamlit Dashboard for Pathway Chatbot Monitoring
Reads Parquet files from S3 and displays comprehensive metrics.

Installation:
pip install streamlit pandas pyarrow boto3 plotly

Run:
streamlit run streamlit_monitoring_dashboard.py
"""

import streamlit as st
import pandas as pd
import boto3
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import os

# Page configuration
st.set_page_config(
    page_title="Pathway Chatbot Monitoring",
    page_icon="📊",
    layout="wide"
)

# Configuration
S3_BUCKET = os.getenv("MONITORING_S3_BUCKET", "pathway-chatbot-monitoring")
S3_PREFIX = os.getenv("MONITORING_S3_PREFIX", "metrics")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

@st.cache_resource
def get_s3_client():
    """Initialize S3 client."""
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=AWS_REGION
    )

@st.cache_data(ttl=300)  # Cache for 5 minutes
def list_reports(s3_client, days_back=30):
    """List available Parquet reports from S3."""
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=f"{S3_PREFIX}/metrics_"
        )
        
        if 'Contents' not in response:
            return []
        
        reports = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('.parquet'):
                # Parse date from filename
                try:
                    filename = key.split('/')[-1]
                    date_str = filename.replace('metrics_', '').replace('.parquet', '').split('_')[0]
                    report_date = datetime.strptime(date_str, '%Y%m%d')
                    
                    if report_date >= cutoff_date:
                        reports.append({
                            'key': key,
                            'date': report_date,
                            'size': obj['Size'],
                            'last_modified': obj['LastModified']
                        })
                except:
                    continue
        
        return sorted(reports, key=lambda x: x['date'], reverse=True)
    
    except Exception as e:
        st.error(f"Error listing reports: {e}")
        return []

@st.cache_data(ttl=300)
def load_report(s3_client, key):
    """Load a Parquet report from S3."""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        df = pd.read_parquet(BytesIO(response['Body'].read()))
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    except Exception as e:
        st.error(f"Error loading report: {e}")
        return None

def main():
    st.title("📊 Pathway Chatbot Monitoring Dashboard")
    st.markdown("Real-time monitoring and analytics for the Pathway Chatbot backend")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    days_back = st.sidebar.slider("Days to show", 1, 90, 30)
    
    # Initialize S3 client
    try:
        s3_client = get_s3_client()
    except Exception as e:
        st.error(f"Failed to connect to S3: {e}")
        st.info("Please set AWS credentials in environment variables")
        return
    
    # List available reports
    reports = list_reports(s3_client, days_back)
    
    if not reports:
        st.warning("No monitoring reports found")
        st.info(f"Looking for reports in: s3://{S3_BUCKET}/{S3_PREFIX}/")
        return
    
    st.sidebar.success(f"Found {len(reports)} reports")
    
    # Date range selector
    selected_dates = st.sidebar.multiselect(
        "Select dates to analyze",
        options=[r['date'].strftime('%Y-%m-%d') for r in reports],
        default=[reports[0]['date'].strftime('%Y-%m-%d')]
    )
    
    if not selected_dates:
        st.info("Please select at least one date to analyze")
        return
    
    # Load selected reports
    dfs = []
    for date_str in selected_dates:
        matching_reports = [r for r in reports if r['date'].strftime('%Y-%m-%d') == date_str]
        for report in matching_reports:
            df = load_report(s3_client, report['key'])
            if df is not None:
                dfs.append(df)
    
    if not dfs:
        st.error("Failed to load any reports")
        return
    
    # Combine all dataframes
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('timestamp')
    
    # Summary metrics
    st.header("📈 Summary Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Requests", f"{len(df):,}")
    
    with col2:
        error_count = len(df[df['status_code'] >= 400])
        error_rate = (error_count / len(df) * 100) if len(df) > 0 else 0
        st.metric("Errors", error_count, f"{error_rate:.2f}%")
    
    with col3:
        avg_duration = df['duration_seconds'].mean()
        st.metric("Avg Response Time", f"{avg_duration:.2f}s")
    
    with col4:
        avg_memory = df['memory_rss_mb'].mean()
        st.metric("Avg Memory", f"{avg_memory:.0f} MB")
    
    with col5:
        max_memory = df['memory_rss_mb'].max()
        st.metric("Peak Memory", f"{max_memory:.0f} MB")
    
    # Time series charts
    st.header("📊 Time Series Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Memory", "Performance", "Traffic", "Security"])
    
    with tab1:
        st.subheader("Memory Usage Over Time")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['memory_rss_mb'],
            mode='lines',
            name='Memory RSS (MB)',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['system_memory_available_mb'],
            mode='lines',
            name='System Available (MB)',
            line=dict(color='green', dash='dash')
        ))
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Memory (MB)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Memory distribution
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='memory_rss_mb', nbins=50, title='Memory Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, y='memory_rss_mb', title='Memory Statistics')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Response Time Analysis")
        
        # Response time over time
        fig = px.scatter(
            df,
            x='timestamp',
            y='duration_seconds',
            color='status_code',
            title='Response Time by Request',
            labels={'duration_seconds': 'Duration (seconds)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # CPU usage
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['cpu_percent'],
            mode='lines',
            name='CPU %',
            line=dict(color='orange')
        ))
        fig.update_layout(
            title='CPU Usage Over Time',
            xaxis_title='Time',
            yaxis_title='CPU %'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Response time percentiles
        percentiles = df['duration_seconds'].quantile([0.5, 0.75, 0.9, 0.95, 0.99])
        st.write("**Response Time Percentiles:**")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("P50", f"{percentiles[0.5]:.2f}s")
        col2.metric("P75", f"{percentiles[0.75]:.2f}s")
        col3.metric("P90", f"{percentiles[0.9]:.2f}s")
        col4.metric("P95", f"{percentiles[0.95]:.2f}s")
        col5.metric("P99", f"{percentiles[0.99]:.2f}s")
    
    with tab3:
        st.subheader("Traffic Patterns")
        
        # Requests over time
        df['hour'] = df['timestamp'].dt.floor('H')
        hourly_requests = df.groupby('hour').size().reset_index(name='count')
        
        fig = px.bar(
            hourly_requests,
            x='hour',
            y='count',
            title='Requests per Hour'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Endpoint distribution
        endpoint_counts = df['endpoint'].value_counts().head(10)
        fig = px.bar(
            x=endpoint_counts.values,
            y=endpoint_counts.index,
            orientation='h',
            title='Top 10 Endpoints',
            labels={'x': 'Request Count', 'y': 'Endpoint'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Status code distribution
        status_counts = df['status_code'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title='Status Code Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Security Monitoring")
        
        # Check if security columns exist
        if 'security_blocked' in df.columns:
            blocked_requests = df[df['security_blocked'] == True]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Blocked Requests", len(blocked_requests))
            with col2:
                block_rate = (len(blocked_requests) / len(df) * 100) if len(df) > 0 else 0
                st.metric("Block Rate", f"{block_rate:.2f}%")
            
            if len(blocked_requests) > 0:
                # Security events over time
                blocked_requests['hour'] = blocked_requests['timestamp'].dt.floor('H')
                security_timeline = blocked_requests.groupby('hour').size().reset_index(name='count')
                
                fig = px.line(
                    security_timeline,
                    x='hour',
                    y='count',
                    title='Security Blocks Over Time',
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk level distribution
                if 'risk_level' in blocked_requests.columns:
                    risk_counts = blocked_requests['risk_level'].value_counts()
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title='Risk Level Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No security data available in selected reports")
    
    # Raw data explorer
    with st.expander("🔍 Raw Data Explorer"):
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"monitoring_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # System health indicators
    st.header("🏥 System Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Memory health
        latest_memory = df.iloc[-1]['memory_rss_mb']
        memory_threshold = 2000  # 2GB
        memory_health = "🟢 Healthy" if latest_memory < memory_threshold * 0.7 else "🟡 Warning" if latest_memory < memory_threshold * 0.9 else "🔴 Critical"
        st.metric("Memory Health", memory_health, f"{latest_memory:.0f} MB")
    
    with col2:
        # Error rate health
        recent_errors = len(df.tail(100)[df.tail(100)['status_code'] >= 400])
        error_health = "🟢 Healthy" if recent_errors < 5 else "🟡 Warning" if recent_errors < 15 else "🔴 Critical"
        st.metric("Error Rate Health", error_health, f"{recent_errors}/100")
    
    with col3:
        # Response time health
        recent_avg_time = df.tail(100)['duration_seconds'].mean()
        time_health = "🟢 Healthy" if recent_avg_time < 2 else "🟡 Warning" if recent_avg_time < 5 else "🔴 Critical"
        st.metric("Response Time Health", time_health, f"{recent_avg_time:.2f}s")

if __name__ == "__main__":
    main()
