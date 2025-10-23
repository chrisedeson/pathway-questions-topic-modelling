"""
Visualization components for the Monitoring Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict


def create_summary_metrics(df: pd.DataFrame):
    """Display summary metrics for the monitoring data."""
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

def create_time_series_charts(df: pd.DataFrame):
    """Display time series charts for memory, performance, and traffic."""
    st.header("📊 Time Series Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Memory", "Performance", "Traffic"])
    
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
        
    with tab2:
        st.subheader("Response Time Analysis")
        
        fig = px.scatter(
            df,
            x='timestamp',
            y='duration_seconds',
            color='status_code',
            title='Response Time by Request',
            labels={'duration_seconds': 'Duration (seconds)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.subheader("Traffic Patterns")
        
        df['hour'] = df['timestamp'].dt.floor('H')
        hourly_requests = df.groupby('hour').size().reset_index(name='count')
        
        fig = px.bar(
            hourly_requests,
            x='hour',
            y='count',
            title='Requests per Hour'
        )
        st.plotly_chart(fig, use_container_width=True)
