"""
Visualization components for the Monitoring Dashboard with Crash Diagnosis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict
import numpy as np


def calculate_health_score(df: pd.DataFrame) -> int:
    """
    Calculate a simple 0-100 health score that even a child can understand.
    
    Deductions:
    - Crashes (5xx errors): -5 points each (max -50)
    - High memory usage: -30 points if >80%, -15 if >60%
    - Slow responses: -20 points if >5s avg, -10 if >2s avg
    - Client errors (4xx): -2 points each (max -20)
    """
    score = 100
    
    # Deduct for server errors (crashes)
    crashes = len(df[df['status_code'] >= 500])
    score -= min(crashes * 5, 50)
    
    # Deduct for high memory usage
    if 'memory_percent' in df.columns:
        avg_mem_pct = df['memory_percent'].mean()
        if avg_mem_pct > 80:
            score -= 30
        elif avg_mem_pct > 60:
            score -= 15
    
    # Deduct for slow responses
    avg_duration = df['duration_seconds'].mean()
    if avg_duration > 5:
        score -= 20
    elif avg_duration > 2:
        score -= 10
    
    # Deduct for client errors
    client_errors = len(df[(df['status_code'] >= 400) & (df['status_code'] < 500)])
    score -= min(client_errors * 2, 20)
    
    return max(score, 0)


def create_health_dashboard(df: pd.DataFrame, health_score: int):
    """Display an easy-to-understand health dashboard."""
    st.header("🏥 System Health Overview")
    st.markdown("*Understanding your system's health at a glance*")
    
    # Health score with color coding
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if health_score >= 80:
            st.success(f"## 🟢 System Health: {health_score}/100 - Excellent!")
            st.markdown("✨ **Your chatbot is running great!** No major issues detected.")
        elif health_score >= 60:
            st.warning(f"## 🟡 System Health: {health_score}/100 - Needs Attention")
            st.markdown("⚠️ **Some issues detected.** Check the tabs below for details.")
        else:
            st.error(f"## 🔴 System Health: {health_score}/100 - CRITICAL!")
            st.markdown("🚨 **Serious problems detected!** Immediate action needed.")
    
    with col2:
        # Quick stats
        st.metric(
            "Total Requests",
            f"{len(df):,}",
            help="How many times people used the chatbot"
        )
        st.metric(
            "Success Rate",
            f"{(len(df[df['status_code'] < 400]) / len(df) * 100):.1f}%",
            help="Percentage of requests that worked correctly"
        )
    
    with col3:
        crashes = len(df[df['status_code'] >= 500])
        st.metric(
            "Crashes 🔴",
            crashes,
            help="Number of times the system completely failed"
        )
        warnings = len(df[(df['status_code'] >= 400) & (df['status_code'] < 500)])
        st.metric(
            "Warnings 🟡",
            warnings,
            help="Number of requests that had problems"
        )
    
    # Visual health indicators
    st.markdown("---")
    st.markdown("### 📊 Key Health Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_memory = df['memory_rss_mb'].mean()
        memory_status = "🟢" if avg_memory < 500 else "🟡" if avg_memory < 800 else "🔴"
        st.metric(
            f"{memory_status} Memory Used",
            f"{avg_memory:.0f} MB",
            help="How much computer memory the chatbot is using. Lower is better!"
        )
    
    with col2:
        avg_response = df['duration_seconds'].mean()
        response_status = "🟢" if avg_response < 1 else "🟡" if avg_response < 3 else "🔴"
        st.metric(
            f"{response_status} Avg Response Time",
            f"{avg_response:.2f}s",
            help="How long it takes to answer questions. Faster is better!"
        )
    
    with col3:
        if 'cpu_percent' in df.columns:
            avg_cpu = df['cpu_percent'].mean()
            cpu_status = "🟢" if avg_cpu < 50 else "🟡" if avg_cpu < 80 else "🔴"
            st.metric(
                f"{cpu_status} CPU Usage",
                f"{avg_cpu:.1f}%",
                help="How hard the computer is working. Lower is better!"
            )
        else:
            st.metric("CPU Usage", "N/A", help="CPU data not available")
    
    with col4:
        if 'num_threads' in df.columns:
            avg_threads = df['num_threads'].mean()
            thread_status = "🟢" if avg_threads < 50 else "🟡" if avg_threads < 100 else "🔴"
            st.metric(
                f"{thread_status} Active Tasks",
                f"{avg_threads:.0f}",
                help="Number of things the system is doing at once"
            )
        else:
            st.metric("Active Tasks", "N/A", help="Thread data not available")


def create_crash_analysis(df: pd.DataFrame):
    """Comprehensive crash analysis with child-friendly explanations."""
    st.header("🚨 Crash Analysis")
    st.markdown("*Understanding what went wrong and why*")
    
    # Filter to server errors (crashes)
    crashes = df[df['status_code'] >= 500].copy()
    
    if crashes.empty:
        st.success("🎉 **No crashes detected!** Your system is healthy.")
        
        # Show success metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Days Without Crashes", f"{(df['timestamp'].max() - df['timestamp'].min()).days}")
        with col2:
            st.metric("Success Rate", f"{(len(df) / len(df) * 100):.2f}%")
        
        return
    
    # Crash summary
    st.error(f"### 🔴 Found {len(crashes)} crashes in the last {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Last Crash",
            crashes['timestamp'].max().strftime("%b %d, %I:%M %p"),
            help="When did the most recent crash happen?"
        )
    
    with col2:
        if 'error_type' in crashes.columns:
            most_common = crashes['error_type'].mode()[0] if len(crashes['error_type'].mode()) > 0 else "Unknown"
            st.metric(
                "Most Common Problem",
                most_common,
                help="What type of error happens most often?"
            )
        else:
            st.metric("Most Common Problem", "N/A")
    
    with col3:
        crash_rate = (len(crashes) / len(df) * 100)
        st.metric(
            "Crash Rate",
            f"{crash_rate:.2f}%",
            help="Percentage of requests that crashed"
        )
    
    st.markdown("---")
    
    # Timeline of crashes
    st.markdown("### 📅 When Did Crashes Happen?")
    
    # Add crash indicator column
    crashes_timeline = crashes.copy()
    crashes_timeline['crash_severity'] = 'Server Error (5xx)'
    
    if 'crash_memory_rss_mb' in crashes_timeline.columns:
        fig = px.scatter(
            crashes_timeline,
            x='timestamp',
            y='crash_memory_rss_mb',
            color='error_type' if 'error_type' in crashes_timeline.columns else 'status_code',
            size='crash_memory_percent' if 'crash_memory_percent' in crashes_timeline.columns else None,
            hover_data={
                'endpoint': True,
                'error_message' if 'error_message' in crashes_timeline.columns else 'error': True,
                'crash_memory_rss_mb': ':.0f',
                'crash_num_threads' if 'crash_num_threads' in crashes_timeline.columns else 'num_threads': True if 'crash_num_threads' in crashes_timeline.columns or 'num_threads' in crashes_timeline.columns else False
            },
            title="Crash Timeline - When and Why?"
        )
        fig.update_layout(
            xaxis_title="Date & Time",
            yaxis_title="Memory When Crashed (MB)",
            hovermode='closest'
        )
    else:
        fig = px.scatter(
            crashes_timeline,
            x='timestamp',
            y='memory_rss_mb',
            color='error_type' if 'error_type' in crashes_timeline.columns else 'status_code',
            hover_data=['endpoint', 'error' if 'error' in crashes_timeline.columns else 'status_code'],
            title="Crash Timeline"
        )
        fig.update_layout(xaxis_title="Date & Time", yaxis_title="Memory (MB)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Error type breakdown
    st.markdown("### 🔍 Why Did It Crash?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'error_type' in crashes.columns and crashes['error_type'].notna().any():
            st.markdown("**Error Types:**")
            error_counts = crashes['error_type'].value_counts()
            fig = px.bar(
                x=error_counts.values,
                y=error_counts.index,
                orientation='h',
                title="Most Common Error Types"
            )
            fig.update_layout(xaxis_title="Count", yaxis_title="Error Type")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No detailed error type information available")
    
    with col2:
        st.markdown("**Which Pages Crash Most?**")
        endpoint_crashes = crashes['endpoint'].value_counts().head(10)
        fig = px.bar(
            x=endpoint_crashes.values,
            y=endpoint_crashes.index,
            orientation='h',
            title="Endpoints with Most Crashes"
        )
        fig.update_layout(xaxis_title="Number of Crashes", yaxis_title="API Endpoint")
        st.plotly_chart(fig, use_container_width=True)
    
    # Memory state during crashes
    st.markdown("---")
    st.markdown("### 💾 Memory State During Crashes")
    
    if 'crash_memory_rss_mb' in crashes.columns:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_crash_memory = crashes['crash_memory_rss_mb'].mean()
            st.metric(
                "Avg Memory at Crash",
                f"{avg_crash_memory:.0f} MB",
                help="How much memory was used when crashes happened"
            )
        
        with col2:
            if 'crash_memory_percent' in crashes.columns:
                avg_crash_pct = crashes['crash_memory_percent'].mean()
                st.metric(
                    "Avg Memory % at Crash",
                    f"{avg_crash_pct:.1f}%",
                    help="Percentage of total memory used"
                )
        
        with col3:
            if 'crash_num_threads' in crashes.columns:
                avg_crash_threads = crashes['crash_num_threads'].mean()
                st.metric(
                    "Avg Threads at Crash",
                    f"{avg_crash_threads:.0f}",
                    help="Number of active tasks when crash occurred"
                )
    
    # Detailed crash reports
    st.markdown("---")
    st.markdown("### 📝 Detailed Crash Reports")
    st.markdown("*For developers and technical staff*")
    
    # Show most recent crashes
    recent_crashes = crashes.sort_values('timestamp', ascending=False).head(10)
    
    for idx, row in recent_crashes.iterrows():
        crash_time = row['timestamp'].strftime("%b %d, %Y %I:%M:%S %p")
        error_type = row.get('error_type', 'Unknown Error')
        
        with st.expander(f"🔴 Crash at {crash_time} - {error_type}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Request Details:**")
                st.write(f"- **Endpoint:** `{row['endpoint']}`")
                st.write(f"- **Method:** `{row['method']}`")
                st.write(f"- **Status Code:** `{row['status_code']}`")
                if 'error_message' in row:
                    st.write(f"- **Error:** {row['error_message']}")
            
            with col2:
                st.markdown("**System State:**")
                if 'crash_memory_rss_mb' in row:
                    st.write(f"- **Memory:** {row['crash_memory_rss_mb']:.0f} MB ({row.get('crash_memory_percent', 0):.1f}%)")
                if 'crash_num_threads' in row:
                    st.write(f"- **Active Threads:** {row['crash_num_threads']}")
                if 'cpu_percent' in row:
                    st.write(f"- **CPU Usage:** {row['cpu_percent']:.1f}%")
            
            # Show traceback if available
            if 'traceback' in row and pd.notna(row['traceback']):
                st.markdown("**Stack Trace:**")
                st.code(row['traceback'], language='python')


def create_memory_leak_detector(df: pd.DataFrame):
    """Detect potential memory leaks with trend analysis."""
    st.header("💾 Memory Usage & Leak Detection")
    st.markdown("*Is memory growing over time?*")
    
    # Sort by time for trend analysis
    df_sorted = df.sort_values('timestamp').copy()
    
    # Calculate rolling average
    window_size = min(50, len(df_sorted) // 4)
    df_sorted['memory_trend'] = df_sorted['memory_rss_mb'].rolling(window=window_size, min_periods=1).mean()
    
    # Compare first and last averages
    first_100 = min(100, len(df_sorted) // 4)
    last_100 = min(100, len(df_sorted) // 4)
    
    if len(df_sorted) > 200:
        first_avg = df_sorted['memory_trend'].iloc[:first_100].mean()
        last_avg = df_sorted['memory_trend'].iloc[-last_100:].mean()
        memory_growth = ((last_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
        
        # Display leak status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if memory_growth > 20:
                st.error("🚨 **Memory Leak Detected!**")
                st.markdown(f"Memory grew by **{memory_growth:.1f}%** over time")
            elif memory_growth > 10:
                st.warning("⚠️ **Possible Memory Leak**")
                st.markdown(f"Memory grew by **{memory_growth:.1f}%** - monitor closely")
            else:
                st.success("✅ **No Memory Leak Detected**")
                st.markdown("Memory usage is stable")
        
        with col2:
            st.metric(
                "Starting Memory",
                f"{first_avg:.0f} MB",
                help="Average memory at the beginning of the period"
            )
        
        with col3:
            st.metric(
                "Current Memory",
                f"{last_avg:.0f} MB",
                delta=f"{last_avg - first_avg:+.0f} MB",
                help="Average memory at the end of the period"
            )
    
    # Memory trend chart
    st.markdown("### 📈 Memory Usage Over Time")
    
    fig = go.Figure()
    
    # Actual memory usage
    fig.add_trace(go.Scatter(
        x=df_sorted['timestamp'],
        y=df_sorted['memory_rss_mb'],
        mode='markers',
        name='Actual Memory',
        marker=dict(size=3, color='lightblue', opacity=0.5),
        hovertemplate='<b>%{x}</b><br>Memory: %{y:.0f} MB<extra></extra>'
    ))
    
    # Trend line
    fig.add_trace(go.Scatter(
        x=df_sorted['timestamp'],
        y=df_sorted['memory_trend'],
        mode='lines',
        name='Trend',
        line=dict(color='red', width=2),
        hovertemplate='<b>%{x}</b><br>Trend: %{y:.0f} MB<extra></extra>'
    ))
    
    # Add warning zone if memory is high
    if df_sorted['memory_rss_mb'].max() > 800:
        fig.add_hline(
            y=800,
            line_dash="dash",
            line_color="orange",
            annotation_text="⚠️ High Memory Zone",
            annotation_position="right"
        )
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Memory (MB)",
        hovermode='x unified',
        title="Memory consumption over time - Is it growing?"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Memory distribution
    st.markdown("### 📊 Memory Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df,
            x='memory_rss_mb',
            nbins=50,
            title="How Often Does Memory Hit Each Level?"
        )
        fig.update_layout(
            xaxis_title="Memory (MB)",
            yaxis_title="Number of Requests"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Memory percentiles
        st.markdown("**Memory Statistics:**")
        st.write(f"- **Average:** {df['memory_rss_mb'].mean():.0f} MB")
        st.write(f"- **Median (50%):** {df['memory_rss_mb'].median():.0f} MB")
        st.write(f"- **90th percentile:** {df['memory_rss_mb'].quantile(0.9):.0f} MB")
        st.write(f"- **95th percentile:** {df['memory_rss_mb'].quantile(0.95):.0f} MB")
        st.write(f"- **99th percentile:** {df['memory_rss_mb'].quantile(0.99):.0f} MB")
        st.write(f"- **Maximum:** {df['memory_rss_mb'].max():.0f} MB")
        
        # Explanation
        st.info("💡 **What this means:** 90% of the time, memory stays below the 90th percentile. If the maximum is much higher, there might be occasional spikes.")


def create_summary_metrics(df: pd.DataFrame):
    """Display summary metrics for the monitoring data."""
    st.header("📈 Quick Stats")
    st.markdown("*Key metrics at a glance*")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Requests",
            f"{len(df):,}",
            help="Total number of chatbot requests"
        )
    
    with col2:
        error_count = len(df[df['status_code'] >= 400])
        error_rate = (error_count / len(df) * 100) if len(df) > 0 else 0
        st.metric(
            "Errors",
            error_count,
            delta=f"{error_rate:.2f}%",
            delta_color="inverse",
            help="Requests that had problems (4xx/5xx)"
        )
    
    with col3:
        avg_duration = df['duration_seconds'].mean()
        st.metric(
            "Avg Response Time",
            f"{avg_duration:.2f}s",
            help="How long it takes to answer"
        )
    
    with col4:
        avg_memory = df['memory_rss_mb'].mean()
        st.metric(
            "Avg Memory",
            f"{avg_memory:.0f} MB",
            help="Average memory usage"
        )
    
    with col5:
        max_memory = df['memory_rss_mb'].max()
        st.metric(
            "Peak Memory",
            f"{max_memory:.0f} MB",
            help="Highest memory usage recorded"
        )


def create_time_series_charts(df: pd.DataFrame):
    """Display time series charts for performance analysis."""
    st.header("⚡ Performance Analysis")
    st.markdown("*Detailed performance metrics over time*")
    
    tab1, tab2, tab3 = st.tabs(["Response Times", "Traffic Patterns", "Error Analysis"])
    
    with tab1:
        st.subheader("📊 Response Time Analysis")
        
        # Response time scatter
        df_with_status = df.copy()
        df_with_status['status_category'] = df_with_status['status_code'].apply(
            lambda x: '🟢 Success (2xx)' if x < 300 
            else '🔵 Redirect (3xx)' if x < 400
            else '🟡 Client Error (4xx)' if x < 500
            else '🔴 Server Error (5xx)'
        )
        
        fig = px.scatter(
            df_with_status,
            x='timestamp',
            y='duration_seconds',
            color='status_category',
            title='Response Time by Request Status',
            labels={'duration_seconds': 'Response Time (seconds)'},
            hover_data=['endpoint', 'method']
        )
        
        # Add slow response threshold line
        fig.add_hline(
            y=3,
            line_dash="dash",
            line_color="orange",
            annotation_text="⚠️ Slow Response (>3s)",
            annotation_position="right"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Response time distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df,
                x='duration_seconds',
                nbins=50,
                title='Response Time Distribution'
            )
            fig.update_layout(xaxis_title='Response Time (seconds)', yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Response Time Statistics:**")
            st.write(f"- **Average:** {df['duration_seconds'].mean():.2f}s")
            st.write(f"- **Median:** {df['duration_seconds'].median():.2f}s")
            st.write(f"- **95th percentile:** {df['duration_seconds'].quantile(0.95):.2f}s")
            st.write(f"- **Slowest:** {df['duration_seconds'].max():.2f}s")
            
            slow_requests = len(df[df['duration_seconds'] > 3])
            st.write(f"- **Slow requests (>3s):** {slow_requests} ({slow_requests/len(df)*100:.1f}%)")
    
    with tab2:
        st.subheader("📈 Traffic Patterns")
        
        # Hourly traffic
        df_traffic = df.copy()
        df_traffic['hour'] = df_traffic['timestamp'].dt.floor('H')
        hourly_requests = df_traffic.groupby('hour').size().reset_index(name='count')
        
        fig = px.bar(
            hourly_requests,
            x='hour',
            y='count',
            title='Requests per Hour'
        )
        fig.update_layout(xaxis_title='Time', yaxis_title='Number of Requests')
        st.plotly_chart(fig, use_container_width=True)
        
        # Traffic by endpoint
        st.markdown("**Most Popular Endpoints:**")
        endpoint_traffic = df['endpoint'].value_counts().head(10)
        fig = px.bar(
            x=endpoint_traffic.values,
            y=endpoint_traffic.index,
            orientation='h',
            title='Top 10 Most Used Endpoints'
        )
        fig.update_layout(xaxis_title='Request Count', yaxis_title='Endpoint')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("❌ Error Analysis")
        
        errors_df = df[df['status_code'] >= 400].copy()
        
        if errors_df.empty:
            st.success("🎉 No errors found in this period!")
        else:
            # Error timeline
            errors_df['hour'] = errors_df['timestamp'].dt.floor('H')
            hourly_errors = errors_df.groupby(['hour', 'status_code']).size().reset_index(name='count')
            
            fig = px.bar(
                hourly_errors,
                x='hour',
                y='count',
                color='status_code',
                title='Errors Over Time'
            )
            fig.update_layout(xaxis_title='Time', yaxis_title='Number of Errors')
            st.plotly_chart(fig, use_container_width=True)
            
            # Error breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Errors by Status Code:**")
                status_counts = errors_df['status_code'].value_counts()
                for status, count in status_counts.items():
                    st.write(f"- **{status}:** {count} occurrences")
            
            with col2:
                st.markdown("**Endpoints with Most Errors:**")
                endpoint_errors = errors_df['endpoint'].value_counts().head(5)
                for endpoint, count in endpoint_errors.items():
                    st.write(f"- `{endpoint}`: {count} errors")


def create_system_diagnostics(df: pd.DataFrame):
    """Display detailed system diagnostics."""
    st.header("🖥️ System Diagnostics")
    st.markdown("*Detailed CPU, memory, and thread analysis*")
    
    tab1, tab2, tab3 = st.tabs(["CPU Usage", "Thread Analysis", "Memory Details"])
    
    with tab1:
        if 'cpu_percent' in df.columns:
            st.subheader("⚡ CPU Usage Over Time")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['cpu_percent'],
                mode='lines',
                name='Process CPU %',
                line=dict(color='blue')
            ))
            
            if 'system_cpu_percent' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['system_cpu_percent'],
                    mode='lines',
                    name='System CPU %',
                    line=dict(color='orange', dash='dash')
                ))
            
            # Add danger zone
            fig.add_hline(
                y=80,
                line_dash="dash",
                line_color="red",
                annotation_text="🔴 High CPU Zone",
                annotation_position="right"
            )
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="CPU Usage (%)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # CPU stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average CPU", f"{df['cpu_percent'].mean():.1f}%")
            with col2:
                st.metric("Peak CPU", f"{df['cpu_percent'].max():.1f}%")
            with col3:
                high_cpu = len(df[df['cpu_percent'] > 80])
                st.metric("High CPU Events", high_cpu)
        else:
            st.info("CPU data not available in monitoring logs")
    
    with tab2:
        if 'num_threads' in df.columns:
            st.subheader("🧵 Thread Count Analysis")
            
            fig = px.line(
                df,
                x='timestamp',
                y='num_threads',
                title='Active Threads Over Time'
            )
            
            # Add warning line
            fig.add_hline(
                y=100,
                line_dash="dash",
                line_color="orange",
                annotation_text="⚠️ High Thread Count",
                annotation_position="right"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Thread stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Threads", f"{df['num_threads'].mean():.0f}")
            with col2:
                st.metric("Peak Threads", f"{df['num_threads'].max():.0f}")
            with col3:
                high_threads = len(df[df['num_threads'] > 100])
                st.metric("Thread Spikes", high_threads)
            
            # Explanation
            st.info("💡 **What this means:** Too many threads can indicate resource leaks or too many concurrent operations. Sudden spikes might correlate with errors.")
        else:
            st.info("Thread data not available in monitoring logs")
    
    with tab3:
        st.subheader("💾 Detailed Memory Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Process Memory:**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['memory_rss_mb'],
                mode='lines',
                name='RSS Memory',
                fill='tozeroy'
            ))
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Memory (MB)",
                title="Resident Set Size (Actual Memory Used)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**System Memory:**")
            if 'system_memory_available_mb' in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['system_memory_available_mb'],
                    mode='lines',
                    name='Available Memory',
                    line=dict(color='green'),
                    fill='tozeroy'
                ))
                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Memory (MB)",
                    title="System Available Memory"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("System memory data not available")
        
        # Memory correlation with errors
        st.markdown("### 🔍 Does High Memory Cause Errors?")
        
        df_corr = df.copy()
        df_corr['is_error'] = (df_corr['status_code'] >= 400).astype(int)
        
        fig = px.scatter(
            df_corr,
            x='memory_percent' if 'memory_percent' in df_corr.columns else 'memory_rss_mb',
            y='duration_seconds',
            color=df_corr['is_error'].map({0: '🟢 Success', 1: '🔴 Error'}),
            title='Memory vs Response Time (colored by status)',
            labels={
                'memory_percent': 'Memory Usage (%)',
                'memory_rss_mb': 'Memory Usage (MB)',
                'duration_seconds': 'Response Time (s)'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation insight
        if 'memory_percent' in df_corr.columns:
            errors = df_corr[df_corr['is_error'] == 1]
            success = df_corr[df_corr['is_error'] == 0]
            
            if len(errors) > 0:
                avg_error_mem = errors['memory_percent'].mean()
                avg_success_mem = success['memory_percent'].mean()
                
                if avg_error_mem > avg_success_mem * 1.2:
                    st.warning(f"⚠️ **Errors happen when memory is higher!** Errors use {avg_error_mem:.1f}% memory on average, while successes use {avg_success_mem:.1f}%.")
                else:
                    st.success(f"✅ **Memory doesn't seem to cause errors.** Similar memory usage for both errors ({avg_error_mem:.1f}%) and successes ({avg_success_mem:.1f}%).")
