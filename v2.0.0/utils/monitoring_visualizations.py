"""
Visualization components for the Monitoring Dashboard with Crash Diagnosis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import json


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
    
    # Detect uptime resets (indicates restarts/crashes)
    if 'system_uptime_seconds' in df.columns:
        st.markdown("### 🔄 Uptime Resets & Crash Detection")
        st.markdown("*System restarts often indicate crashes*")
        
        df_sorted = df.sort_values('timestamp').copy()
        df_sorted['uptime_reset'] = df_sorted['system_uptime_seconds'].diff() < 0
        
        resets = df_sorted[df_sorted['uptime_reset']].copy()
        
        if len(resets) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🔄 Detected Restarts",
                    len(resets),
                    help="Number of times the system restarted (uptime reset to zero)"
                )
            
            with col2:
                # Calculate average uptime between restarts
                if len(resets) > 1:
                    time_diffs = resets['timestamp'].diff().dropna()
                    avg_uptime_hours = time_diffs.mean().total_seconds() / 3600
                    st.metric(
                        "⏱️ Avg Time Between Restarts",
                        f"{avg_uptime_hours:.1f} hours",
                        help="How long the system typically runs before restarting"
                    )
                else:
                    st.metric("⏱️ Avg Time Between Restarts", "N/A")
            
            with col3:
                # Check if recent (last 24 hours)
                last_reset = resets['timestamp'].max()
                hours_since_reset = (df['timestamp'].max() - last_reset).total_seconds() / 3600
                if hours_since_reset < 1:
                    st.error(f"⚠️ **{hours_since_reset*60:.0f} min ago**")
                    st.caption("Last restart")
                elif hours_since_reset < 24:
                    st.warning(f"⏰ **{hours_since_reset:.1f} hours ago**")
                    st.caption("Last restart")
                else:
                    st.success(f"✅ **{hours_since_reset/24:.1f} days ago**")
                    st.caption("Last restart")
            
            # OOM (Out-of-Memory) diagnosis
            st.markdown("#### 🔍 Crash Diagnosis: Was it OOMKilled?")
            
            # Check memory levels at restart points
            oom_likely = []
            for idx, reset_row in resets.iterrows():
                # Get memory just before reset
                before_reset = df_sorted[df_sorted['timestamp'] < reset_row['timestamp']].tail(5)
                if not before_reset.empty:
                    avg_memory_before = before_reset['memory_rss_mb'].mean()
                    memory_percent_before = (avg_memory_before / 2048) * 100  # Assuming 2GB limit
                    
                    if memory_percent_before >= 90:
                        oom_likely.append({
                            'timestamp': reset_row['timestamp'],
                            'memory_mb': avg_memory_before,
                            'memory_percent': memory_percent_before,
                            'likely_oom': True
                        })
                    else:
                        oom_likely.append({
                            'timestamp': reset_row['timestamp'],
                            'memory_mb': avg_memory_before,
                            'memory_percent': memory_percent_before,
                            'likely_oom': False
                        })
            
            if oom_likely:
                oom_df = pd.DataFrame(oom_likely)
                oom_count = oom_df['likely_oom'].sum()
                graceful_count = len(oom_df) - oom_count
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if oom_count > 0:
                        st.error(f"🔴 **{oom_count} OOMKilled**")
                        st.caption("Memory >= 90% before restart")
                        st.markdown("**These crashes were likely caused by running out of memory!**")
                    else:
                        st.success("✅ **No OOMKilled crashes**")
                        st.caption("Memory was below 90%")
                
                with col2:
                    if graceful_count > 0:
                        st.info(f"🔵 **{graceful_count} Graceful**")
                        st.caption("Memory < 90% before restart")
                        st.markdown("**These were likely intentional restarts or deployments.**")
                
                # Show restart timeline with memory correlation
                fig = go.Figure()
                
                # All memory usage
                fig.add_trace(go.Scatter(
                    x=df_sorted['timestamp'],
                    y=df_sorted['memory_rss_mb'],
                    mode='lines',
                    name='Memory Usage',
                    line=dict(color='lightblue', width=1),
                    opacity=0.5
                ))
                
                # Mark restart points
                for idx, row in oom_df.iterrows():
                    color = 'red' if row['likely_oom'] else 'green'
                    symbol = 'x' if row['likely_oom'] else 'circle'
                    name = 'OOMKilled' if row['likely_oom'] else 'Graceful Restart'
                    
                    fig.add_trace(go.Scatter(
                        x=[row['timestamp']],
                        y=[row['memory_mb']],
                        mode='markers',
                        name=name,
                        marker=dict(size=15, color=color, symbol=symbol, line=dict(color='white', width=2)),
                        showlegend=True,
                        hovertemplate=f'<b>{name}</b><br>Time: %{{x}}<br>Memory: {row["memory_mb"]:.0f} MB ({row["memory_percent"]:.1f}%)<extra></extra>'
                    ))
                
                # Add emergency threshold line
                fig.add_hline(
                    y=1843,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="🔴 Emergency Threshold (90%)",
                    annotation_position="right"
                )
                
                fig.update_layout(
                    title="Restart Timeline with Memory Correlation",
                    xaxis_title="Time",
                    yaxis_title="Memory (MB)",
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ **No unexpected restarts detected!** System has been stable.")
    
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
    """Detect potential memory leaks with trend analysis and predictions."""
    st.header("💾 Memory Usage & Leak Detection")
    st.markdown("*Is memory growing over time?*")
    
    # Sort by time for trend analysis
    df_sorted = df.sort_values('timestamp').copy()
    
    # Calculate rolling average
    window_size = min(50, len(df_sorted) // 4)
    df_sorted['memory_trend'] = df_sorted['memory_rss_mb'].rolling(window=window_size, min_periods=1).mean()
    
    # Linear regression for prediction
    prediction_result = None
    if len(df_sorted) > 10:
        # Prepare data for linear regression
        df_sorted['time_numeric'] = (df_sorted['timestamp'] - df_sorted['timestamp'].min()).dt.total_seconds()
        X = df_sorted['time_numeric'].values.reshape(-1, 1)
        y = df_sorted['memory_rss_mb'].values
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate predictions
        df_sorted['memory_predicted'] = model.predict(X)
        
        # Predict future (24 hours ahead)
        current_time = df_sorted['time_numeric'].max()
        future_times = np.array([current_time + (3600 * i) for i in range(1, 25)]).reshape(-1, 1)
        future_predictions = model.predict(future_times)
        
        # Check if trending upward
        slope_mb_per_hour = model.coef_[0] / 3600  # Convert per-second to per-hour
        is_growing = slope_mb_per_hour > 0.1  # Growing more than 0.1 MB/hour
        
        # Calculate time until critical threshold (90% of 2048 MB = 1843 MB)
        EMERGENCY_THRESHOLD_MB = 1843
        current_avg = df_sorted['memory_rss_mb'].iloc[-100:].mean()
        
        hours_until_critical = None
        if is_growing and slope_mb_per_hour > 0:
            mb_until_critical = EMERGENCY_THRESHOLD_MB - current_avg
            if mb_until_critical > 0:
                hours_until_critical = mb_until_critical / slope_mb_per_hour
        
        prediction_result = {
            'slope_mb_per_hour': slope_mb_per_hour,
            'is_growing': is_growing,
            'hours_until_critical': hours_until_critical,
            'future_predictions': future_predictions,
            'model_score': model.score(X, y)
        }
    
    # Compare first and last averages
    first_100 = min(100, len(df_sorted) // 4)
    last_100 = min(100, len(df_sorted) // 4)
    
    if len(df_sorted) > 200:
        first_avg = df_sorted['memory_trend'].iloc[:first_100].mean()
        last_avg = df_sorted['memory_trend'].iloc[-last_100:].mean()
        memory_growth = ((last_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
        
        # Display leak status with prediction
        col1, col2, col3, col4 = st.columns(4)
        
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
        
        with col4:
            if prediction_result and prediction_result['is_growing']:
                if prediction_result['hours_until_critical']:
                    hours = prediction_result['hours_until_critical']
                    if hours < 24:
                        st.error(f"⏰ **{hours:.1f} hours**")
                        st.caption("Until critical memory")
                    elif hours < 72:
                        st.warning(f"⏰ **{hours/24:.1f} days**")
                        st.caption("Until critical memory")
                    else:
                        st.info(f"⏰ **{hours/24:.0f} days**")
                        st.caption("Until critical memory")
                else:
                    st.success("✅ **Safe**")
                    st.caption("Below critical threshold")
            else:
                st.success("📊 **Stable**")
                st.caption("No growth detected")
    
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
    
    # Trend line (rolling average)
    fig.add_trace(go.Scatter(
        x=df_sorted['timestamp'],
        y=df_sorted['memory_trend'],
        mode='lines',
        name='Trend (Rolling Avg)',
        line=dict(color='orange', width=2),
        hovertemplate='<b>%{x}</b><br>Trend: %{y:.0f} MB<extra></extra>'
    ))
    
    # Linear regression prediction line
    if prediction_result:
        fig.add_trace(go.Scatter(
            x=df_sorted['timestamp'],
            y=df_sorted['memory_predicted'],
            mode='lines',
            name='Linear Prediction',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='<b>%{x}</b><br>Predicted: %{y:.0f} MB<extra></extra>'
        ))
        
        # Add slope annotation
        slope_text = f"Growth Rate: {prediction_result['slope_mb_per_hour']:.2f} MB/hour"
        fig.add_annotation(
            text=slope_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1
        )
    
    # Add warning zone if memory is high
    if df_sorted['memory_rss_mb'].max() > 800:
        fig.add_hline(
            y=800,
            line_dash="dash",
            line_color="orange",
            annotation_text="⚠️ High Memory Zone",
            annotation_position="right"
        )
    
    # Add emergency threshold line
    fig.add_hline(
        y=1843,
        line_dash="dash",
        line_color="red",
        annotation_text="🔴 Emergency Threshold (90%)",
        annotation_position="right"
    )
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Memory (MB)",
        hovermode='x unified',
        title="Memory consumption over time - Is it growing?"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Per-endpoint memory breakdown
    if 'endpoint' in df.columns:
        st.markdown("### 📊 Memory Usage by Endpoint")
        st.markdown("*Which endpoints use the most memory?*")
        
        endpoint_memory = df.groupby('endpoint').agg({
            'memory_rss_mb': ['mean', 'max', 'count']
        }).reset_index()
        endpoint_memory.columns = ['endpoint', 'avg_memory', 'max_memory', 'request_count']
        endpoint_memory = endpoint_memory.sort_values('avg_memory', ascending=False).head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                endpoint_memory,
                x='avg_memory',
                y='endpoint',
                orientation='h',
                title='Average Memory by Endpoint (Top 10)',
                labels={'avg_memory': 'Average Memory (MB)', 'endpoint': 'Endpoint'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                endpoint_memory,
                x='max_memory',
                y='endpoint',
                orientation='h',
                title='Peak Memory by Endpoint (Top 10)',
                labels={'max_memory': 'Peak Memory (MB)', 'endpoint': 'Endpoint'},
                color='max_memory',
                color_continuous_scale='Reds'
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
        df_traffic['hour'] = df_traffic['timestamp'].dt.floor('h')
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
            errors_df['hour'] = errors_df['timestamp'].dt.floor('h')
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


def create_data_completeness_check(df: pd.DataFrame):
    """Check for gaps in monitoring data and assess data quality."""
    st.header("📋 Data Completeness Check")
    st.markdown("*Are we collecting monitoring data consistently?*")
    
    if df.empty:
        st.error("❌ No data available for completeness check")
        return
    
    # Sort by timestamp
    df_sorted = df.sort_values('timestamp').copy()
    
    # Calculate expected hourly data points
    date_range = (df_sorted['timestamp'].max() - df_sorted['timestamp'].min())
    expected_hours = int(date_range.total_seconds() / 3600)
    
    # Group by hour and count records
    df_sorted['hour_bucket'] = df_sorted['timestamp'].dt.floor('h')
    hourly_counts = df_sorted.groupby('hour_bucket').size().reset_index(name='count')
    
    # Detect gaps (hours with no data)
    all_hours = pd.date_range(
        start=df_sorted['timestamp'].min().floor('h'),
        end=df_sorted['timestamp'].max().floor('h'),
        freq='h'
    )
    
    gaps = []
    for i in range(len(all_hours) - 1):
        hour = all_hours[i]
        if hour not in hourly_counts['hour_bucket'].values:
            # Calculate gap duration
            gap_start = hour
            gap_end = hour
            j = i + 1
            while j < len(all_hours) and all_hours[j] not in hourly_counts['hour_bucket'].values:
                gap_end = all_hours[j]
                j += 1
            
            gap_hours = (gap_end - gap_start).total_seconds() / 3600 + 1
            
            # Classify severity
            if gap_hours < 1:
                severity = "minor"
            elif gap_hours < 6:
                severity = "warning"
            else:
                severity = "critical"
            
            gaps.append({
                'start': gap_start,
                'end': gap_end,
                'duration_hours': gap_hours,
                'severity': severity
            })
    
    # Calculate health score
    data_points = len(df_sorted)
    avg_per_hour = data_points / max(expected_hours, 1)
    gap_penalty = len([g for g in gaps if g['severity'] in ['warning', 'critical']]) * 5
    completeness_score = max(0, min(100, 100 - gap_penalty))
    
    # Display health score
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if completeness_score >= 90:
            st.success(f"### 🟢 {completeness_score}/100")
            st.caption("Excellent data quality")
        elif completeness_score >= 70:
            st.warning(f"### 🟡 {completeness_score}/100")
            st.caption("Some gaps detected")
        else:
            st.error(f"### 🔴 {completeness_score}/100")
            st.caption("Poor data quality")
    
    with col2:
        st.metric(
            "Total Data Points",
            f"{data_points:,}",
            help="Total monitoring records collected"
        )
    
    with col3:
        st.metric(
            "Avg Records/Hour",
            f"{avg_per_hour:.1f}",
            help="How many records collected each hour on average"
        )
    
    with col4:
        st.metric(
            "Data Gaps Found",
            len(gaps),
            help="Number of time periods with missing data"
        )
    
    # Show gaps if any exist
    if gaps:
        st.markdown("---")
        st.markdown("### ⚠️ Detected Data Gaps")
        
        gaps_df = pd.DataFrame(gaps)
        
        # Count by severity
        severity_counts = gaps_df['severity'].value_counts()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            critical = severity_counts.get('critical', 0)
            if critical > 0:
                st.error(f"🔴 **{critical} Critical**")
                st.caption(">6 hours missing")
            else:
                st.success("✅ **0 Critical**")
        
        with col2:
            warning = severity_counts.get('warning', 0)
            if warning > 0:
                st.warning(f"🟡 **{warning} Warnings**")
                st.caption("1-6 hours missing")
            else:
                st.success("✅ **0 Warnings**")
        
        with col3:
            minor = severity_counts.get('minor', 0)
            if minor > 0:
                st.info(f"🔵 **{minor} Minor**")
                st.caption("<1 hour missing")
            else:
                st.success("✅ **0 Minor**")
        
        # Show gap timeline
        st.markdown("#### 📅 Gap Timeline")
        
        # Create visualization
        fig = go.Figure()
        
        # Add hourly data availability
        fig.add_trace(go.Scatter(
            x=hourly_counts['hour_bucket'],
            y=hourly_counts['count'],
            mode='lines',
            name='Records per Hour',
            fill='tozeroy',
            line=dict(color='green')
        ))
        
        # Mark gaps
        for gap in gaps:
            color = {'minor': 'yellow', 'warning': 'orange', 'critical': 'red'}[gap['severity']]
            
            # Use add_shape instead of add_vrect to avoid timestamp arithmetic issues
            fig.add_shape(
                type="rect",
                x0=gap['start'],
                x1=gap['end'],
                y0=0,
                y1=1,
                yref="paper",
                fillcolor=color,
                opacity=0.3,
                layer="below",
                line_width=0
            )
            
            # Add annotation separately at a fixed position
            mid_time = gap['start'] + (gap['end'] - gap['start']) / 2
            fig.add_annotation(
                x=mid_time,
                y=1,
                yref="paper",
                text=f"{gap['duration_hours']:.0f}h gap",
                showarrow=False,
                yshift=10,
                font=dict(size=10, color=color),
                bgcolor="white",
                opacity=0.8
            )
        
        fig.update_layout(
            title="Data Collection Timeline - Green areas have data, colored areas are gaps",
            xaxis_title="Time",
            yaxis_title="Records Collected",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed gap table
        with st.expander("📋 Detailed Gap Report"):
            gaps_df_display = gaps_df.copy()
            gaps_df_display['start'] = gaps_df_display['start'].dt.strftime('%Y-%m-%d %H:%M')
            gaps_df_display['end'] = gaps_df_display['end'].dt.strftime('%Y-%m-%d %H:%M')
            gaps_df_display = gaps_df_display.sort_values('duration_hours', ascending=False)
            
            st.dataframe(gaps_df_display, use_container_width=True)
    else:
        st.success("✅ **Perfect! No data gaps detected.** Monitoring is collecting data consistently.")


def create_emergency_alert_banner(s3_client=None, bucket: Optional[str] = None, prefix: Optional[str] = None):
    """
    Load and display emergency alerts from S3 ALERT_*.json files.
    Shows a prominent banner at the top of the dashboard.
    """
    if not s3_client or not bucket or not prefix:
        return  # Silently skip if S3 not configured
    
    try:
        # List all ALERT_*.json files
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=f"{prefix}/"
        )
        
        alerts = []
        
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                filename = key.split('/')[-1]
                
                if filename.startswith('ALERT_') and filename.endswith('.json'):
                    try:
                        # Load alert JSON
                        obj_data = s3_client.get_object(Bucket=bucket, Key=key)
                        alert_data = json.loads(obj_data['Body'].read().decode('utf-8'))
                        
                        # Add metadata
                        alert_data['alert_file'] = filename
                        alert_data['alert_time'] = obj['LastModified']
                        
                        alerts.append(alert_data)
                    except Exception as e:
                        st.error(f"Error loading alert {filename}: {e}")
        
        if alerts:
            # Sort by time (most recent first)
            alerts.sort(key=lambda x: x['alert_time'], reverse=True)
            
            # Show most recent alert prominently
            latest_alert = alerts[0]
            
            # Determine alert color and emoji
            severity = latest_alert.get('severity', 'warning')
            if severity == 'critical':
                alert_color = "🔴"
                alert_style = "error"
            else:
                alert_color = "🟡"
                alert_style = "warning"
            
            # Display alert banner
            if alert_style == "error":
                st.error(f"""
                ### {alert_color} CRITICAL ALERT: {latest_alert.get('alert_type', 'Unknown').replace('_', ' ').title()}
                
                **Message:** {latest_alert.get('message', 'No message provided')}
                
                **Details:**
                - Memory Usage: {latest_alert.get('memory_mb', 'N/A')} MB ({latest_alert.get('memory_percent', 'N/A')}%)
                - Threshold: {latest_alert.get('threshold_mb', 'N/A')} MB ({latest_alert.get('threshold_percent', 'N/A')}%)
                - System Memory: {latest_alert.get('system_memory_mb', 'N/A')} MB
                - Alert Time: {latest_alert['alert_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}
                
                ⚠️ **Immediate action required!** The system is under high memory pressure.
                """)
            else:
                st.warning(f"""
                ### {alert_color} WARNING: {latest_alert.get('alert_type', 'Unknown').replace('_', ' ').title()}
                
                **Message:** {latest_alert.get('message', 'No message provided')}
                
                **Details:**
                - Memory Usage: {latest_alert.get('memory_mb', 'N/A')} MB ({latest_alert.get('memory_percent', 'N/A')}%)
                - Alert Time: {latest_alert['alert_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}
                """)
            
            # Show count of other alerts
            if len(alerts) > 1:
                with st.expander(f"📋 View All Alerts ({len(alerts)} total)"):
                    for i, alert in enumerate(alerts, 1):
                        st.markdown(f"""
                        **Alert {i}:** {alert.get('alert_type', 'Unknown')} - {alert['alert_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}
                        - Message: {alert.get('message', 'N/A')}
                        - Memory: {alert.get('memory_mb', 'N/A')} MB ({alert.get('memory_percent', 'N/A')}%)
                        - Severity: {alert.get('severity', 'N/A')}
                        """)
                        st.markdown("---")
        
        return len(alerts) if alerts else 0
        
    except Exception as e:
        # Don't show errors if S3 not accessible (might be intentional)
        return 0
