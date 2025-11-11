"""
Alert marker utilities for monitoring charts.
Adds Render-style alert annotations to charts for crashes, OOM kills, and high memory events.
"""

import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Tuple
from config import TOTAL_MEMORY_MB, EMERGENCY_THRESHOLD_PERCENT


def detect_alert_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect all alert-worthy events in the monitoring data.
    
    Returns DataFrame with alert events including:
    - HTTP crashes (status >= 500)
    - System crashes (uptime resets)
    - OOM kills (memory >= 90% before restart)
    - High memory warnings (memory >= 80%)
    - Emergency uploads
    
    Args:
        df: Monitoring DataFrame
    
    Returns:
        DataFrame with alert events
    """
    alerts = []
    
    df_sorted = df.sort_values('timestamp').copy()
    
    # Detect HTTP crashes (5xx errors)
    http_crashes = df_sorted[df_sorted['status_code'] >= 500].copy()
    for idx, row in http_crashes.iterrows():
        alerts.append({
            'timestamp': row['timestamp'],
            'memory_mb': row.get('memory_rss_mb', 0),
            'alert_type': 'http_crash',
            'severity': 'error',
            'symbol': 'x',
            'color': 'red',
            'size': 15,
            'label': f"HTTP {row['status_code']} Error",
            'details': row.get('error', 'Server error')
        })
    
    # Detect uptime resets (system crashes)
    if 'system_uptime_seconds' in df_sorted.columns:
        df_sorted['uptime_reset'] = df_sorted['system_uptime_seconds'].diff() < 0
        resets = df_sorted[df_sorted['uptime_reset']].copy()
        
        for idx, row in resets.iterrows():
            # Check if it was an OOM kill (memory >= 90% before reset)
            before_reset = df_sorted[df_sorted['timestamp'] < row['timestamp']].tail(5)
            
            if not before_reset.empty:
                avg_memory_before = before_reset['memory_rss_mb'].mean()
                memory_percent_before = (avg_memory_before / TOTAL_MEMORY_MB) * 100
                
                if memory_percent_before >= EMERGENCY_THRESHOLD_PERCENT:
                    alerts.append({
                        'timestamp': row['timestamp'],
                        'memory_mb': avg_memory_before,
                        'alert_type': 'oom_kill',
                        'severity': 'critical',
                        'symbol': 'x',
                        'color': 'darkred',
                        'size': 20,
                        'label': 'OOM Kill',
                        'details': f'Out of Memory crash ({memory_percent_before:.1f}% before restart)'
                    })
                else:
                    alerts.append({
                        'timestamp': row['timestamp'],
                        'memory_mb': avg_memory_before,
                        'alert_type': 'system_restart',
                        'severity': 'warning',
                        'symbol': 'circle',
                        'color': 'orange',
                        'size': 12,
                        'label': 'System Restart',
                        'details': f'Graceful restart ({memory_percent_before:.1f}% memory)'
                    })
    
    # Detect high memory warnings (>= 80%)
    high_memory_threshold = TOTAL_MEMORY_MB * 0.80
    high_memory = df_sorted[df_sorted['memory_rss_mb'] >= high_memory_threshold].copy()
    
    # Group consecutive high memory points to avoid clutter
    if not high_memory.empty:
        high_memory['time_group'] = (high_memory['timestamp'].diff() > pd.Timedelta(minutes=5)).cumsum()
        
        for group_id, group in high_memory.groupby('time_group'):
            # Take the peak memory point in this group
            peak_row = group.loc[group['memory_rss_mb'].idxmax()]
            memory_pct = (peak_row['memory_rss_mb'] / TOTAL_MEMORY_MB) * 100
            
            alerts.append({
                'timestamp': peak_row['timestamp'],
                'memory_mb': peak_row['memory_rss_mb'],
                'alert_type': 'high_memory',
                'severity': 'warning',
                'symbol': 'triangle-up',
                'color': 'orange',
                'size': 12,
                'label': 'High Memory',
                'details': f'Memory at {memory_pct:.1f}% ({peak_row["memory_rss_mb"]:.0f} MB)'
            })
    
    return pd.DataFrame(alerts) if alerts else pd.DataFrame()


def add_alert_markers_to_chart(fig: go.Figure, df: pd.DataFrame, y_column: str = 'memory_rss_mb') -> go.Figure:
    """
    Add alert markers to an existing Plotly chart (Render-style).
    
    Args:
        fig: Plotly Figure object
        df: Monitoring DataFrame
        y_column: Column name for y-axis values (default: 'memory_rss_mb')
    
    Returns:
        Enhanced Figure with alert markers
    """
    alerts_df = detect_alert_events(df)
    
    if alerts_df.empty:
        return fig
    
    # Group by alert type for cleaner legend
    for alert_type in alerts_df['alert_type'].unique():
        type_alerts = alerts_df[alerts_df['alert_type'] == alert_type]
        
        if type_alerts.empty:
            continue
        
        # Use the first row to get color/symbol/label
        first_row = type_alerts.iloc[0]
        
        fig.add_trace(go.Scatter(
            x=type_alerts['timestamp'],
            y=type_alerts['memory_mb'],
            mode='markers',
            name=first_row['label'],
            marker=dict(
                size=first_row['size'],
                color=first_row['color'],
                symbol=first_row['symbol'],
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{text}</b><br>Time: %{x}<br>Memory: %{y:.0f} MB<extra></extra>',
            text=type_alerts['details'],
            showlegend=True
        ))
    
    return fig


def create_event_timeline_data(df: pd.DataFrame) -> List[Dict]:
    """
    Create event timeline data for sidebar display.
    
    Returns list of events sorted by timestamp (most recent first).
    
    Args:
        df: Monitoring DataFrame
    
    Returns:
        List of event dictionaries
    """
    alerts_df = detect_alert_events(df)
    
    if alerts_df.empty:
        return []
    
    # Sort by timestamp descending (most recent first)
    alerts_df = alerts_df.sort_values('timestamp', ascending=False)
    
    events = []
    for idx, row in alerts_df.iterrows():
        # Determine icon based on severity
        if row['severity'] == 'critical':
            icon = '🔴'
        elif row['severity'] == 'error':
            icon = '⚠️'
        elif row['severity'] == 'warning':
            icon = '🟡'
        else:
            icon = '🔵'
        
        events.append({
            'timestamp': row['timestamp'],
            'icon': icon,
            'label': row['label'],
            'details': row['details'],
            'severity': row['severity']
        })
    
    return events
