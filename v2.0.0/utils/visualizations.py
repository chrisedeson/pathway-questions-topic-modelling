"""
Visualization components for charts and analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
from config import BYU_COLORS, CHART_COLOR_PALETTE


def create_kpi_cards(kpis: Dict[str, any]):
    """
    Display KPI cards in a grid layout.
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Questions Processed",
            value=f"{kpis['total_questions']:,}",
            help="Total number of questions analyzed"
        )
    
    with col2:
        matched_pct = (kpis['matched_existing'] / kpis['total_questions'] * 100) if kpis['total_questions'] > 0 else 0
        st.metric(
            label="✅ Matched to Existing Topics",
            value=f"{kpis['matched_existing']:,}",
            delta=f"{matched_pct:.1f}%",
            help="Questions successfully matched to existing topics"
        )
    
    with col3:
        st.metric(
            label="🆕 New Topics Discovered",
            value=f"{kpis['new_topics_discovered']:,}",
            delta=f"{kpis['questions_in_new_topics']} questions",
            help="Unique new topics identified by clustering"
        )
    
    with col4:
        st.metric(
            label="🌍 Countries",
            value=f"{kpis['countries']:,}",
            help="Number of unique countries represented"
        )
    
    # Second row of KPIs
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        if kpis['avg_similarity'] > 0:
            st.metric(
                label="📈 Avg Similarity Score",
                value=f"{kpis['avg_similarity']:.3f}",
                help="Average similarity score for matched questions"
            )
    
    with col6:
        if kpis['last_updated']:
            last_updated_str = kpis['last_updated'].strftime("%Y-%m-%d %H:%M")
            st.metric(
                label="🕐 Last Updated",
                value=last_updated_str,
                help="Last time data was updated"
            )


def plot_classification_distribution(df: pd.DataFrame):
    """
    Pie chart showing distribution of existing vs new topics.
    """
    if 'classification' not in df.columns or df.empty:
        st.info("No classification data available")
        return
    
    classification_counts = df['classification'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=classification_counts.index,
        values=classification_counts.values,
        hole=0.4,
        marker=dict(colors=[BYU_COLORS['primary'], BYU_COLORS['secondary']]),
        textinfo='label+percent+value',
        textfont=dict(size=14)
    )])
    
    fig.update_layout(
        title="Question Classification Distribution",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig, use_container_width=True, key="classification_distribution_pie")


def plot_country_distribution(df: pd.DataFrame, top_n: int = 10):
    """
    Bar chart showing top countries by question volume.
    """
    if 'country' not in df.columns or df.empty:
        st.info("No country data available")
        return
    
    country_counts = df['country'].value_counts().head(top_n)
    
    fig = go.Figure(data=[go.Bar(
        x=country_counts.values,
        y=country_counts.index,
        orientation='h',
        marker=dict(color=BYU_COLORS['primary']),
        text=country_counts.values,
        textposition='outside'
    )])
    
    fig.update_layout(
        title=f"Top {top_n} Countries by Question Volume",
        xaxis_title="Number of Questions",
        yaxis_title="Country",
        height=400,
        showlegend=False,
        yaxis=dict(autorange="reversed")
    )
    
    st.plotly_chart(fig, use_container_width=True, key="country_distribution_bar")


def plot_timeline(df: pd.DataFrame, key: str = "timeline_line_chart"):
    """
    Line chart showing questions over time.
    
    Args:
        df: DataFrame with timestamp and classification columns
        key: Unique key for the plotly chart (required if used multiple times)
    """
    if 'timestamp' not in df.columns or df.empty:
        st.info("No timestamp data available")
        return
    
    # Group by date and classification
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['timestamp'], format='ISO8601').dt.date
    
    timeline_data = df_copy.groupby(['date', 'classification']).size().reset_index(name='count')
    
    fig = px.line(
        timeline_data,
        x='date',
        y='count',
        color='classification',
        color_discrete_map={
            'Existing Topic': BYU_COLORS['primary'],
            'New Topic': BYU_COLORS['secondary']
        },
        markers=True
    )
    
    fig.update_layout(
        title="Question Volume Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Questions",
        height=400,
        hovermode='x unified',
        legend=dict(
            title="Classification",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True, key=key)


def plot_similarity_distribution(df: pd.DataFrame):
    """
    Histogram showing distribution of similarity scores.
    """
    if 'similarity_score' not in df.columns or df.empty:
        st.info("No similarity score data available")
        return
    
    # Filter to only existing topics (which have similarity scores)
    df_filtered = df[df['classification'] == 'Existing Topic']
    
    if df_filtered.empty:
        st.info("No similarity scores available for existing topics")
        return
    
    fig = go.Figure(data=[go.Histogram(
        x=df_filtered['similarity_score'],
        nbinsx=30,
        marker=dict(color=BYU_COLORS['primary']),
        hovertemplate='Similarity Score: %{x:.3f}<br>Count: %{y}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Similarity Score Distribution (Existing Topics)",
        xaxis_title="Similarity Score",
        yaxis_title="Number of Questions",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, key="similarity_distribution_histogram")


def plot_top_topics(df: pd.DataFrame, top_n: int = 10):
    """
    Bar chart showing most common matched topics.
    """
    if 'matched_topic' not in df.columns or df.empty:
        st.info("No topic data available")
        return
    
    # Filter to existing topics only
    df_filtered = df[df['classification'] == 'Existing Topic']
    
    if df_filtered.empty:
        st.info("No matched topics available")
        return
    
    topic_counts = df_filtered['matched_topic'].value_counts().head(top_n)
    
    fig = go.Figure(data=[go.Bar(
        x=topic_counts.values,
        y=topic_counts.index,
        orientation='h',
        marker=dict(color=BYU_COLORS['accent1']),
        text=topic_counts.values,
        textposition='outside'
    )])
    
    fig.update_layout(
        title=f"Top {top_n} Most Common Topics",
        xaxis_title="Number of Questions",
        yaxis_title="Topic",
        height=500,
        showlegend=False,
        yaxis=dict(autorange="reversed")
    )
    
    st.plotly_chart(fig, use_container_width=True, key="top_topics_bar")


def plot_hourly_heatmap(df: pd.DataFrame):
    """
    Heatmap showing question volume by day of week and hour.
    """
    if 'timestamp' not in df.columns or df.empty:
        st.info("No timestamp data available")
        return
    
    df_copy = df.copy()
    df_copy['hour'] = pd.to_datetime(df_copy['timestamp'], format='ISO8601').dt.hour
    df_copy['day_of_week'] = pd.to_datetime(df_copy['timestamp'], format='ISO8601').dt.day_name()
    
    # Create pivot table
    heatmap_data = df_copy.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
    heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot = heatmap_pivot.reindex([day for day in day_order if day in heatmap_pivot.index])
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='Blues',
        hovertemplate='Day: %{y}<br>Hour: %{x}:00<br>Questions: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Question Volume Heatmap (Day × Hour)",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True, key="hourly_heatmap")


def plot_language_distribution(df: pd.DataFrame):
    """
    Pie chart showing distribution of user languages.
    """
    if 'user_language' not in df.columns or df.empty:
        st.info("No language data available")
        return
    
    language_counts = df['user_language'].value_counts().head(10)
    
    fig = go.Figure(data=[go.Pie(
        labels=language_counts.index,
        values=language_counts.values,
        hole=0.3,
        marker=dict(colors=CHART_COLOR_PALETTE),
        textinfo='label+percent',
        textfont=dict(size=12)
    )])
    
    fig.update_layout(
        title="User Language Distribution",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig, use_container_width=True, key="language_distribution_pie")
