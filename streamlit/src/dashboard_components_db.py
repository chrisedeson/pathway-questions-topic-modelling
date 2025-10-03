"""
Database-backed dashboard components for instant loading with pre-computed data
Enhanced with smart caching and background updates
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np

from data_service import get_data_service

# Import our smart data manager
try:
    from streamlit_data_manager import data_manager
except ImportError:
    data_manager = None


def display_dashboard_header():
    """Display main dashboard header"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #1f4e79 0%, #2e7d9a 100%); 
                border-radius: 10px; margin-bottom: 2rem; color: white;">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">📊 BYU Pathway Questions Dashboard</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            Real-time insights from student questions and topic analysis
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_time_period_selector() -> str:
    """Display time period selector and return selected period"""
    st.markdown("### 📅 Time Period")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📅 Last 24 Hours", use_container_width=True):
            st.session_state.selected_period = '24h'
    
    with col2:
        if st.button("📅 Last 7 Days", use_container_width=True):
            st.session_state.selected_period = '7d'
    
    with col3:
        if st.button("📅 Last 30 Days", use_container_width=True):
            st.session_state.selected_period = '30d'
    
    with col4:
        if st.button("📅 All Time", use_container_width=True):
            st.session_state.selected_period = 'all'
    
    # Default to 'all' if not set
    if 'selected_period' not in st.session_state:
        st.session_state.selected_period = 'all'
    
    # Display current selection
    period_labels = {
        '24h': 'Last 24 Hours',
        '7d': 'Last 7 Days', 
        '30d': 'Last 30 Days',
        'all': 'All Time'
    }
    
    current_label = period_labels.get(st.session_state.selected_period, 'All Time')
    st.info(f"📊 Currently viewing: **{current_label}**")
    
    return st.session_state.selected_period


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_dashboard_data(time_period: str) -> Dict[str, Any]:
    """Load dashboard data from database service"""
    try:
        data_service = get_data_service()
        return data_service.get_dashboard_data(time_period)
    except Exception as e:
        st.error(f"❌ Error loading dashboard data: {e}")
        return {'error': str(e)}


def display_key_metrics(dashboard_data: Dict[str, Any]):
    """Display key metrics cards"""
    if 'error' in dashboard_data:
        st.error(f"❌ Unable to load metrics: {dashboard_data['error']}")
        return
    
    metrics = dashboard_data.get('metrics', {})
    
    st.markdown("### 📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_questions = metrics.get('total_questions', 0)
        st.metric(
            label="📝 Total Questions",
            value=f"{total_questions:,}",
            help="Total number of questions in the selected time period"
        )
    
    with col2:
        total_topics = metrics.get('total_topics', 0)
        st.metric(
            label="🏷️ Topics Discovered",
            value=f"{total_topics:,}",
            help="Number of unique topics identified through analysis"
        )
    
    with col3:
        # Calculate most common language
        lang_dist = dashboard_data.get('language_distribution', {})
        if lang_dist:
            most_common_lang = max(lang_dist.items(), key=lambda x: x[1])
            st.metric(
                label="🗣️ Primary Language",
                value=most_common_lang[0].upper(),
                delta=f"{most_common_lang[1]} questions",
                help="Most frequently used language in questions"
            )
        else:
            st.metric(label="🗣️ Primary Language", value="N/A")
    
    with col4:
        # Display last update time
        last_updated = dashboard_data.get('last_updated')
        if last_updated:
            updated_dt = pd.to_datetime(last_updated)
            time_ago = datetime.now() - updated_dt.tz_localize(None)
            
            if time_ago.total_seconds() < 3600:  # Less than 1 hour
                time_str = f"{int(time_ago.total_seconds() / 60)}m ago"
            else:
                time_str = f"{int(time_ago.total_seconds() / 3600)}h ago"
            
            st.metric(
                label="🔄 Data Freshness",
                value=time_str,
                help=f"Data last updated: {updated_dt.strftime('%Y-%m-%d %H:%M')}"
            )
        else:
            st.metric(label="🔄 Data Freshness", value="Unknown")


def display_language_distribution_chart(dashboard_data: Dict[str, Any]):
    """Display language distribution as a pie chart"""
    lang_dist = dashboard_data.get('language_distribution', {})
    
    if not lang_dist:
        st.info("📊 No language distribution data available")
        return
    
    st.markdown("### 🌍 Language Distribution")
    
    # Prepare data for chart
    languages = list(lang_dist.keys())
    counts = list(lang_dist.values())
    
    # Create pie chart
    fig = px.pie(
        values=counts,
        names=languages,
        title="Questions by Language",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Questions: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed breakdown
    with st.expander("📋 Detailed Language Breakdown"):
        lang_df = pd.DataFrame([
            {'Language': lang.upper(), 'Questions': count, 'Percentage': f"{(count/sum(counts)*100):.1f}%"} 
            for lang, count in sorted(lang_dist.items(), key=lambda x: x[1], reverse=True)
        ])
        st.dataframe(lang_df, use_container_width=True, hide_index=True)


def display_topic_frequency_chart(dashboard_data: Dict[str, Any]):
    """Display top topics by frequency"""
    topic_dist = dashboard_data.get('topic_distribution', {})
    
    if not topic_dist:
        st.info("📊 No topic distribution data available")
        return
    
    st.markdown("### 🏷️ Top Topics by Frequency")
    
    # Sort topics by frequency and take top 15
    sorted_topics = sorted(topic_dist.items(), key=lambda x: x[1], reverse=True)[:15]
    
    if not sorted_topics:
        st.info("No topic data available")
        return
    
    topics = [item[0] for item in sorted_topics]
    counts = [item[1] for item in sorted_topics]
    
    # Create horizontal bar chart
    fig = px.bar(
        x=counts,
        y=topics,
        orientation='h',
        title="Most Frequently Asked Topics",
        labels={'x': 'Number of Questions', 'y': 'Topics'},
        color=counts,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=max(400, len(topics) * 30),
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        title_x=0.5
    )
    
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Questions: %{x}<extra></extra>'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed table
    with st.expander("📋 All Topics Breakdown"):
        topic_df = pd.DataFrame([
            {'Topic': topic, 'Questions': count} 
            for topic, count in sorted(topic_dist.items(), key=lambda x: x[1], reverse=True)
        ])
        st.dataframe(topic_df, use_container_width=True, hide_index=True)


def display_sentiment_overview(dashboard_data: Dict[str, Any]):
    """Display sentiment analysis overview"""
    sentiment_summary = dashboard_data.get('sentiment_summary', {})
    
    if not sentiment_summary or 'error' in sentiment_summary:
        st.info("📊 Sentiment analysis data not available")
        return
    
    st.markdown("### 😊 Sentiment Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        avg_sentiment = sentiment_summary.get('avg_sentiment', 0)
        
        # Convert sentiment score to label and color
        if avg_sentiment > 0.1:
            sentiment_label = "Positive 😊"
            sentiment_color = "green"
        elif avg_sentiment < -0.1:
            sentiment_label = "Negative 😟"
            sentiment_color = "red"
        else:
            sentiment_label = "Neutral 😐"
            sentiment_color = "blue"
        
        st.metric(
            label="📊 Overall Sentiment",
            value=sentiment_label,
            delta=f"Score: {avg_sentiment:.3f}",
            help="Average sentiment score across all questions (-1 = very negative, +1 = very positive)"
        )
    
    with col2:
        urgent_count = sentiment_summary.get('urgent_topics_count', 0)
        st.metric(
            label="🚨 Urgent Topics",
            value=urgent_count,
            help="Number of topics with high urgency scores requiring attention"
        )
    
    # Show urgent topics if any
    urgent_topics = sentiment_summary.get('urgency_topics', [])
    if urgent_topics:
        st.markdown("#### 🚨 Topics Requiring Attention")
        
        for topic in urgent_topics[:5]:  # Show top 5
            cluster_id = topic.get('cluster_id', 'Unknown')
            urgency_score = topic.get('urgency_score', 0)
            avg_sentiment = topic.get('avg_sentiment', 0)
            
            with st.expander(f"🚨 Cluster {cluster_id} (Urgency: {urgency_score:.2f})"):
                st.write(f"**Average Sentiment:** {avg_sentiment:.3f}")
                st.write(f"**Urgency Score:** {urgency_score:.3f}")
                st.info("This topic shows patterns indicating student confusion or frustration.")


def display_trends_overview(dashboard_data: Dict[str, Any]):
    """Display trends analysis overview"""
    trend_summary = dashboard_data.get('trend_summary', {})
    
    if not trend_summary or 'error' in trend_summary:
        st.info("📊 Trend analysis data not available")
        return
    
    st.markdown("### 📈 Trending Topics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Trending Up")
        trending_up = trend_summary.get('trending_up', [])
        
        if trending_up:
            for i, topic in enumerate(trending_up[:5], 1):
                st.write(f"{i}. {topic}")
        else:
            st.info("No upward trending topics detected")
    
    with col2:
        st.markdown("#### 📉 Trending Down")
        trending_down = trend_summary.get('trending_down', [])
        
        if trending_down:
            for i, topic in enumerate(trending_down[:5], 1):
                st.write(f"{i}. {topic}")
        else:
            st.info("No downward trending topics detected")


def display_analysis_status_widget():
    """Display analysis status in sidebar or as widget"""
    try:
        data_service = get_data_service()
        status = data_service.get_analysis_status()
        
        latest_run = status.get('latest_run', {})
        
        if latest_run:
            run_status = latest_run.get('status', 'unknown')
            
            if run_status == 'completed':
                completed_at = latest_run.get('completed_at')
                if completed_at:
                    completed_dt = pd.to_datetime(completed_at)
                    st.success(f"✅ Analysis up to date (completed {completed_dt.strftime('%Y-%m-%d %H:%M')})")
                else:
                    st.success("✅ Analysis completed")
            elif run_status == 'running':
                st.info("🔄 Analysis currently running...")
            elif run_status == 'failed':
                st.error("❌ Last analysis failed")
            else:
                st.warning(f"⚠️ Analysis status: {run_status}")
        else:
            st.warning("⚠️ No analysis runs found")
            
    except Exception as e:
        st.error(f"❌ Error checking analysis status: {e}")


def display_data_freshness_indicator(dashboard_data: Dict[str, Any]):
    """Display data freshness indicator"""
    last_updated = dashboard_data.get('last_updated')
    
    if not last_updated:
        st.warning("⚠️ Data freshness unknown")
        return
    
    try:
        updated_dt = pd.to_datetime(last_updated)
        now = datetime.now()
        time_diff = now - updated_dt.tz_localize(None)
        
        if time_diff.total_seconds() < 1800:  # Less than 30 minutes
            st.success(f"🟢 Data is fresh (updated {int(time_diff.total_seconds() / 60)} minutes ago)")
        elif time_diff.total_seconds() < 3600:  # Less than 1 hour
            st.info(f"🟡 Data is recent (updated {int(time_diff.total_seconds() / 60)} minutes ago)")
        else:
            hours_ago = int(time_diff.total_seconds() / 3600)
            st.warning(f"🟠 Data may be stale (updated {hours_ago} hours ago)")
            
    except Exception as e:
        st.error(f"❌ Error parsing data timestamp: {e}")


def display_help_and_interpretation():
    """Display help information for understanding charts and metrics"""
    with st.expander("ℹ️ How to Interpret This Dashboard"):
        st.markdown("""
        ### 📊 Dashboard Guide
        
        **Key Metrics:**
        - **Total Questions:** Number of student questions in the selected time period
        - **Topics Discovered:** Unique topic clusters identified by AI analysis
        - **Primary Language:** Most common language used in questions
        - **Data Freshness:** How recently the analysis was last updated
        
        **Language Distribution:**
        - Shows the variety of languages students use to ask questions
        - Helps identify need for multilingual support
        
        **Topic Frequency:**
        - Topics that students ask about most frequently
        - Higher frequency may indicate areas needing better documentation or support
        
        **Sentiment Analysis:**
        - Overall emotional tone of student questions
        - Urgent topics may indicate confusion or frustration that needs attention
        
        **Trending Topics:**
        - Topics becoming more or less popular over time
        - Helps identify emerging issues or improving areas
        
        ### 🔄 Data Updates
        - Dashboard data is pre-computed and updated when developers run analysis
        - Use the time period selector to view different date ranges
        - Check data freshness indicator to see when data was last updated
        """)


def display_export_options(dashboard_data: Dict[str, Any]):
    """Display options to export dashboard data"""
    st.markdown("### 📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Metrics", use_container_width=True):
            metrics_df = pd.DataFrame([dashboard_data.get('metrics', {})])
            csv = metrics_df.to_csv(index=False)
            st.download_button(
                label="Download Metrics CSV",
                data=csv,
                file_name=f"metrics_{dashboard_data.get('time_period', 'all')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🗣️ Export Languages", use_container_width=True):
            lang_data = dashboard_data.get('language_distribution', {})
            if lang_data:
                lang_df = pd.DataFrame(list(lang_data.items()), columns=['Language', 'Count'])
                csv = lang_df.to_csv(index=False)
                st.download_button(
                    label="Download Language CSV",
                    data=csv,
                    file_name=f"languages_{dashboard_data.get('time_period', 'all')}.csv",
                    mime="text/csv"
                )
    
    with col3:
        if st.button("🏷️ Export Topics", use_container_width=True):
            topic_data = dashboard_data.get('topic_distribution', {})
            if topic_data:
                topic_df = pd.DataFrame(list(topic_data.items()), columns=['Topic', 'Count'])
                csv = topic_df.to_csv(index=False)
                st.download_button(
                    label="Download Topics CSV",
                    data=csv,
                    file_name=f"topics_{dashboard_data.get('time_period', 'all')}.csv",
                    mime="text/csv"
                )


def display_full_dashboard():
    """Display the complete dashboard with smart caching integration"""
    # Header
    display_dashboard_header()
    
    # Check if we have cached analysis data from startup
    if st.session_state.get('cached_analysis') and st.session_state.get('startup_data_loaded'):
        # Display cached analysis info
        cached_analysis = st.session_state['cached_analysis']
        
        st.info(f"""
        📊 **Displaying Cached Analysis** (Run ID: `{cached_analysis['run_id'][:8]}...`)
        - **Completed:** {cached_analysis['completed_at'][:19]}
        - **Questions:** {cached_analysis['total_questions']}
        - **Topics:** {cached_analysis['total_topics']}
        """)
        
        # Option to load fresh data
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 **Load Fresh Data**", type="secondary"):
                # Clear cached analysis and reload
                if 'cached_analysis' in st.session_state:
                    del st.session_state['cached_analysis']
                if 'startup_data_loaded' in st.session_state:
                    del st.session_state['startup_data_loaded']
                st.rerun()
    
    # Time period selector
    selected_period = display_time_period_selector()
    
    # Load data for selected period (either from cache or fresh from DB)
    if st.session_state.get('cached_analysis') and st.session_state.get('startup_data_loaded'):
        # Use cached data (faster)
        dashboard_data = load_dashboard_data_from_cache(st.session_state['cached_analysis'], selected_period)
    else:
        # Load fresh from database
        dashboard_data = load_dashboard_data(selected_period)
    
    if 'error' in dashboard_data:
        st.error(f"❌ Failed to load dashboard data: {dashboard_data['error']}")
        
        # Offer alternative data loading
        st.markdown("### 🔧 Alternative Data Loading")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 **Try Load from Cache**"):
                try:
                    from streamlit_data_manager import data_manager
                    latest_analysis = data_manager.load_latest_analysis_from_database()
                    if latest_analysis:
                        st.session_state['cached_analysis'] = latest_analysis
                        st.session_state['startup_data_loaded'] = True
                        st.success("✅ Analysis loaded from cache!")
                        st.rerun()
                    else:
                        st.warning("No cached analysis available")
                except Exception as e:
                    st.error(f"Cache loading failed: {str(e)}")
        
        with col2:
            if st.button("🚀 **Run New Analysis**"):
                st.info("💡 Use Developer Mode in the sidebar to run a new analysis")
        
        return
    
    # Analysis status
    display_analysis_status_widget()
    
    # Data freshness
    display_data_freshness_indicator(dashboard_data)
    
    # Key metrics
    display_key_metrics(dashboard_data)
    
    st.markdown("---")
    
    # Charts and visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        display_language_distribution_chart(dashboard_data)
    
    with col2:
        display_sentiment_overview(dashboard_data)
    
    # Topic frequency chart (full width)
    display_topic_frequency_chart(dashboard_data)
    
    # Trends overview
    display_trends_overview(dashboard_data)
    
    st.markdown("---")
    
    # Export options
    display_export_options(dashboard_data)
    
    # Help section
    display_help_and_interpretation()


def load_dashboard_data_from_cache(cached_analysis: Dict[str, Any], time_period: str) -> Dict[str, Any]:
    """Load dashboard data from cached analysis results"""
    try:
        # Extract data from cached analysis
        cached_results = cached_analysis.get('cached_results', {})
        
        # Convert cached data to dashboard format
        dashboard_data = {
            'time_period': time_period,
            'total_questions': cached_analysis.get('total_questions', 0),
            'total_topics': cached_analysis.get('total_topics', 0),
            'last_analysis': cached_analysis.get('completed_at'),
            'run_id': cached_analysis.get('run_id'),
            
            # Mock data for charts (replace with actual cached data when available)
            'questions_by_language': {'en': cached_analysis.get('total_questions', 0)},
            'sentiment_distribution': {'neutral': cached_analysis.get('total_questions', 0)},
            'topic_frequencies': {},
            'recent_questions': [],
            'trends_data': {},
            'data_source': 'cache'
        }
        
        return dashboard_data
        
    except Exception as e:
        return {'error': f"Failed to load cached data: {str(e)}"}


# Add this function at the top of the file, after the imports
def get_cached_analysis_status():
    """Get status of cached analysis"""
    if st.session_state.get('cached_analysis') and st.session_state.get('startup_data_loaded'):
        cached_analysis = st.session_state['cached_analysis']
        return {
            'has_cache': True,
            'run_id': cached_analysis.get('run_id'),
            'completed_at': cached_analysis.get('completed_at'),
            'total_questions': cached_analysis.get('total_questions'),
            'total_topics': cached_analysis.get('total_topics')
        }
    else:
        return {'has_cache': False}