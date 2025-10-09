"""
Trends & Analytics Page - Detailed visualizations and insights
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.visualizations import (
    create_kpi_cards, plot_classification_distribution, plot_country_distribution,
    plot_timeline, plot_similarity_distribution, plot_top_topics,
    plot_hourly_heatmap, plot_language_distribution
)


def main():
    st.title("📈 Trends & Analytics")
    st.markdown("*Deep dive into patterns, trends, and insights from student questions*")
    st.markdown("---")
    
    # Check if data is loaded
    if 'merged_df' not in st.session_state or 'kpis' not in st.session_state:
        st.error("❌ No data loaded. Please return to the home page.")
        st.stop()
    
    df = st.session_state['merged_df']
    kpis = st.session_state['kpis']
    
    # KPI Cards
    st.markdown("## 📊 Key Performance Indicators")
    create_kpi_cards(kpis)
    
    st.markdown("---")
    
    # Main visualizations in tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🌍 Geographic Insights",
        "⏰ Temporal Patterns",
        "🎯 Topic Analysis"
    ])
    
    with tab1:
        st.markdown("### 📊 Overall Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            plot_classification_distribution(df)
        
        with col2:
            plot_similarity_distribution(df)
        
        st.markdown("---")
        
        st.markdown("### 📝 Question Volume Trends")
        plot_timeline(df)
        
        st.markdown("---")
        
        # Summary insights
        with st.expander("💡 Key Insights"):
            existing_count = len(df[df['classification'] == 'Existing Topic'])
            new_count = len(df[df['classification'] == 'New Topic'])
            total = len(df)
            
            existing_pct = (existing_count / total * 100) if total > 0 else 0
            new_pct = (new_count / total * 100) if total > 0 else 0
            
            st.markdown(f"""
            **Classification Summary:**
            - **{existing_pct:.1f}%** of questions matched existing topics
            - **{new_pct:.1f}%** of questions formed new topics
            - This suggests the current topic taxonomy covers most student questions effectively
            
            **Similarity Scores:**
            - Average similarity score: **{kpis['avg_similarity']:.3f}**
            - Higher scores indicate better matches to existing topics
            """)
    
    with tab2:
        st.markdown("### 🌍 Geographic Distribution")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            plot_country_distribution(df, top_n=15)
        
        with col2:
            if 'country' in df.columns:
                st.markdown("#### 📍 Country Statistics")
                country_stats = df['country'].value_counts().head(10)
                
                for i, (country, count) in enumerate(country_stats.items(), 1):
                    pct = (count / len(df) * 100)
                    st.markdown(f"{i}. **{country}**: {count:,} ({pct:.1f}%)")
        
        st.markdown("---")
        
        # Language distribution if available
        if 'user_language' in df.columns:
            st.markdown("### 🌐 Language Distribution")
            plot_language_distribution(df)
        
        with st.expander("💡 Geographic Insights"):
            if 'country' in df.columns:
                unique_countries = df['country'].nunique()
                mode_result = df['country'].mode()
                top_country = mode_result[0] if len(mode_result) > 0 else "N/A"
                top_country_count = df['country'].value_counts().iloc[0] if not df['country'].empty else 0
                top_country_pct = (top_country_count / len(df) * 100) if len(df) > 0 else 0
                
                st.markdown(f"""
                **Geographic Reach:**
                - Questions from **{unique_countries}** different countries
                - Top country: **{top_country}** ({top_country_count:,} questions, {top_country_pct:.1f}%)
                - This demonstrates the global reach of BYU Pathway
                """)
    
    with tab3:
        st.markdown("### ⏰ Temporal Patterns")
        
        # Timeline
        plot_timeline(df, key="timeline_temporal_tab")
        
        st.markdown("---")
        
        # Heatmap
        st.markdown("### 🗓️ Activity Heatmap")
        plot_hourly_heatmap(df)
        
        st.markdown("---")
        
        with st.expander("💡 Temporal Insights"):
            if 'timestamp' in df.columns:
                df_copy = df.copy()
                df_copy['hour'] = df_copy['timestamp'].dt.hour
                df_copy['day_of_week'] = df_copy['timestamp'].dt.day_name()
                
                # Find peak hour
                if not df_copy['hour'].isna().all():
                    hour_mode = df_copy['hour'].mode()
                    day_mode = df_copy['day_of_week'].mode()
                    
                    if len(hour_mode) > 0 and len(day_mode) > 0:
                        peak_hour = hour_mode[0]
                        peak_day = day_mode[0]
                        
                        st.markdown(f"""
                        **Activity Patterns:**
                        - Most questions are asked around **{peak_hour}:00**
                        - Most active day: **{peak_day}**
                    - Use this information to optimize support staff scheduling
                    """)
    
    with tab4:
        st.markdown("### 🎯 Topic Analysis")
        
        # Top topics
        plot_top_topics(df, top_n=15)
        
        st.markdown("---")
        
        # Topic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            if 'matched_topic' in df.columns:
                st.markdown("#### 📋 Topic Coverage")
                existing_topics = df[df['classification'] == 'Existing Topic']
                if not existing_topics.empty and 'matched_topic' in existing_topics.columns:
                    unique_topics = existing_topics['matched_topic'].nunique()
                    st.metric("Unique Topics Matched", unique_topics)
                    
                    # Most common topic
                    mode_result = existing_topics['matched_topic'].mode()
                    if len(mode_result) > 0:
                        top_topic = mode_result[0]
                        top_topic_count = existing_topics['matched_topic'].value_counts().iloc[0]
                        st.metric("Most Common Topic", top_topic)
                        st.metric("Questions in Top Topic", top_topic_count)
        
        with col2:
            if 'new_topics' in st.session_state.get('raw_data', {}):
                st.markdown("#### 🆕 New Topics")
                new_topics_df = st.session_state['raw_data']['new_topics']
                if not new_topics_df.empty and 'cluster_id' in new_topics_df.columns:
                    unique_clusters = new_topics_df['cluster_id'].nunique()
                    st.metric("New Topics Discovered", unique_clusters)
                    
                    avg_questions_per_cluster = len(new_topics_df) / unique_clusters
                    st.metric("Avg Questions per New Topic", f"{avg_questions_per_cluster:.1f}")
        
        with st.expander("💡 Topic Insights"):
            st.markdown("""
            **Topic Analysis Summary:**
            - Review the most common topics to understand student priorities
            - New topics indicate emerging themes that may need attention
            - Consider adding frequently appearing new topics to the official taxonomy
            - Use topic distribution to allocate resources effectively
            """)
    
    # Footer
    st.markdown("---")
    st.info("""
    ### 💡 How to Use These Analytics
    
    - **Track trends** over time to identify seasonal patterns
    - **Monitor geographic distribution** to understand global reach
    - **Analyze peak times** for better resource allocation
    - **Review top topics** to prioritize content creation
    - **Export data** from the Questions Table page for deeper analysis
    """)


if __name__ == "__main__":
    main()
