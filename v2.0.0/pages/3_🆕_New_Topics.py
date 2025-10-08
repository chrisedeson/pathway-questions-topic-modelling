"""
New Topics Page - Explore newly discovered topics from clustering
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COLUMN_DISPLAY_NAMES
from utils.data_loader import export_to_csv


def main():
    st.title("🆕 New Topics Discovery")
    st.markdown("*Explore newly discovered topics identified through clustering analysis*")
    st.markdown("---")
    
    # Check if data is loaded
    if 'raw_data' not in st.session_state:
        st.error("❌ No data loaded. Please return to the home page.")
        st.stop()
    
    raw_data = st.session_state['raw_data']
    
    if 'new_topics' not in raw_data or raw_data['new_topics'].empty:
        st.info("""
        ### ℹ️ No new topics discovered
        
        The clustering analysis did not identify any new topics in the latest run.
        This could mean:
        - All questions matched existing topics well
        - The similarity threshold is set appropriately
        - Your topic taxonomy is comprehensive
        
        💡 **Tip:** Check the Questions Table to see how questions are classified.
        """)
        st.stop()
    
    new_topics_df = raw_data['new_topics'].copy()
    
    # Overview metrics
    st.markdown("## 📊 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    unique_clusters = new_topics_df['cluster_id'].nunique() if 'cluster_id' in new_topics_df.columns else 0
    total_questions = len(new_topics_df)
    avg_per_cluster = total_questions / unique_clusters if unique_clusters > 0 else 0
    
    with col1:
        st.metric("🆕 New Topics Found", unique_clusters)
    
    with col2:
        st.metric("📝 Total Questions", total_questions)
    
    with col3:
        st.metric("📊 Avg Questions/Topic", f"{avg_per_cluster:.1f}")
    
    with col4:
        if 'topic_name' in new_topics_df.columns:
            named_topics = new_topics_df['topic_name'].notna().sum()
            st.metric("✅ Named Topics", named_topics)
    
    st.markdown("---")
    
    # Topic exploration
    st.markdown("## 🔍 Explore New Topics")
    
    if 'cluster_id' not in new_topics_df.columns:
        st.error("⚠️ Cluster ID information not available in the data.")
        st.stop()
    
    # Get unique clusters
    clusters = sorted(new_topics_df['cluster_id'].unique())
    
    # Topic selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_cluster = st.selectbox(
            "Select a topic to explore",
            clusters,
            format_func=lambda x: f"Topic {x} ({len(new_topics_df[new_topics_df['cluster_id'] == x])} questions)",
            help="Choose a new topic to see its details and representative questions"
        )
    
    with col2:
        # Sort options
        sort_option = st.selectbox(
            "Sort questions by",
            ["Most Recent", "Oldest First"],
            help="Sort questions within the selected topic"
        )
    
    st.markdown("---")
    
    # Filter to selected cluster
    cluster_df = new_topics_df[new_topics_df['cluster_id'] == selected_cluster].copy()
    
    # Sort
    if 'timestamp' in cluster_df.columns:
        ascending = (sort_option == "Oldest First")
        cluster_df = cluster_df.sort_values('timestamp', ascending=ascending)
    
    # Topic details
    st.markdown(f"### 📌 Topic {selected_cluster} Details")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Topic name and keywords
        if 'topic_name' in cluster_df.columns:
            topic_name = cluster_df['topic_name'].iloc[0]
            if pd.notna(topic_name):
                st.markdown(f"**Generated Topic Name:** {topic_name}")
        
        if 'topic_keywords' in cluster_df.columns:
            keywords = cluster_df['topic_keywords'].iloc[0]
            if pd.notna(keywords):
                st.markdown(f"**Keywords:** {keywords}")
    
    with col2:
        st.metric("Questions in this topic", len(cluster_df))
        
        if 'country' in cluster_df.columns:
            unique_countries = cluster_df['country'].nunique()
            st.metric("Countries represented", unique_countries)
    
    st.markdown("---")
    
    # Representative questions
    st.markdown("### 📝 Questions in This Topic")
    
    # Display options
    with st.expander("🎛️ Display Options"):
        columns_to_show = st.multiselect(
            "Select columns to display",
            [col for col in cluster_df.columns if col != 'cluster_id'],
            default=['input', 'timestamp', 'country'] if 'input' in cluster_df.columns else cluster_df.columns[:3].tolist(),
            format_func=lambda x: COLUMN_DISPLAY_NAMES.get(x, x)
        )
    
    if columns_to_show:
        st.dataframe(
            cluster_df[columns_to_show],
            use_container_width=True,
            height=400,
            hide_index=True
        )
    
    # Export button
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = export_to_csv(cluster_df)
        st.download_button(
            label=f"📥 Download Topic {selected_cluster} (CSV)",
            data=csv_data,
            file_name=f"new_topic_{selected_cluster}.csv",
            mime="text/csv",
            help="Download questions from this topic"
        )
    
    with col2:
        csv_all = export_to_csv(new_topics_df)
        st.download_button(
            label="📥 Download All New Topics (CSV)",
            data=csv_all,
            file_name="all_new_topics.csv",
            mime="text/csv",
            help="Download all questions from new topics"
        )
    
    st.markdown("---")
    
    # Topic summary table
    st.markdown("## 📊 All New Topics Summary")
    
    # Create summary table
    summary_data = []
    for cluster_id in clusters:
        cluster_data = new_topics_df[new_topics_df['cluster_id'] == cluster_id]
        
        summary = {
            'Topic ID': cluster_id,
            'Questions': len(cluster_data),
            'Topic Name': cluster_data['topic_name'].iloc[0] if 'topic_name' in cluster_data.columns else 'N/A'
        }
        
        if 'topic_keywords' in cluster_data.columns:
            keywords = cluster_data['topic_keywords'].iloc[0]
            summary['Keywords'] = keywords if pd.notna(keywords) else 'N/A'
        
        if 'country' in cluster_data.columns:
            summary['Countries'] = cluster_data['country'].nunique()
        
        summary_data.append(summary)
    
    summary_df = pd.DataFrame(summary_data)
    
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Topic ID': st.column_config.NumberColumn('Topic ID', width='small'),
            'Questions': st.column_config.NumberColumn('Questions', width='small'),
            'Topic Name': st.column_config.TextColumn('Topic Name', width='large'),
            'Keywords': st.column_config.TextColumn('Keywords', width='large'),
            'Countries': st.column_config.NumberColumn('Countries', width='small')
        }
    )
    
    # Insights and recommendations
    st.markdown("---")
    
    with st.expander("💡 Insights & Recommendations"):
        st.markdown(f"""
        ### 🔍 Analysis Summary
        
        **Discovery Statistics:**
        - Identified **{unique_clusters}** new potential topics
        - Covering **{total_questions}** questions that didn't match existing topics well
        - Average of **{avg_per_cluster:.1f}** questions per new topic
        
        **Recommendations:**
        1. **Review each topic** to determine if it should be added to the official taxonomy
        2. **Look for patterns** in the keywords and questions
        3. **Consider geographic factors** - some topics may be region-specific
        4. **Merge similar topics** if multiple clusters represent the same underlying theme
        5. **Update the Google Sheet** with approved topics for future runs
        
        **Next Steps:**
        - Download topics for team review
        - Discuss with subject matter experts
        - Update the topic taxonomy as needed
        - Re-run the analysis to validate improvements
        """)
    
    # Tips
    st.markdown("---")
    st.info("""
    ### 💡 Tips for Topic Review
    
    - **Read multiple questions** from each topic to understand the theme
    - **Check the keywords** to see what terms define the topic
    - **Consider frequency** - topics with more questions may need priority
    - **Geographic distribution** - some topics may be region-specific
    - **Compare with existing topics** - ensure new topics are truly unique
    - **Download data** for offline review and team discussion
    """)


if __name__ == "__main__":
    main()
