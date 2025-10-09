"""
New Topics Page - Explore newly discovered topics from clustering
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COLUMN_DISPLAY_NAMES
from utils.data_loader import export_to_csv, ensure_data_loaded


def main():
    st.title("🆕 New Topics Discovery")
    st.markdown("*Explore newly discovered topics identified through clustering analysis*")
    st.markdown("---")
    
    # Ensure data is loaded (handles page refresh)
    ensure_data_loaded()
    
    raw_data = st.session_state['raw_data']
    merged_df = st.session_state['merged_df']
    
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
    
    # new_topics has: topic_name, representative_question, question_count
    new_topics_df = raw_data['new_topics'].copy()
    
    # Get all questions classified as 'new' from merged_df
    new_questions_df = merged_df[merged_df['classification'] == 'new'].copy()
    
    # Overview metrics
    st.markdown("## 📊 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    unique_topics = len(new_topics_df)
    total_questions = len(new_questions_df)
    avg_per_topic = total_questions / unique_topics if unique_topics > 0 else 0
    
    with col1:
        st.metric("🆕 New Topics Found", unique_topics)
    
    with col2:
        st.metric("📝 Total Questions", total_questions)
    
    with col3:
        st.metric("📊 Avg Questions/Topic", f"{avg_per_topic:.1f}")
    
    with col4:
        if 'topic_name' in new_topics_df.columns:
            named_topics = new_topics_df['topic_name'].notna().sum()
            st.metric("✅ Named Topics", named_topics)
    
    st.markdown("---")
    
    # Topic exploration
    st.markdown("## 🔍 Explore New Topics")
    
    if new_topics_df.empty:
        st.info("No new topics available to explore.")
        st.stop()
    
    # Add an index column for selection
    new_topics_df['topic_id'] = range(1, len(new_topics_df) + 1)
    
    # Topic selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create selection options with topic name and question count
        topic_options = {}
        for idx, row in new_topics_df.iterrows():
            topic_name = row.get('topic_name', f"Topic {row['topic_id']}")
            question_count = row.get('question_count', 0)
            topic_options[row['topic_id']] = f"{topic_name} ({question_count} questions)"
        
        selected_topic_id = st.selectbox(
            "Select a topic to explore",
            list(topic_options.keys()),
            format_func=lambda x: topic_options[x],
            help="Choose a new topic to see its details"
        )
    
    with col2:
        # Sort options
        sort_option = st.selectbox(
            "Sort by",
            ["Most Questions", "Fewest Questions", "Alphabetical"],
            help="Sort topics by different criteria"
        )
    
    # Apply sorting to the topics list
    if sort_option == "Most Questions" and 'question_count' in new_topics_df.columns:
        new_topics_df = new_topics_df.sort_values('question_count', ascending=False)
    elif sort_option == "Fewest Questions" and 'question_count' in new_topics_df.columns:
        new_topics_df = new_topics_df.sort_values('question_count', ascending=True)
    elif sort_option == "Alphabetical" and 'topic_name' in new_topics_df.columns:
        new_topics_df = new_topics_df.sort_values('topic_name')
    
    st.markdown("---")
    
    # Get selected topic details
    selected_topic = new_topics_df[new_topics_df['topic_id'] == selected_topic_id].iloc[0]
    
    # Topic details
    st.markdown(f"### 📌 {selected_topic.get('topic_name', f'Topic {selected_topic_id}')} Details")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Representative question
        if 'representative_question' in selected_topic:
            rep_question = selected_topic['representative_question']
            if pd.notna(rep_question):
                st.markdown("**Representative Question:**")
                st.info(rep_question)
    
    with col2:
        if 'question_count' in selected_topic:
            st.metric("Questions in this topic", int(selected_topic['question_count']))
    
    st.markdown("---")
    
    # Show questions that belong to this topic from merged_df
    st.markdown("### 📝 Questions in This Topic")
    
    topic_name_to_find = selected_topic.get('topic_name', f'Topic {selected_topic_id}')
    topic_questions = new_questions_df[new_questions_df['topic_name'] == topic_name_to_find].copy()
    
    if not topic_questions.empty:
        st.dataframe(
            topic_questions,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        # Export this topic
        st.markdown("---")
        csv_data = export_to_csv(topic_questions)
        st.download_button(
            label=f"📥 Download This Topic (CSV)",
            data=csv_data,
            file_name=f"new_topic_{selected_topic_id}.csv",
            mime="text/csv",
            help="Download questions from this topic"
        )
    else:
        st.info(f"No detailed questions found for this topic. Showing summary only.")
    
    st.markdown("---")
    
    # Topic summary table
    st.markdown("## 📊 All New Topics Summary")
    
    # Display the summary table
    display_df = new_topics_df[['topic_name', 'representative_question', 'question_count']].copy()
    display_df.columns = ['Topic Name', 'Representative Question', 'Questions']
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Topic Name': st.column_config.TextColumn('Topic Name', width='medium'),
            'Representative Question': st.column_config.TextColumn('Representative Question', width='large'),
            'Questions': st.column_config.NumberColumn('Questions', width='small')
        }
    )
    
    # Export all new topics
    col1, col2 = st.columns(2)
    
    with col1:
        csv_all_summary = export_to_csv(new_topics_df)
        st.download_button(
            label="📥 Download Topics Summary (CSV)",
            data=csv_all_summary,
            file_name="new_topics_summary.csv",
            mime="text/csv",
            help="Download summary of all new topics"
        )
    
    with col2:
        if not new_questions_df.empty:
            csv_all_questions = export_to_csv(new_questions_df)
            st.download_button(
                label="📥 Download All New Questions (CSV)",
                data=csv_all_questions,
                file_name="all_new_topic_questions.csv",
                mime="text/csv",
                help="Download all questions from new topics"
            )
    
    # Insights and recommendations
    st.markdown("---")
    
    with st.expander("💡 Insights & Recommendations"):
        st.markdown(f"""
        ### 🔍 Analysis Summary
        
        **Discovery Statistics:**
        - Identified **{unique_topics}** new potential topics
        - Covering **{total_questions}** questions that didn't match existing topics well
        - Average of **{avg_per_topic:.1f}** questions per new topic
        
        **Recommendations:**
        1. **Review each topic** to determine if it should be added to the official taxonomy
        2. **Look for patterns** in the representative questions
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
    - **Check the representative question** to see what defines the topic
    - **Consider frequency** - topics with more questions may need priority
    - **Compare with existing topics** - ensure new topics are truly unique
    - **Download data** for offline review and team discussion
    """)


if __name__ == "__main__":
    main()
