"""
Topic management interface components for BYU Pathway Hybrid Topic Analysis
Allows users to view, edit, and manage topics and subtopics
"""
import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple
import time
import uuid

def create_topic_editor_ui(topics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an interface for editing topics, subtopics, and questions
    
    Args:
        topics_df: DataFrame with columns ['Topic', 'Subtopic', 'Question']
    
    Returns:
        Modified DataFrame with user changes
    """
    st.subheader("📝 Topic Editor")
    
    # Create tabs for different editing modes
    tab1, tab2, tab3 = st.tabs(["📊 View & Edit", "➕ Add New", "🗑️ Bulk Actions"])
    
    with tab1:
        edited_df = create_view_edit_interface(topics_df)
        return edited_df
    
    with tab2:
        return create_add_new_interface(topics_df)
        
    with tab3:
        return create_bulk_actions_interface(topics_df)

def create_view_edit_interface(topics_df: pd.DataFrame) -> pd.DataFrame:
    """Create the main view and edit interface"""
    
    if topics_df.empty:
        st.info("No topics data to display. Upload a file or connect to Google Sheets first.")
        return topics_df
    
    st.write("**💡 Tip**: Click on cells to edit them directly. Changes are applied immediately.")
    
    # Group by topic for better organization
    topics_grouped = topics_df.groupby('Topic')
    
    # Create expandable sections for each topic
    modified_rows = []
    
    for topic_name, topic_group in topics_grouped:
        with st.expander(f"📂 **{topic_name}** ({len(topic_group)} questions)", expanded=False):
            
            # Show subtopics for this topic
            subtopics = topic_group['Subtopic'].unique()
            st.write(f"**Subtopics**: {', '.join(subtopics)}")
            
            # Create editable data editor for this topic
            edited_group = st.data_editor(
                topic_group,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "Topic": st.column_config.SelectboxColumn(
                        "Topic",
                        options=topics_df['Topic'].unique().tolist(),
                        required=True
                    ),
                    "Subtopic": st.column_config.TextColumn(
                        "Subtopic",
                        help="Subtopic classification",
                        required=True
                    ),
                    "Question": st.column_config.TextColumn(
                        "Question", 
                        help="The actual question text",
                        required=True
                    )
                },
                key=f"editor_{topic_name}_{hash(topic_name)}"
            )
            
            modified_rows.append(edited_group)
    
    # Combine all modified groups
    if modified_rows:
        result_df = pd.concat(modified_rows, ignore_index=True)
    else:
        result_df = topics_df.copy()
    
    return result_df

def create_add_new_interface(topics_df: pd.DataFrame) -> pd.DataFrame:
    """Create interface for adding new topics/questions"""
    
    st.write("**Add new topics, subtopics, and questions to your collection.**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Existing topics for selection
        existing_topics = ["[Create New Topic]"] + sorted(topics_df['Topic'].unique().tolist())
        selected_topic = st.selectbox("**Topic**", existing_topics)
        
        if selected_topic == "[Create New Topic]":
            new_topic = st.text_input("**New Topic Name**", placeholder="Enter new topic name")
            topic_to_use = new_topic
        else:
            topic_to_use = selected_topic
    
    with col2:
        # Subtopics based on selected topic
        if selected_topic != "[Create New Topic]" and selected_topic:
            existing_subtopics = topics_df[topics_df['Topic'] == selected_topic]['Subtopic'].unique().tolist()
            existing_subtopics = ["[Create New Subtopic]"] + sorted(existing_subtopics)
        else:
            existing_subtopics = ["[Create New Subtopic]"]
        
        selected_subtopic = st.selectbox("**Subtopic**", existing_subtopics)
        
        if selected_subtopic == "[Create New Subtopic]":
            new_subtopic = st.text_input("**New Subtopic Name**", placeholder="Enter new subtopic name")
            subtopic_to_use = new_subtopic
        else:
            subtopic_to_use = selected_subtopic
    
    # Question input
    new_question = st.text_area(
        "**Question**",
        placeholder="Enter the question text here...",
        height=100
    )
    
    # Add button
    if st.button("➕ **Add New Entry**", type="primary"):
        if topic_to_use and subtopic_to_use and new_question:
            # Add new row to DataFrame
            new_row = pd.DataFrame([{
                'Topic': topic_to_use,
                'Subtopic': subtopic_to_use,
                'Question': new_question
            }])
            
            result_df = pd.concat([topics_df, new_row], ignore_index=True)
            st.success(f"✅ Added new entry to '{topic_to_use}' > '{subtopic_to_use}'")
            st.rerun()
        else:
            st.error("❌ Please fill in all fields")
    
    return topics_df

def create_bulk_actions_interface(topics_df: pd.DataFrame) -> pd.DataFrame:
    """Create interface for bulk actions like delete, move, etc."""
    
    st.write("**Perform bulk operations on your topics and questions.**")
    
    if topics_df.empty:
        st.info("No data available for bulk operations.")
        return topics_df
    
    # Action selection
    action = st.selectbox(
        "**Select Action**",
        ["Delete Questions", "Move Questions", "Rename Topic", "Rename Subtopic"]
    )
    
    if action == "Delete Questions":
        return create_delete_interface(topics_df)
    elif action == "Move Questions":
        return create_move_interface(topics_df)
    elif action == "Rename Topic":
        return create_rename_topic_interface(topics_df)
    elif action == "Rename Subtopic":
        return create_rename_subtopic_interface(topics_df)
    
    return topics_df

def create_delete_interface(topics_df: pd.DataFrame) -> pd.DataFrame:
    """Interface for deleting questions"""
    
    st.write("**⚠️ Select questions to delete**")
    
    # Topic filter
    topic_filter = st.selectbox(
        "**Filter by Topic** (optional)",
        ["All Topics"] + sorted(topics_df['Topic'].unique().tolist())
    )
    
    if topic_filter != "All Topics":
        filtered_df = topics_df[topics_df['Topic'] == topic_filter]
    else:
        filtered_df = topics_df
    
    # Show questions with checkboxes
    if len(filtered_df) > 0:
        questions_to_delete = []
        
        for idx, row in filtered_df.iterrows():
            key = f"delete_{idx}_{hash(row['Question'][:50])}"
            if st.checkbox(
                f"**{row['Topic']}** > **{row['Subtopic']}**: {row['Question'][:100]}...",
                key=key
            ):
                questions_to_delete.append(idx)
        
        if questions_to_delete:
            st.warning(f"⚠️ **{len(questions_to_delete)} questions** selected for deletion")
            
            if st.button("🗑️ **Delete Selected Questions**", type="secondary"):
                if st.session_state.get('confirm_delete', False):
                    # Remove selected rows
                    result_df = topics_df.drop(questions_to_delete).reset_index(drop=True)
                    st.success(f"✅ Deleted {len(questions_to_delete)} questions")
                    return result_df
                else:
                    st.session_state.confirm_delete = True
                    st.error("⚠️ **Click again to confirm deletion**")
    
    return topics_df

def create_move_interface(topics_df: pd.DataFrame) -> pd.DataFrame:
    """Interface for moving questions between topics/subtopics"""
    
    st.write("**Move questions to different topics or subtopics**")
    
    # Source topic/subtopic selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**From:**")
        source_topic = st.selectbox("Source Topic", topics_df['Topic'].unique().tolist())
        source_subtopics = topics_df[topics_df['Topic'] == source_topic]['Subtopic'].unique().tolist()
        source_subtopic = st.selectbox("Source Subtopic", source_subtopics)
    
    with col2:
        st.write("**To:**")
        target_topic = st.selectbox("Target Topic", topics_df['Topic'].unique().tolist())
        target_subtopics = topics_df[topics_df['Topic'] == target_topic]['Subtopic'].unique().tolist()
        target_subtopic = st.selectbox("Target Subtopic", target_subtopics)
    
    # Show questions to move
    source_questions = topics_df[
        (topics_df['Topic'] == source_topic) & 
        (topics_df['Subtopic'] == source_subtopic)
    ]
    
    if len(source_questions) > 0:
        questions_to_move = []
        
        st.write(f"**Questions in {source_topic} > {source_subtopic}:**")
        for idx, row in source_questions.iterrows():
            if st.checkbox(f"{row['Question'][:100]}...", key=f"move_{idx}"):
                questions_to_move.append(idx)
        
        if questions_to_move:
            if st.button("🔄 **Move Selected Questions**", type="primary"):
                # Update topic/subtopic for selected questions
                result_df = topics_df.copy()
                result_df.loc[questions_to_move, 'Topic'] = target_topic
                result_df.loc[questions_to_move, 'Subtopic'] = target_subtopic
                
                st.success(f"✅ Moved {len(questions_to_move)} questions to {target_topic} > {target_subtopic}")
                return result_df
    
    return topics_df

def create_rename_topic_interface(topics_df: pd.DataFrame) -> pd.DataFrame:
    """Interface for renaming topics"""
    
    current_topic = st.selectbox("**Select Topic to Rename**", topics_df['Topic'].unique().tolist())
    new_topic_name = st.text_input("**New Topic Name**", value=current_topic)
    
    if st.button("✏️ **Rename Topic**", type="primary") and new_topic_name != current_topic:
        result_df = topics_df.copy()
        result_df.loc[result_df['Topic'] == current_topic, 'Topic'] = new_topic_name
        st.success(f"✅ Renamed topic '{current_topic}' to '{new_topic_name}'")
        return result_df
    
    return topics_df

def create_rename_subtopic_interface(topics_df: pd.DataFrame) -> pd.DataFrame:
    """Interface for renaming subtopics"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_topic = st.selectbox("**Topic**", topics_df['Topic'].unique().tolist())
        subtopics = topics_df[topics_df['Topic'] == selected_topic]['Subtopic'].unique().tolist()
        current_subtopic = st.selectbox("**Select Subtopic to Rename**", subtopics)
    
    with col2:
        new_subtopic_name = st.text_input("**New Subtopic Name**", value=current_subtopic)
    
    if st.button("✏️ **Rename Subtopic**", type="primary") and new_subtopic_name != current_subtopic:
        result_df = topics_df.copy()
        mask = (result_df['Topic'] == selected_topic) & (result_df['Subtopic'] == current_subtopic)
        result_df.loc[mask, 'Subtopic'] = new_subtopic_name
        st.success(f"✅ Renamed subtopic '{current_subtopic}' to '{new_subtopic_name}' in topic '{selected_topic}'")
        return result_df
    
    return topics_df

def create_changes_confirmation_dialog(original_df: pd.DataFrame, modified_df: pd.DataFrame) -> bool:
    """
    Create a confirmation dialog showing what changes will be applied
    Returns True if user confirms, False otherwise
    """
    
    # Check if there are any changes
    if original_df.equals(modified_df):
        st.info("No changes detected.")
        return False
    
    # Calculate changes
    changes_summary = analyze_dataframe_changes(original_df, modified_df)
    
    if not changes_summary:
        st.info("No changes detected.")
        return False
    
    st.warning("⚠️ **Review Changes Before Applying**")
    
    with st.expander("📋 **Changes Summary**", expanded=True):
        for change_type, details in changes_summary.items():
            if details:
                st.write(f"**{change_type}**: {len(details)} items")
                for detail in details[:5]:  # Show first 5
                    st.write(f"  • {detail}")
                if len(details) > 5:
                    st.write(f"  • ... and {len(details) - 5} more")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("✅ **Apply Changes**", type="primary", use_container_width=True):
            return True
    
    with col2:
        if st.button("❌ **Cancel Changes**", use_container_width=True):
            st.rerun()
    
    return False

def analyze_dataframe_changes(original_df: pd.DataFrame, modified_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Analyze differences between two DataFrames and return summary of changes"""
    
    changes = {
        "Added Questions": [],
        "Removed Questions": [],
        "Modified Questions": [],
        "Topic Changes": [],
        "Subtopic Changes": []
    }
    
    # Compare questions
    original_questions = set(original_df['Question'].tolist())
    modified_questions = set(modified_df['Question'].tolist())
    
    # Added questions
    added_questions = modified_questions - original_questions
    changes["Added Questions"] = list(added_questions)
    
    # Removed questions
    removed_questions = original_questions - modified_questions
    changes["Removed Questions"] = list(removed_questions)
    
    # Modified questions (same question text but different topic/subtopic)
    common_questions = original_questions & modified_questions
    
    for question in common_questions:
        orig_row = original_df[original_df['Question'] == question].iloc[0]
        mod_row = modified_df[modified_df['Question'] == question].iloc[0]
        
        if orig_row['Topic'] != mod_row['Topic']:
            changes["Topic Changes"].append(
                f"'{question[:50]}...' moved from '{orig_row['Topic']}' to '{mod_row['Topic']}'"
            )
        
        if orig_row['Subtopic'] != mod_row['Subtopic']:
            changes["Subtopic Changes"].append(
                f"'{question[:50]}...' subtopic changed from '{orig_row['Subtopic']}' to '{mod_row['Subtopic']}'"
            )
    
    # Filter out empty changes
    changes = {k: v for k, v in changes.items() if v}
    
    return changes

def display_topic_statistics(topics_df: pd.DataFrame):
    """Display statistics about the topics data"""
    
    if topics_df.empty:
        st.info("No data to display statistics.")
        return
    
    st.subheader("📊 Topic Statistics")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("**Total Questions**", len(topics_df))
    
    with col2:
        st.metric("**Total Topics**", topics_df['Topic'].nunique())
    
    with col3:
        st.metric("**Total Subtopics**", topics_df['Subtopic'].nunique())
    
    with col4:
        avg_questions_per_topic = len(topics_df) / topics_df['Topic'].nunique() if topics_df['Topic'].nunique() > 0 else 0
        st.metric("**Avg Questions/Topic**", f"{avg_questions_per_topic:.1f}")
    
    # Topic distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Questions per Topic**")
        topic_counts = topics_df['Topic'].value_counts()
        st.bar_chart(topic_counts)
    
    with col2:
        st.write("**Top 10 Topics**")
        top_topics = topic_counts.head(10)
        for topic, count in top_topics.items():
            st.write(f"• **{topic}**: {count} questions")

def create_similarity_threshold_ui() -> float:
    """Create UI for configuring similarity threshold"""
    
    st.subheader("🎯 Similarity Threshold Configuration")
    
    with st.expander("💡 **What is Similarity Threshold?**", expanded=False):
        st.markdown("""
        **Similarity Threshold** determines how similar a new question must be to an existing topic question to be automatically classified.
        
        **How it works:**
        - Questions are compared using AI embeddings (semantic similarity)
        - Similarity scores range from 0.0 (completely different) to 1.0 (identical)
        - If similarity ≥ threshold → Question is classified into existing topic
        - If similarity < threshold → Question goes to clustering for new topic discovery
        
        **Threshold Guidelines:**
        - **0.85-0.95**: Very strict - only very similar questions match (fewer false positives)
        - **0.70-0.85**: Moderate - good balance of precision and recall 
        - **0.50-0.70**: Lenient - more questions match existing topics (more false positives)
        - **Below 0.50**: Too lenient - may group unrelated questions
        
        **Recommendation**: Start with 0.70 and adjust based on results.
        """)
    
    threshold = st.slider(
        "**Similarity Threshold**",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
        help="Questions with similarity above this threshold will be classified into existing topics"
    )
    
    # Visual indicator
    if threshold >= 0.85:
        st.info("🎯 **Strict**: Only very similar questions will match existing topics")
    elif threshold >= 0.70:
        st.success("✅ **Balanced**: Good balance of precision and recall (recommended)")
    else:
        st.warning("⚠️ **Lenient**: More questions will match, but watch for incorrect matches")
    
    return threshold