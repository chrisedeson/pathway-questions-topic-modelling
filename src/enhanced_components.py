"""
Enhanced UI Components for BYU Pathway Hybrid Topic Analysis
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import asyncio
import os
import io
from streamlit_autorefresh import st_autorefresh

from utils import validate_questions_file, create_session_state_defaults, calculate_clustering_metrics
from visualizations import display_metrics_overview
from google_sheets_utils import (
    GoogleSheetsManager, display_sheets_permission_status,
    create_sheets_connection_ui, SheetsPermission
)
from topic_management import (
    create_topic_editor_ui, create_changes_confirmation_dialog,
    display_topic_statistics, create_similarity_threshold_ui
)
from hybrid_topic_processor import HybridTopicProcessor
from config import GOOGLE_SHEETS_AUTO_REFRESH_INTERVAL


def create_chart_header(title: str, explanation: str, icon: str = "❔"):
    """Create a chart header with a helpful tooltip explanation"""
    col1, col2 = st.columns([10, 1])
    with col1:
        st.subheader(title)
    with col2:
        st.markdown(f"""
        <div style="text-align: right; padding-top: 10px;">
            <span title="{explanation}" style="font-size: 16px; cursor: help;">{icon}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Also show as an expandable info box for mobile/accessibility
    with st.expander("ℹ️ What does this chart show?", expanded=False):
        st.info(explanation)


def display_header():
    """Display app header with styling"""
    st.markdown('<h1 class="main-header" style="margin-bottom: 0.5rem;">🎓 BYU Pathway Hybrid Topic Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header" style="margin-bottom: 1rem;">AI-Powered Similarity Classification & Topic Discovery</p>', unsafe_allow_html=True)


def check_api_key() -> bool:
    """Check if API key is available"""
    from config import OPENAI_API_KEY
    
    if not OPENAI_API_KEY:
        st.error("🔑 OpenAI API key not found. Please add your API key to the .env file.")
        st.code("OPENAI_API_KEY=your_api_key_here")
        st.info("💡 **Need an interactive analysis session?** Use the hybrid processing notebook in the insights folder.")
        return False
    return True


def create_data_source_selection():
    """Create UI for selecting data source (file upload or Google Sheets)"""
    
    st.subheader("📁 Data Source Selection")
    
    source_type = st.radio(
        "**Choose your data source:**",
        ["📤 File Upload", "🔗 Google Sheets"],
        help="Select how you want to provide your data"
    )
    
    if source_type == "📤 File Upload":
        return create_file_upload_ui()
    else:
        return create_google_sheets_ui()


def create_file_upload_ui() -> Dict[str, Any]:
    """Create file upload interface"""
    
    st.write("**Upload your data files:**")
    
    # Questions file upload
    questions_file = st.file_uploader(
        "**📋 Questions File** (CSV or TXT)",
        type=['csv', 'txt'],
        help="Upload a file containing the questions to analyze. CSV files should have a 'question' column."
    )
    
    # Topics file upload (optional)
    topics_file = st.file_uploader(
        "**📊 Topics File** (CSV) - Optional",
        type=['csv'],
        help="Upload a CSV file with existing topics. Should have columns: 'Topic', 'Subtopic', 'Question'"
    )
    
    result = {"source_type": "file", "questions_data": None, "topics_data": None}
    
    if questions_file:
        try:
            if questions_file.name.endswith('.csv'):
                questions_df = pd.read_csv(questions_file)
                if 'question' not in questions_df.columns:
                    # Try to find a suitable column
                    text_columns = [col for col in questions_df.columns if questions_df[col].dtype == 'object']
                    if text_columns:
                        st.warning(f"No 'question' column found. Using '{text_columns[0]}' column.")
                        questions_df = questions_df.rename(columns={text_columns[0]: 'question'})
                    else:
                        st.error("No suitable text column found in CSV file.")
                        return result
            else:  # TXT file
                content = questions_file.read().decode('utf-8')
                questions = [q.strip() for q in content.split('\n') if q.strip()]
                if not questions:
                    st.error("No valid questions found in the text file.")
                    return result
                questions_df = pd.DataFrame({'question': questions})
            
            # Additional validation
            if questions_df.empty:
                st.error("The uploaded file contains no valid data.")
                return result
                
            result["questions_data"] = questions_df
            st.success(f"✅ Loaded {len(questions_df)} questions from file")
            
        except Exception as e:
            st.error(f"Error loading questions file: {str(e)}")
    
    if topics_file:
        try:
            topics_df = pd.read_csv(topics_file)
            
            required_columns = ['Topic', 'Subtopic', 'Question']
            missing_columns = [col for col in required_columns if col not in topics_df.columns]
            
            if missing_columns:
                st.error(f"Topics file missing required columns: {missing_columns}")
            elif topics_df.empty:
                st.error("The topics file contains no valid data.")
            else:
                result["topics_data"] = topics_df
                st.success(f"✅ Loaded {len(topics_df)} topic questions from file")
                
        except Exception as e:
            st.error(f"Error loading topics file: {str(e)}")
    
    return result


def create_google_sheets_ui() -> Dict[str, Any]:
    """Create Google Sheets interface with separate inputs for questions and topics"""
    
    result = {"source_type": "sheets", "questions_data": None, "topics_data": None}
    
    # Auto-refresh setup
    if st.checkbox("🔄 **Auto-refresh** (updates every 10 seconds)", value=False):
        st_autorefresh(interval=GOOGLE_SHEETS_AUTO_REFRESH_INTERVAL * 1000, key="sheets_refresh")
    
    # Create two columns for questions and topics sheets
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📋 Questions Sheet** (Required)")
        st.write("*Format: Simple list of questions or CSV with 'question' column*")
        
        questions_url = st.text_input(
            "Questions Sheet URL",
            placeholder="https://docs.google.com/spreadsheets/d/your-questions-sheet-id/edit",
            help="Paste the URL of your Google Sheet containing questions"
        )
        
        questions_worksheet = st.text_input(
            "Questions Worksheet Name (optional)",
            placeholder="Sheet1",
            help="Leave empty to use the first worksheet"
        )
        
        if questions_url:
            sheets_manager = GoogleSheetsManager()
            
            # Check permissions for questions sheet
            permission, error_msg = sheets_manager.check_sheet_permissions(questions_url)
            display_sheets_permission_status(permission, error_msg)
            
            if permission != SheetsPermission.NO_ACCESS:
                try:
                    # Try to read questions sheet
                    questions_df, error_msg = sheets_manager.read_topics_from_sheet(questions_url, questions_worksheet)
                    
                    if questions_df is not None:
                        # Check if it has a 'question' column or try to find questions
                        if 'Question' in questions_df.columns:
                            questions_data = pd.DataFrame({'question': questions_df['Question']})
                        elif 'question' in questions_df.columns:
                            questions_data = questions_df[['question']]
                        else:
                            # Try to use first text column as questions
                            text_cols = [col for col in questions_df.columns if questions_df[col].dtype == 'object']
                            if text_cols:
                                questions_data = pd.DataFrame({'question': questions_df[text_cols[0]]})
                                st.warning(f"Using '{text_cols[0]}' column as questions")
                            else:
                                st.error("No suitable question column found")
                                questions_data = None
                        
                        if questions_data is not None and not questions_data.empty:
                            result["questions_data"] = questions_data
                            st.success(f"✅ Loaded {len(questions_data)} questions")
                            
                            with st.expander("📋 Questions Preview", expanded=False):
                                st.dataframe(questions_data.head(), use_container_width=True)
                    
                    elif error_msg:
                        st.error(f"Error reading questions sheet: {error_msg}")
                        
                except Exception as e:
                    st.error(f"Error accessing questions sheet: {str(e)}")
    
    with col2:
        st.write("**📊 Topics Sheet** (Optional)")
        st.write("*Format: CSV with 'Topic', 'Subtopic', 'Question' columns*")
        
        topics_url = st.text_input(
            "Topics Sheet URL",
            placeholder="https://docs.google.com/spreadsheets/d/your-topics-sheet-id/edit",
            help="Paste the URL of your Google Sheet containing existing topics (optional)"
        )
        
        topics_worksheet = st.text_input(
            "Topics Worksheet Name (optional)",
            placeholder="Sheet1",
            help="Leave empty to use the first worksheet",
            key="topics_worksheet"
        )
        
        if topics_url:
            sheets_manager = GoogleSheetsManager()
            
            # Check permissions for topics sheet
            permission, error_msg = sheets_manager.check_sheet_permissions(topics_url)
            display_sheets_permission_status(permission, error_msg)
            
            if permission != SheetsPermission.NO_ACCESS:
                try:
                    # Try to read topics sheet
                    topics_df, error_msg = sheets_manager.read_topics_from_sheet(topics_url, topics_worksheet)
                    
                    if topics_df is not None and not topics_df.empty:
                        required_columns = ['Topic', 'Subtopic', 'Question']
                        missing_columns = [col for col in required_columns if col not in topics_df.columns]
                        
                        if missing_columns:
                            st.error(f"Missing required columns: {missing_columns}")
                        else:
                            result["topics_data"] = topics_df
                            st.success(f"✅ Loaded {len(topics_df)} topic questions")
                            
                            with st.expander("� Topics Preview", expanded=False):
                                st.dataframe(topics_df.head(), use_container_width=True)
                    
                    elif error_msg:
                        st.error(f"Error reading topics sheet: {error_msg}")
                        
                except Exception as e:
                    st.error(f"Error accessing topics sheet: {str(e)}")
    
    return result


def create_hybrid_processing_tab():
    """Create the main hybrid processing interface"""
    
    st.header("🚀 Hybrid Topic Analysis")
    
    # Data source selection
    data_result = create_data_source_selection()
    
    # Check if we have valid questions data
    if (data_result["questions_data"] is None or 
        data_result["questions_data"].empty or 
        len(data_result["questions_data"]) == 0):
        st.info("Please upload questions data or connect to Google Sheets to continue.")
        return
    
    questions_df = data_result["questions_data"]
    topics_df = data_result.get("topics_data")
    
    # Configuration section
    st.subheader("⚙️ Processing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        similarity_threshold = create_similarity_threshold_ui()
        
        processing_mode = st.selectbox(
            "**Processing Mode**",
            ["sample", "all"],
            help="Process all questions or a sample for testing"
        )
        
        if processing_mode == "sample":
            sample_size = st.number_input(
                "**Sample Size**",
                min_value=100,
                max_value=min(5000, len(questions_df)),
                value=min(2000, len(questions_df)),
                step=100
            )
        else:
            sample_size = len(questions_df)
    
    with col2:
        if topics_df is not None:
            st.write("**📊 Existing Topics Data:**")
            st.info(f"✅ {len(topics_df)} topic questions loaded")
            display_topic_statistics(topics_df)
        else:
            st.warning("⚠️ **No existing topics data**. All questions will be clustered into new topics.")
            
            # Offer to create sample topics
            if st.button("📝 **Create Sample Topics**"):
                sample_topics = create_sample_topics_data()
                st.session_state['topics_data'] = sample_topics
                st.rerun()
    
    # Process button
    if st.button("🚀 **Start Hybrid Analysis**", type="primary", use_container_width=True):
        if topics_df is None:
            topics_df = pd.DataFrame(columns=['Topic', 'Subtopic', 'Question'])
        
        # Run hybrid analysis
        result = run_hybrid_analysis(questions_df, topics_df, similarity_threshold, processing_mode, sample_size)
        
        if result:
            st.session_state['hybrid_results'] = result
            st.success("✅ **Analysis Complete!** Check the results tabs below.")
            st.rerun()


def run_hybrid_analysis(questions_df: pd.DataFrame, 
                       topics_df: pd.DataFrame,
                       threshold: float,
                       mode: str,
                       sample_size: int) -> Dict[str, Any]:
    """Run the hybrid topic analysis"""
    
    try:
        processor = HybridTopicProcessor()
        
        # Update processor configuration
        processor.similarity_threshold = threshold
        processor.processing_mode = mode
        processor.sample_size = sample_size
        
        # Run analysis
        with st.spinner("Running hybrid analysis..."):
            result = asyncio.run(processor.process_hybrid_analysis(
                questions_df, topics_df, threshold
            ))
            
        return result
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None


def display_hybrid_results(results: Dict[str, Any]):
    """Display hybrid analysis results"""
    
    st.header("📊 Hybrid Analysis Results")
    
    similar_df = results.get('similar_questions_df', pd.DataFrame())
    clustered_df = results.get('clustered_questions_df')
    topic_names = results.get('topic_names', {})
    output_files = results.get('output_files', [])
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 **Total Processed**", len(results.get('eval_questions_df', [])))
    
    with col2:
        st.metric("✅ **Matched Existing**", len(similar_df))
    
    with col3:
        clustered_count = len(clustered_df) if clustered_df is not None else 0
        st.metric("🆕 **New Topics**", clustered_count)
    
    with col4:
        st.metric("📁 **Output Files**", len(output_files))
    
    # Results tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Similar Questions", 
        "🆕 New Topics", 
        "📁 Output Files",
        "📊 Analysis Details"
    ])
    
    with tab1:
        display_similar_questions_tab(similar_df)
    
    with tab2:
        display_new_topics_tab(clustered_df, topic_names)
    
    with tab3:
        display_output_files_tab(output_files)
    
    with tab4:
        display_analysis_details_tab(results)


def display_similar_questions_tab(similar_df: pd.DataFrame):
    """Display questions that matched existing topics"""
    
    st.subheader("✅ Questions Matched to Existing Topics")
    
    if similar_df.empty:
        st.info("No questions matched existing topics with the current similarity threshold.")
        return
    
    # Summary by topic
    topic_counts = similar_df['matched_topic'].value_counts()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Matches by Topic:**")
        for topic, count in topic_counts.head(10).items():
            st.write(f"• **{topic}**: {count} questions")
    
    with col2:
        st.write("**Similarity Distribution:**")
        st.bar_chart(similar_df['similarity_score'])
    
    # Detailed table
    st.write("**Detailed Results:**")
    st.dataframe(
        similar_df[['question', 'matched_topic', 'matched_subtopic', 'similarity_score']],
        use_container_width=True
    )


def display_new_topics_tab(clustered_df: Optional[pd.DataFrame], topic_names: Dict[int, str]):
    """Display newly discovered topics"""
    
    st.subheader("🆕 Newly Discovered Topics")
    
    if clustered_df is None or clustered_df.empty:
        st.info("No new topics were discovered. All questions matched existing topics or were classified as noise.")
        return
    
    # Topic summary
    cluster_groups = clustered_df.groupby('cluster_id')
    
    for cluster_id, group in cluster_groups:
        topic_name = topic_names.get(cluster_id, f"Cluster {cluster_id}")
        
        with st.expander(f"📂 **{topic_name}** ({len(group)} questions)", expanded=False):
            
            # Representative questions
            st.write("**Sample Questions:**")
            for i, question in enumerate(group['question'].head(5)):
                st.write(f"{i+1}. {question}")
            
            if len(group) > 5:
                st.write(f"... and {len(group) - 5} more questions")
            
            # Action buttons for this topic
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"✅ **Approve Topic**", key=f"approve_{cluster_id}"):
                    st.success(f"Topic '{topic_name}' approved for addition!")
            
            with col2:
                if st.button(f"❌ **Reject Topic**", key=f"reject_{cluster_id}"):
                    st.warning(f"Topic '{topic_name}' rejected.")


def display_output_files_tab(output_files: list):
    """Display and allow download of output files"""
    
    st.subheader("📁 Output Files")
    
    if not output_files:
        st.info("No output files generated.")
        return
    
    file_descriptions = {
        0: ("📋 **Similar Questions File**", "Questions that matched existing topics with their similarity scores"),
        1: ("🆕 **New Topics File**", "Newly discovered topics with representative questions and question counts"),
        2: ("📊 **Complete Review File**", "All questions with their topic assignments for comprehensive review")
    }
    
    for i, filepath in enumerate(output_files):
        if not os.path.exists(filepath):
            st.warning(f"File not found: {filepath}")
            continue
            
        title, description = file_descriptions.get(i, ("📄 **Output File**", "Analysis output file"))
        
        with st.expander(title, expanded=True):
            st.write(description)
            
            # Load and display preview
            try:
                df = pd.read_csv(filepath)
                st.write(f"**Preview** ({len(df)} rows):")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Download button
                with open(filepath, 'rb') as f:
                    file_data = f.read()
                
                st.download_button(
                    label=f"⬇️ **Download {os.path.basename(filepath)}**",
                    data=file_data,
                    file_name=os.path.basename(filepath),
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")


def display_analysis_details_tab(results: Dict[str, Any]):
    """Display detailed analysis information"""
    
    st.subheader("📊 Analysis Details")
    
    # Configuration used
    st.write("**Configuration Used:**")
    config_info = {
        "Similarity Threshold": results.get('similarity_threshold', 'N/A'),
        "Processing Mode": results.get('processing_mode', 'N/A'),
        "Questions Processed": len(results.get('eval_questions_df', [])),
        "Embedding Model": "text-embedding-3-small",
        "Chat Model": "gpt-5-nano"
    }
    
    for key, value in config_info.items():
        st.write(f"• **{key}**: {value}")
    
    # Processing summary
    similar_df = results.get('similar_questions_df', pd.DataFrame())
    clustered_df = results.get('clustered_questions_df')
    
    st.write("**Processing Summary:**")
    st.write(f"• **Total questions processed**: {len(results.get('eval_questions_df', []))}")
    st.write(f"• **Questions matched to existing topics**: {len(similar_df)}")
    st.write(f"• **Questions assigned to new topics**: {len(clustered_df) if clustered_df is not None else 0}")
    st.write(f"• **New topics discovered**: {len(set(clustered_df['cluster_id'])) if clustered_df is not None else 0}")


def create_topic_management_tab():
    """Create topic management interface"""
    
    st.header("📝 Topic Management")
    
    # Check if we have topics data
    if 'topics_data' not in st.session_state or st.session_state['topics_data'] is None:
        st.info("No topics data loaded. Please load data from the Hybrid Analysis tab first.")
        return
    
    topics_df = st.session_state['topics_data']
    
    # Topic editor
    edited_topics_df = create_topic_editor_ui(topics_df)
    
    # Check for changes and show confirmation dialog
    if not topics_df.equals(edited_topics_df):
        if create_changes_confirmation_dialog(topics_df, edited_topics_df):
            # Apply changes
            st.session_state['topics_data'] = edited_topics_df
            st.success("✅ **Changes applied successfully!**")
            
            # If using Google Sheets, offer to sync back
            if st.session_state.get('data_source_type') == 'sheets':
                if st.button("🔄 **Sync Changes to Google Sheets**", type="primary"):
                    sync_changes_to_sheets(edited_topics_df)
            
            st.rerun()


def sync_changes_to_sheets(topics_df: pd.DataFrame):
    """Sync changes back to Google Sheets"""
    
    sheet_url = st.session_state.get('current_sheet_url')
    worksheet_name = st.session_state.get('current_worksheet_name')
    
    if not sheet_url:
        st.error("No Google Sheet URL found.")
        return
    
    try:
        with st.spinner("Syncing changes to Google Sheets..."):
            sheets_manager = GoogleSheetsManager()
            error_msg = sheets_manager.write_topics_to_sheet(sheet_url, topics_df, worksheet_name)
            
            if error_msg:
                st.error(f"Error syncing to Google Sheets: {error_msg}")
            else:
                st.success("✅ **Changes synced to Google Sheets successfully!**")
                
    except Exception as e:
        st.error(f"Unexpected error syncing to Google Sheets: {str(e)}")


def create_sample_topics_data() -> pd.DataFrame:
    """Create sample topics data for testing"""
    
    sample_data = [
        {"Topic": "Technical Support", "Subtopic": "Login Issues", "Question": "How do I reset my password?"},
        {"Topic": "Technical Support", "Subtopic": "Login Issues", "Question": "I can't log into my account"},
        {"Topic": "Technical Support", "Subtopic": "System Issues", "Question": "The website is not loading"},
        {"Topic": "Academic Support", "Subtopic": "Course Materials", "Question": "Where can I find my textbooks?"},
        {"Topic": "Academic Support", "Subtopic": "Course Materials", "Question": "How do I access course resources?"},
        {"Topic": "Academic Support", "Subtopic": "Grades", "Question": "How do I check my grades?"},
        {"Topic": "Financial Aid", "Subtopic": "Scholarships", "Question": "What scholarships are available?"},
        {"Topic": "Financial Aid", "Subtopic": "Payment", "Question": "How do I pay my tuition?"},
    ]
    
    return pd.DataFrame(sample_data)


def display_app_footer():
    """Display app footer"""
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #888;">BYU Pathway Hybrid Topic Analysis • Powered by OpenAI GPT & Embeddings</p>',
        unsafe_allow_html=True
    )