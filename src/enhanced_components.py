"""
Enhanced UI Components for BYU Pathway Questions Analysis
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any
import os
from pathlib import Path
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import visualizations
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import asyncio
import os
import io

from utils import validate_questions_file, create_session_state_defaults, calculate_clustering_metrics
from google_sheets_utils import (
    GoogleSheetsManager, display_sheets_permission_status,
    create_sheets_connection_ui, SheetsPermission
)
from hybrid_topic_processor import HybridTopicProcessor


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
    st.markdown('<h1 class="main-header" style="margin-bottom: 0.5rem;">BYU Pathway Hybrid Topic Analysis</h1>', unsafe_allow_html=True)
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
    
    # Centralized Service Account Information
    st.info("📧 **Service Account**: `streamlit-sheets-reader@byu-pathway-chatbot.iam.gserviceaccount.com`")
    
    with st.expander("📋 **How to Share Your Google Sheets**", expanded=False):
        st.markdown("""
        **Step-by-step instructions for BOTH sheets:**
        
        1. **Open your Google Sheet** in a web browser
        2. **Click the 'Share' button** (top right corner)
        3. **Add this email**: `streamlit-sheets-reader@byu-pathway-chatbot.iam.gserviceaccount.com`
        4. **Set permission to 'Viewer'** (read-only access)
        5. **Click 'Send'** to grant access
        
        ⚠️ **Important**: Both sheets need to be shared with this service account.
        
        💡 **Alternative**: Make your sheets public by clicking "Anyone with the link" and setting to "Viewer".
        """)
    
    st.markdown("---")
    
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
                    questions_df, error_msg = sheets_manager.read_questions_from_sheet(questions_url, questions_worksheet)
                    
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
                                st.dataframe(questions_data.head(), width="stretch")
                    
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
                            st.session_state['topics_data'] = topics_df  # Store in session state
                            st.success(f"✅ Loaded {len(topics_df)} topic questions")
                            
                            with st.expander("� Topics Preview", expanded=False):
                                st.dataframe(topics_df.head(), width="stretch")
                    
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
            index=0,
            help="Process all questions or a sample for testing",
            key="processing_mode_select"
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
        
        # Clustering configuration
        with st.expander("🔧 **Advanced Clustering Settings**", expanded=False):
            from config import MIN_CLUSTER_SIZE
            st.write(f"**Current MIN_CLUSTER_SIZE**: {MIN_CLUSTER_SIZE}")
            
            # Provide guidance
            expected_questions = sample_size if processing_mode == "sample" else len(questions_df)
            recommended_size = max(5, min(15, expected_questions // 200))  # Dynamic recommendation
            
            st.info(f"💡 **Recommendation**: For {expected_questions} questions, consider MIN_CLUSTER_SIZE between {recommended_size-2} and {recommended_size+3}")
            st.markdown("""
            **Clustering Guidelines:**
            - **Smaller values (3-5)**: More topics, but risk over-clustering
            - **Medium values (8-12)**: Balanced, good for most datasets
            - **Larger values (15-25)**: Fewer, broader topics
            
            **Optimal ratio**: Aim for 10-30% of questions becoming new topics.
            """)
    
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
    if st.button("🚀 **Start Hybrid Analysis**", type="primary", width="stretch"):
        if topics_df is None:
            topics_df = pd.DataFrame(columns=['Topic', 'Subtopic', 'Question'])
        
        # Run hybrid analysis
        result = run_hybrid_analysis(questions_df, topics_df, similarity_threshold, processing_mode, sample_size)
        
        if result:
            st.session_state['hybrid_results'] = result
            st.success("✅ **Analysis Complete!** Results will show in a few moments...")
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
                questions_df, topics_df, threshold, mode, sample_size
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Similar Questions", 
        "🆕 New Topics", 
        "📊 Visualizations",
        "📁 Output Files",
        "� Analysis Details"
    ])
    
    with tab1:
        display_similar_questions_tab(similar_df)
    
    with tab2:
        display_new_topics_tab(clustered_df, topic_names)
    
    with tab3:
        display_visualizations_tab(results)
    
    with tab4:
        display_output_files_tab(output_files)
    
    with tab5:
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
    """Display newly discovered topics in an interactive table format"""
    
    st.subheader("🆕 Newly Discovered Topics")
    
    if clustered_df is None or clustered_df.empty:
        st.info("No new topics were discovered. All questions matched existing topics or were classified as noise.")
        return
    
    # Prepare data for the table
    topic_summary_data = []
    cluster_groups = clustered_df.groupby('cluster_id')
    
    for cluster_id, group in cluster_groups:
        if cluster_id >= 0:  # Exclude noise cluster (-1)
            topic_name = topic_names.get(cluster_id, f"Cluster {cluster_id}")
            question_count = len(group)
            
            # Get representative questions (top 3)
            representative_questions = group['question'].head(3).tolist()
            sample_questions = "; ".join(representative_questions)
            
            # Truncate if too long
            if len(sample_questions) > 200:
                sample_questions = sample_questions[:200] + "..."
            
            topic_summary_data.append({
                "Topic ID": cluster_id,
                "Topic Name": topic_name,
                "Question Count": question_count,
                "Sample Questions": sample_questions
            })
    
    if not topic_summary_data:
        st.info("No valid topics were discovered (only noise clusters found).")
        return
    
    # Create DataFrame for display
    topics_df = pd.DataFrame(topic_summary_data)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total New Topics", len(topics_df))
    with col2:
        total_questions = topics_df["Question Count"].sum()
        st.metric("🔢 Questions in New Topics", total_questions)
    with col3:
        avg_questions = round(topics_df["Question Count"].mean(), 1)
        st.metric("📈 Avg Questions per Topic", avg_questions)
    
    st.markdown("---")
    
    # Interactive table with search and sorting
    st.subheader("📋 Topics Overview")
    
    # Add search functionality
    search_term = st.text_input("🔍 Search topics", placeholder="Search by topic name or sample questions...")
    
    if search_term:
        # Filter based on search term
        mask = (
            topics_df["Topic Name"].str.contains(search_term, case=False, na=False) |
            topics_df["Sample Questions"].str.contains(search_term, case=False, na=False)
        )
        filtered_df = topics_df[mask]
    else:
        filtered_df = topics_df
    
    # Sort options
    col1, col2 = st.columns(2)
    with col1:
        sort_by = st.selectbox(
            "📊 Sort by:",
            ["Question Count", "Topic Name", "Topic ID"],
            index=0
        )
    with col2:
        sort_order = st.selectbox(
            "📈 Order:",
            ["Descending", "Ascending"],
            index=0
        )
    
    # Apply sorting
    ascending = sort_order == "Ascending"
    sorted_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
    
    # Display the interactive table
    if not sorted_df.empty:
        st.dataframe(
            sorted_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Topic ID": st.column_config.NumberColumn("ID", width="small"),
                "Topic Name": st.column_config.TextColumn("Topic Name", width="medium"),
                "Question Count": st.column_config.NumberColumn("Questions", width="small"),
                "Sample Questions": st.column_config.TextColumn("Sample Questions", width="large")
            }
        )
        
        # Detailed view option
        st.markdown("---")
        st.subheader("🔍 Detailed Topic View")
        
        selected_topic_id = st.selectbox(
            "Select a topic to view all questions:",
            options=sorted_df["Topic ID"].tolist(),
            format_func=lambda x: f"Topic {x}: {topic_names.get(x, f'Cluster {x}')}"
        )
        
        if selected_topic_id is not None:
            selected_topic_name = topic_names.get(selected_topic_id, f"Cluster {selected_topic_id}")
            topic_questions = clustered_df[clustered_df['cluster_id'] == selected_topic_id]['question'].tolist()
            
            st.write(f"**📂 {selected_topic_name}** ({len(topic_questions)} questions)")
            
            # Display all questions in an expandable format
            with st.expander(f"View all {len(topic_questions)} questions", expanded=False):
                for i, question in enumerate(topic_questions, 1):
                    st.write(f"{i}. {question}")
    
    else:
        st.warning("No topics match your search criteria.")


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
                st.write(f"**Complete Data** ({len(df)} rows):")
                st.dataframe(df, width="stretch")
                
                # Download button
                with open(filepath, 'rb') as f:
                    file_data = f.read()
                
                st.download_button(
                    label=f"⬇️ **Download {os.path.basename(filepath)}**",
                    data=file_data,
                    file_name=os.path.basename(filepath),
                    mime="text/csv",
                    width="stretch"
                )
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")


def display_visualizations_tab(results: Dict[str, Any]):
    """Display comprehensive visualizations"""
    
    st.subheader("📊 Data Visualizations")
    
    # Get the data and models from results
    eval_df = results.get('eval_questions_df')
    topic_model = results.get('topic_model')
    embeddings = results.get('embeddings')
    similar_df = results.get('similar_questions_df')
    clustered_df = results.get('clustered_questions_df')
    
    if eval_df is None or eval_df.empty:
        st.info("No data available for visualization.")
        return
    
    # Create visualization sub-tabs
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "🎯 Topic Distribution", 
        "📊 Confidence Analysis",
        "🌳 Topic Relationships", 
        "🔤 Topic Keywords"
    ])
    
    with viz_tab1:
        st.write("### Question Distribution Analysis")
        
        # Try to create proper dataframe for visualization
        if similar_df is not None and not similar_df.empty:
            # Prepare data for visualization
            viz_df = similar_df.copy()
            if 'matched_topic' in viz_df.columns:
                viz_df['Topic_Name'] = viz_df['matched_topic']
                viz_df['Probability'] = viz_df.get('similarity_score', 0.5)
                viz_df['Question'] = viz_df.get('question', 'Unknown')
                viz_df['Topic_ID'] = range(len(viz_df))
                
                # Interactive scatter plot if we have embeddings
                if embeddings is not None and topic_model is not None:
                    visualizations.display_interactive_scatter(viz_df, embeddings, topic_model)
                else:
                    visualizations.display_topic_distribution_chart(viz_df)
            else:
                st.info("No topic distribution data available for visualization.")
        else:
            st.info("No matched questions available for distribution visualization.")
    
    with viz_tab2:
        st.write("### Confidence Analysis")
        
        if similar_df is not None and not similar_df.empty and 'similarity_score' in similar_df.columns:
            # Prepare confidence data
            conf_df = similar_df.copy()
            conf_df['Probability'] = conf_df['similarity_score']
            conf_df['Topic_Name'] = conf_df.get('matched_topic', 'Unknown')
            conf_df['Topic_ID'] = range(len(conf_df))
            
            visualizations.display_confidence_distribution(conf_df)
        else:
            st.info("No confidence data available for analysis.")
    
    with viz_tab3:
        st.write("### Topic Relationships")
        
        if topic_model is not None:
            topic_names = results.get('topic_names', {})
            visualizations.display_topic_hierarchy(topic_model, topic_names)
            st.divider()
            visualizations.display_topic_similarity_heatmap(topic_model, topic_names)
        else:
            st.info("Topic model not available for relationship analysis.")
    
    with viz_tab4:
        st.write("### Topic Keywords")
        
        if topic_model is not None:
            topic_names = results.get('topic_names', {})
            visualizations.display_topic_words_chart(topic_model, topic_names)
        else:
            st.info("Topic model not available for keyword analysis.")


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


def create_similarity_threshold_ui():
    """Create UI for similarity threshold configuration"""
    return st.slider(
        "**Similarity Threshold**",
        min_value=0.5,
        max_value=0.95,
        value=0.70,
        step=0.05,
        help="Questions above this threshold will be matched to existing topics"
    )


def display_topic_statistics(topics_df: pd.DataFrame):
    """Display simple topic statistics"""
    topic_counts = topics_df.groupby('Topic').size().sort_values(ascending=False)
    st.write(f"**Topics**: {len(topic_counts)}")
    st.write(f"**Questions**: {len(topics_df)}")
    
    # Show top topics
    if len(topic_counts) > 0:
        with st.expander("📊 Top Topics", expanded=False):
            for topic, count in topic_counts.head(5).items():
                st.write(f"• **{topic}**: {count} questions")


def display_app_footer():
    """Display app footer"""
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #888;">BYU Pathway Hybrid Topic Analysis • Powered by OpenAI GPT & Embeddings</p>',
        unsafe_allow_html=True
    )