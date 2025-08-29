"""
UI Components for BYU Pathway Questions Analysis
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional
from pathlib import Path

from utils import validate_questions_file, create_session_state_defaults, calculate_clustering_metrics
from visualizations import display_metrics_overview


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
    st.markdown('<h1 class="main-header" style="margin-bottom: 0.5rem;">🎓 BYU Pathway Questions Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header" style="margin-bottom: 1rem;">AI-Powered Topic Discovery and Question Clustering</p>', unsafe_allow_html=True)


def check_api_key() -> bool:
    """Check if API key is available"""
    from config import OPENAI_API_KEY
    
    if not OPENAI_API_KEY:
        st.error("🔑 OpenAI API key not found. Please add your API key to the .env file.")
        st.code("OPENAI_API_KEY=your_api_key_here")
        st.info("💡 **Need an interactive analysis session?** [Open in Google Colab](https://colab.research.google.com/drive/1TPGv-qhQXlJe5Z3OAOrZ1O6USNlwHl21?usp=sharing) - `BYU_Pathway_Topic_Modeling_Colab.ipynb`")
        return False
    return True


def display_analysis_summary(df: pd.DataFrame, metrics: Optional[dict] = None):
    """Display analysis summary with detailed table (no duplicate metrics)"""
    st.header("📊 Analysis Summary")
    
    # Calculate metrics if not provided
    if metrics is None:
        metrics = calculate_clustering_metrics(df)
    
    # Skip display_metrics_overview - it's now only in Enhanced Metrics tab
    
    st.divider()
    
    # Enhanced Topic Analysis
    st.subheader("🏆 Topic Analysis")
    
    topic_counts = df.groupby('Topic_Name').size().reset_index(name='Count')
    topic_counts = topic_counts.sort_values('Count', ascending=False)
    
    # Add topic statistics
    topic_stats = df.groupby('Topic_Name').agg({
        'Probability': ['mean', 'std', 'min', 'max'],
        'Topic_ID': 'first'
    }).round(3)
    topic_stats.columns = ['Avg_Confidence', 'Std_Confidence', 'Min_Confidence', 'Max_Confidence', 'Topic_ID']
    topic_stats = topic_stats.reset_index()
    
    # Merge with counts
    topic_analysis = topic_counts.merge(topic_stats, on='Topic_Name', how='left')
    
    # Display top topics
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("**📈 Top Topics by Question Count**")
        top_10 = topic_analysis.head(10)[['Topic_Name', 'Count', 'Avg_Confidence']]
        st.dataframe(
            top_10, 
            hide_index=True,
            use_container_width=True,
            column_config={
                'Topic_Name': st.column_config.TextColumn('Topic', width=200),
                'Count': st.column_config.NumberColumn('Questions', width=100),
                'Avg_Confidence': st.column_config.NumberColumn('Avg Confidence', format='%.3f', width=120)
            }
        )
    
    with col2:
        st.markdown("**🎯 Quick Stats**")
        st.info(f"""
        **Largest Topic:** {topic_analysis.iloc[0]['Topic_Name']} ({topic_analysis.iloc[0]['Count']} questions)
        
        **Most Confident:** {topic_analysis.loc[topic_analysis['Avg_Confidence'].idxmax()]['Topic_Name']} (avg: {topic_analysis['Avg_Confidence'].max():.3f})
        
        **Coverage:** {metrics['categorized_percentage']:.1f}% of questions categorized
        """)


def display_question_explorer(df: pd.DataFrame):
    """Display question exploration interface"""
    st.header("🔍 Question Explorer")
    
    # Add helpful explanation with performance note
    st.info("🎯 **Explore Your Questions:** Use the filters below to find specific questions, explore topics, or see how confident the AI was about different categorizations!")
    
    # Performance optimization notice
    if len(df) > 500:
        st.success("⚡ **Performance Optimized:** This explorer uses smart caching and table view for instant filtering and sorting, even with large datasets!")
    
    # Initialize session state for performance optimization
    if 'question_explorer_state' not in st.session_state:
        st.session_state.question_explorer_state = {
            'last_filter_hash': None,
            'cached_filtered_df': None
        }
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_topics = st.multiselect(
            "Select Topics",
            options=sorted(df['Topic_Name'].unique()),
            help="Filter questions by topic - choose one or more topics to focus on",
            key="topics_multiselect"
        )
    
    with col2:
        min_confidence = st.slider(
            "Minimum Confidence",
            0.0, 1.0, 0.0, 0.1,
            help="Filter by confidence score - higher values show questions the AI was more sure about",
            key="confidence_slider"
        )
    
    with col3:
        search_text = st.text_input(
            "Search in Questions",
            help="Search for specific words or phrases within the question text",
            key="search_input"
        )
    
    # Create filter hash to detect changes and cache expensive filtering operation
    filter_state = f"{selected_topics}_{min_confidence}_{search_text}"
    current_filter_hash = hash(filter_state)
    
    # Only recompute filtering if filters actually changed
    if (st.session_state.question_explorer_state['last_filter_hash'] != current_filter_hash or
        st.session_state.question_explorer_state['cached_filtered_df'] is None):
        
        # Apply filters (expensive operation)
        filtered_df = df.copy()
        
        if selected_topics:
            filtered_df = filtered_df[filtered_df['Topic_Name'].isin(selected_topics)]
        
        if min_confidence > 0:
            filtered_df = filtered_df[filtered_df['Probability'] >= min_confidence]
        
        if search_text:
            filtered_df = filtered_df[
                filtered_df['Question'].str.contains(search_text, case=False, na=False)
            ]
        
        # Cache the result
        st.session_state.question_explorer_state['last_filter_hash'] = current_filter_hash
        st.session_state.question_explorer_state['cached_filtered_df'] = filtered_df
    else:
        # Use cached result - much faster!
        filtered_df = st.session_state.question_explorer_state['cached_filtered_df']
    
    # Display results
    st.subheader(f"📋 Questions ({len(filtered_df)} found)")
    
    if len(filtered_df) == 0:
        st.warning("😅 No questions match your filters. Try adjusting the criteria above!")
        st.info("💡 **Tips:** Try selecting fewer topics, lowering the confidence threshold, or using different search words.")
        return
    
    # Add explanation of confidence colors
    with st.expander("❔ What do the confidence colors mean?"):
        st.markdown("""
        **Confidence Score Colors:**
        - 🟢 **Green (0.7+):** AI is very confident about this topic assignment
        - 🟡 **Yellow (0.4-0.7):** AI is somewhat confident, but not totally sure  
        - 🔴 **Red (below 0.4):** AI had to guess - this question might fit multiple topics
        
        **Higher confidence = Better topic assignment**
        """)
    
    # Sort and pagination options with theme-friendly widgets
    sort_col1, sort_col2 = st.columns(2)
    
    with sort_col1:
        sort_by = st.selectbox(
            "Sort by",
            ['Topic_Name', 'Probability', 'Question'],
            help="Choose how to sort the results - try sorting by confidence to see the most/least certain classifications",
            key="question_sort_by"
        )
    
    with sort_col2:
        # Use radio buttons for better theme compatibility
        sort_order_radio = st.radio(
            "Order",
            ['Ascending', 'Descending'],
            key="question_sort_order_radio",
            horizontal=True,
            help="Choose sort direction"
        )
        sort_order = sort_order_radio
    
    # Apply sorting efficiently 
    ascending = sort_order == 'Ascending'
    sorted_df = filtered_df.sort_values(sort_by, ascending=ascending)
    
    # Pagination for better performance
    items_per_page = 50
    total_pages = (len(sorted_df) + items_per_page - 1) // items_per_page
    
    if total_pages > 1:
        page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
        with page_col2:
            # Use number input for better theme compatibility
            current_page = st.number_input(
                f"Page (1-{total_pages}, showing {items_per_page} items per page)",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                key="question_page_number",
                help=f"Navigate through {total_pages} pages of results"
            )
    else:
        current_page = 1
    
    # Calculate start and end indices for current page
    start_idx = (current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(sorted_df))
    page_df = sorted_df.iloc[start_idx:end_idx]
    
    # Display page info
    if total_pages > 1:
        st.info(f"📄 Showing questions {start_idx + 1}-{end_idx} of {len(sorted_df)} total results (Page {current_page} of {total_pages})")
    
    # Use container to reduce rerendering
    display_container = st.container()
    
    with display_container:
        # Display questions using st.dataframe for maximum performance 
        use_table_view = st.toggle("📋 Use High-Performance Table View", 
                                  value=True, 
                                  help="Table view loads instantly and handles large datasets efficiently. Card view is prettier but slower.",
                                  key="performance_table_toggle")
        
        if use_table_view:
            # High-performance table view - prepare data efficiently
            confidence_colors = []
            for prob in page_df['Probability']:
                if prob > 0.7:
                    confidence_colors.append("🟢")
                elif prob > 0.4:
                    confidence_colors.append("🟡")
                else:
                    confidence_colors.append("🔴")
            
            # Create display dataframe efficiently using vectorized operations
            display_df = pd.DataFrame({
                'Question': page_df['Question'].values,
                'Topic': page_df['Topic_Name'].values,
                'Confidence': [f"{color} {prob:.3f}" for color, prob in zip(confidence_colors, page_df['Probability'])]
            })
            
            # Display with optimized configuration
            st.dataframe(
                display_df,
                column_config={
                    "Question": st.column_config.TextColumn(
                        "Question",
                        help="The original question text",
                        width="large"
                    ),
                    "Topic": st.column_config.TextColumn(
                        "Topic", 
                        help="AI-assigned topic category",
                        width="medium"
                    ),
                    "Confidence": st.column_config.TextColumn(
                        "Confidence",
                        help="How confident the AI was about this topic assignment",
                        width="small"
                    )
                },
                use_container_width=True,
                hide_index=True,
                height=min(400, len(display_df) * 35 + 50)  # Dynamic height for better UX
            )
        else:
            # Card view - prettier but slower, only for small datasets
            if len(page_df) > 20:
                st.warning("⚡ **Performance Notice:** Card view with many questions may be slow. Consider using Table View for better performance!")
            
            for _, row in page_df.iterrows():
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"**{row['Question']}**")
                        st.caption(f"Topic: {row['Topic_Name']} | Confidence: {row['Probability']:.3f}")
                    
                    with col2:
                        confidence_color = "🟢" if row['Probability'] > 0.7 else "🟡" if row['Probability'] > 0.4 else "🔴"
                        st.markdown(f"{confidence_color} {row['Probability']:.3f}")
    
    # Show summary at bottom
    if len(sorted_df) > items_per_page:
        st.caption(f"💡 **Performance tip:** Table view is enabled by default for faster browsing!")


def upload_and_analyze_tab():
    """Upload and analyze tab with enhanced UI"""
    st.header("📤 Upload Questions File")
    
    # Enhanced file upload guidance
    st.info("🎯 **Getting Started:** Upload your questions file to discover topics and patterns using AI!")
    
    with st.expander("📋 File Format Requirements", expanded=True):
        st.markdown("""
        **✅ What you need:**
        - A `.txt` file (plain text)
        - One question per line
        - At least 50+ questions (more is better!)
        - Questions should be clear and complete
        
        **✨ Tips for best results:** 
        - Keep questions in their natural language
        - Include a variety of question types
        - More questions = better topic discovery
        
        **❌ Avoid:**
        - Very short phrases (less than 3 words)
        - Questions with lots of typos
        - Mixed languages in the same file
        """)
    
    uploaded_file = st.file_uploader(
        "Choose a text file with questions",
        type=['txt'],
        help="Upload a .txt file with one question per line - the AI will find patterns and group similar questions together",
        key="main_file_uploader"
    )
    
    if uploaded_file is not None:
        # Validate file
        content = uploaded_file.read().decode('utf-8')
        uploaded_file.seek(0)  # Reset file pointer
        
        is_valid, questions, message = validate_questions_file(content)
        
        if is_valid:
            st.success(message)
            
            # File stats
            st.markdown(f"**📊 File Stats:** {len(questions)} questions ready for analysis")
            
            # File preview
            with st.expander("👀 Preview Questions (First 5)", expanded=False):
                for i, q in enumerate(questions[:5], 1):
                    st.markdown(f"**{i}.** {q}")
                if len(questions) > 5:
                    st.caption(f"... and {len(questions) - 5} more questions")
            
            # Analysis button and controls
            st.divider()
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.markdown("**🚀 Ready to analyze?**")
                analyze_button = st.button(
                    "▶ Run Analysis",
                    type="primary",
                    help="Start the AI topic modeling analysis - this will take 3-5 minutes depending on the number of questions"
                )
            
            with col3:
                st.markdown("💡 **Need more control?**")
                st.markdown("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zOLprkByUwWutJPGo2VMB83yMd9LxQGB?usp=sharing)")
            
            # Run analysis
            if analyze_button:
                st.divider()
                st.header("🔄 Analysis in Progress")
                
                # Import here to avoid circular imports
                from topic_modeling import process_questions_file
                from utils import save_results, calculate_clustering_metrics
                
                # Process file
                result = process_questions_file(uploaded_file)
                
                if result is not None:
                    results_df, topic_model, embeddings = result
                    
                    # Calculate metrics
                    metrics = calculate_clustering_metrics(results_df, embeddings)
                    
                    # Save results
                    csv_path = save_results(results_df, topic_model)
                    st.success(f"✅ Analysis complete! Results saved to {csv_path}")
                    
                    # Store in session state
                    st.session_state.current_results = results_df
                    st.session_state.current_topic_model = topic_model
                    st.session_state.current_embeddings = embeddings  # Store embeddings too
                    st.session_state.clustering_metrics = metrics
                    st.session_state.analysis_complete = True
                    
                    # Auto-rerun to show results
                    st.rerun()
        else:
            st.error(message)


def view_previous_results_tab():
    """View previous results tab"""
    st.header("📊 Previous Analysis Results")
    
    # Load available results
    results_dir = Path("results")
    if not results_dir.exists():
        st.info("No previous results found. Upload a file to start your first analysis!")
        return
    
    csv_files = list(results_dir.glob("pathway_questions_analysis_*.csv"))
    if not csv_files:
        st.info("No previous results found. Upload a file to start your first analysis!")
        return
    
    # Select file to view
    file_options = {f.name: f for f in sorted(csv_files, key=lambda x: x.stat().st_mtime, reverse=True)}
    
    selected_file = st.selectbox(
        "Select Analysis Results",
        options=list(file_options.keys()),
        help="Choose from previously saved analyses"
    )
    
    if selected_file:
        # Load and display results
        try:
            df = pd.read_csv(file_options[selected_file])
            
            st.success(f"✅ Loaded {len(df)} questions from {selected_file}")
            
            # Calculate metrics for this dataset
            metrics = calculate_clustering_metrics(df)
            
            # Store in session state for viewing
            st.session_state.current_results = df
            st.session_state.clustering_metrics = metrics
            
            # Display summary
            display_analysis_summary(df, metrics)
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")


def display_app_footer():
    """Display app footer"""
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 2rem;">
        <p>🎓 BYU Pathway Questions Analysis Dashboard</p>
        <p><small>Powered by OpenAI, BERTopic, and Streamlit • Built for Educational Insights</small></p>
    </div>
    """, unsafe_allow_html=True)
