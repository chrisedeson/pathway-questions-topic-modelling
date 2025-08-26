import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
from datetime import datetime
import numpy as np
from bertopic import BERTopic
from openai import OpenAI
from dotenv import load_dotenv
import pickle
import tempfile
import umap.umap_ as umap
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
import io
import time
import json

# Page configuration
st.set_page_config(
    page_title="BYU Pathway Questions Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
    }
    .topic-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

def load_analysis_results():
    """Load the latest analysis results"""
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    
    # Find the most recent results file
    csv_files = list(results_dir.glob("pathway_questions_analysis_*.csv"))
    if not csv_files:
        return None
    
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    return pd.read_csv(latest_file)

def load_topic_model():
    """Load saved topic model if available"""
    model_files = list(Path(".").glob("topic_model_*.pkl"))
    if not model_files:
        return None
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    with open(latest_model, 'rb') as f:
        return pickle.load(f)

def process_questions_file(uploaded_file):
    """Process uploaded questions file and run topic modeling"""
    load_dotenv()
    
    # Read uploaded file
    content = uploaded_file.read().decode('utf-8')
    questions = [line.strip() for line in content.split('\n') if line.strip()]
    
    if len(questions) < 10:
        st.error(f"❌ Not enough questions. Found {len(questions)}, need at least 10 for meaningful analysis.")
        return None
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Generate embeddings
        status_text.text("🔄 Generating embeddings with OpenAI...")
        progress_bar.progress(20)
        
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-3-large"
            )
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
            
            progress_bar.progress(min(20 + int((i / len(questions)) * 40), 60))
        
        embeddings = np.array(embeddings)
        
        # Step 2: Reduce dimensionality and cluster
        status_text.text("🔄 Clustering questions...")
        progress_bar.progress(70)
        
        # UMAP for clustering
        umap_model = umap.UMAP(n_neighbors=15, n_components=5, random_state=42, metric='cosine')
        umap_embeddings = umap_model.fit_transform(embeddings)
        
        # HDBSCAN clustering
        hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')
        cluster_labels = hdbscan_model.fit_predict(umap_embeddings)
        
        progress_bar.progress(80)
        
        # Step 3: Topic modeling with BERTopic
        status_text.text("🔄 Generating topic labels...")
        
        # Create BERTopic model
        vectorizer_model = CountVectorizer(stop_words="english", max_features=1000)
        topic_model = BERTopic(
            embedding_model=None,  # We provide embeddings directly
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            verbose=True
        )
        
        # Fit the model
        topics, probs = topic_model.fit_transform(questions, embeddings)
        
        progress_bar.progress(90)
        
        # Step 4: Enhance topic labels with OpenAI
        status_text.text("🔄 Enhancing topic labels with AI...")
        
        topic_info = topic_model.get_topic_info()
        enhanced_labels = {}
        
        for topic_id in topic_info['Topic'].unique():
            if topic_id == -1:  # Skip noise
                continue
                
            keywords = topic_model.get_topic(topic_id)[:10]
            keyword_str = ", ".join([word for word, _ in keywords])
            
            prompt = f"""Based on these keywords from student questions: {keyword_str}
            
            Create a clear, concise topic label (2-4 words) that describes the main theme.
            Focus on what students are asking about. Examples: "Course Registration", "Financial Aid", "Technical Support"
            
            Topic label:"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.3
            )
            
            enhanced_labels[topic_id] = response.choices[0].message.content.strip().strip('"')
        
        # Step 5: Create results DataFrame
        results_df = pd.DataFrame({
            'Question': questions,
            'Topic_ID': topics,
            'Probability': probs,
            'Topic_Name': [enhanced_labels.get(topic_id, f"Topic {topic_id}") if topic_id != -1 else "Uncategorized" 
                          for topic_id in topics]
        })
        
        # UMAP for visualization
        umap_viz = umap.UMAP(n_neighbors=15, n_components=2, random_state=42, metric='cosine')
        viz_embeddings = umap_viz.fit_transform(embeddings)
        results_df['UMAP_X'] = viz_embeddings[:, 0]
        results_df['UMAP_Y'] = viz_embeddings[:, 1]
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        return results_df, topic_model
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        return None

def display_results(df):
    """Display analysis results with interactive visualizations"""
    
    # Sidebar
    st.sidebar.title("📊 Analysis Controls")
    
    # Filters
    topics = df['Topic_Name'].unique()
    selected_topics = st.sidebar.multiselect(
        "Filter by Topics",
        topics,
        default=topics[:5] if len(topics) > 5 else topics
    )
    
    # Filter data
    if selected_topics:
        filtered_df = df[df['Topic_Name'].isin(selected_topics)]
    else:
        filtered_df = df
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Questions", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Topics Found", len(df['Topic_Name'].unique()))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        categorized = len(df[df['Topic_ID'] != -1])
        st.metric("Categorized", f"{categorized} ({categorized/len(df)*100:.1f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_confidence = df['Probability'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Topic Overview
    st.header("📈 Topic Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Topic distribution chart
        topic_counts = filtered_df.groupby('Topic_Name').size().reset_index(name='Count')
        topic_counts = topic_counts.sort_values('Count', ascending=True)
        
        fig = px.bar(
            topic_counts,
            x='Count',
            y='Topic_Name',
            orientation='h',
            title="Questions per Topic",
            color='Count',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            height=500,
            showlegend=False,
            yaxis_title=None,
            xaxis_title="Number of Questions"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top topics
        st.subheader("🏆 Top Topics")
        for idx, row in topic_counts.tail(5).iterrows():
            st.markdown(f"**{row['Topic_Name']}**")
            st.write(f"📊 {row['Count']} questions")
            st.markdown("---")
    
    st.divider()
    
    # Interactive Scatter Plot
    if 'UMAP_X' in df.columns and 'UMAP_Y' in df.columns:
        st.header("🗺️ Question Clusters Map")
        
        # Create scatter plot
        fig = px.scatter(
            filtered_df,
            x='UMAP_X',
            y='UMAP_Y',
            color='Topic_Name',
            hover_data={
                'Question': True,
                'Topic_Name': True,
                'Probability': ':.2f',
                'UMAP_X': False,
                'UMAP_Y': False
            },
            title="Interactive Question Clusters (hover to see questions)",
            width=800,
            height=600
        )
        
        fig.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>' +
                         'Topic: %{customdata[1]}<br>' +
                         'Confidence: %{customdata[2]}<br>' +
                         '<extra></extra>',
            hovertext=filtered_df['Question'].str[:100] + '...'
        )
        
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Question Browser
    st.header("🔍 Question Explorer")
    
    # Topic filter for questions
    topic_filter = st.selectbox(
        "Select Topic to Explore",
        ["All Topics"] + sorted(df['Topic_Name'].unique().tolist())
    )
    
    if topic_filter == "All Topics":
        questions_to_show = df
    else:
        questions_to_show = df[df['Topic_Name'] == topic_filter]
    
    # Search functionality
    search_term = st.text_input("🔎 Search questions", placeholder="Enter keywords to search...")
    
    if search_term:
        mask = questions_to_show['Question'].str.contains(search_term, case=False, na=False)
        questions_to_show = questions_to_show[mask]
    
    # Display questions
    st.subheader(f"📝 Questions ({len(questions_to_show)} found)")
    
    if len(questions_to_show) > 0:
        # Pagination
        questions_per_page = 10
        total_pages = (len(questions_to_show) - 1) // questions_per_page + 1
        
        page = st.selectbox(
            f"Page (showing {questions_per_page} per page)",
            range(1, total_pages + 1)
        )
        
        start_idx = (page - 1) * questions_per_page
        end_idx = start_idx + questions_per_page
        page_questions = questions_to_show.iloc[start_idx:end_idx]
        
        for idx, row in page_questions.iterrows():
            with st.expander(f"Q{idx+1}: {row['Question'][:100]}{'...' if len(row['Question']) > 100 else ''}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("**Full Question:**")
                    st.write(row['Question'])
                
                with col2:
                    st.write("**Topic:**")
                    st.write(row['Topic_Name'])
                    st.write("**Confidence:**")
                    st.write(f"{row['Probability']:.2f}")
    else:
        st.info("No questions found matching your criteria.")
    
    st.divider()
    
    # Data Export
    st.header("💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Download Current View as CSV", type="primary"):
            csv = questions_to_show.to_csv(index=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"pathway_questions_filtered_{timestamp}.csv"
            
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Download Full Analysis", type="secondary"):
            csv = df.to_csv(index=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"pathway_questions_complete_{timestamp}.csv"
            
            st.download_button(
                label="⬇️ Download Complete CSV",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 2rem;">
        <p>🎓 BYU Pathway Questions Analysis Dashboard</p>
        <p><small>Powered by OpenAI, BERTopic, and Streamlit • Built for Educational Insights</small></p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header with reduced spacing
    st.markdown('<h1 class="main-header" style="margin-bottom: 0.5rem;">🎓 BYU Pathway Questions Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header" style="margin-bottom: 1rem;">AI-Powered Topic Discovery and Question Clustering</p>', unsafe_allow_html=True)
    
    # Check for API key
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        st.error("🔑 OpenAI API key not found. Please add your API key to the .env file.")
        st.code("OPENAI_API_KEY=your_api_key_here")
        st.info("💡 **Need an interactive analysis session?** [Open in Google Colab](https://colab.research.google.com/drive/1TPGv-qhQXlJe5Z3OAOrZ1O6USNlwHl21?usp=sharing) - `BYU_Pathway_Topic_Modeling_Colab.ipynb`")
        return
    
    # Initialize session state for tab control
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    
    # Create tabs with better organization
    if st.session_state.analysis_complete:
        # After analysis, show results-focused tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis Results", "🔍 Explore Questions", "📈 Visualizations", "📤 Upload New File"])
        
        with tab1:
            display_analysis_summary(st.session_state.current_results)
        
        with tab2:
            display_question_explorer(st.session_state.current_results)
        
        with tab3:
            display_visualizations(st.session_state.current_results)
        
        with tab4:
            upload_and_analyze_tab()
    else:
        # Before analysis, focus on upload
        tab1, tab2 = st.tabs(["📤 Upload & Analyze", "📊 View Previous Results"])
        
        with tab1:
            upload_and_analyze_tab()
        
        with tab2:
            view_previous_results_tab()

def upload_and_analyze_tab():
    """Upload and analyze tab with clean UI flow"""
    st.header("📤 Upload Questions File")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a text file with questions (one per line)",
        type=['txt'],
        help="Upload a .txt file with one question per line. Recommended: 50+ questions for meaningful analysis.",
        key="main_file_uploader"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        analyze_button = st.button(
            "🚀 Run Analysis",
            type="primary",
            disabled=(uploaded_file is None),
            help="Upload a file first to enable analysis"
        )
    
    with col2:
        st.markdown("💡 **Need more control?**")
        st.markdown("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TPGv-qhQXlJe5Z3OAOrZ1O6USNlwHl21?usp=sharing) `BYU_Pathway_Topic_Modeling_Colab.ipynb`")
    
    if uploaded_file is not None:
        # Show file preview
        content = uploaded_file.read().decode('utf-8')
        uploaded_file.seek(0)  # Reset file pointer
        
        questions = [line.strip() for line in content.split('\n') if line.strip()]
        
        st.success(f"✅ File loaded: {len(questions)} questions found")
        
        with st.expander("📖 Preview first 5 questions"):
            for i, q in enumerate(questions[:5], 1):
                st.write(f"{i}. {q}")
    
    if analyze_button and uploaded_file is not None:
        st.divider()
        st.header("🔄 Analysis in Progress")
        
        # Run analysis
        with st.spinner("Processing your questions..."):
            result = process_questions_file(uploaded_file)
        
        if result is not None:
            results_df, topic_model = result
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            csv_path = results_dir / f"pathway_questions_analysis_{timestamp}.csv"
            results_df.to_csv(csv_path, index=False)
            
            st.success(f"✅ Analysis complete! Results saved to {csv_path}")
            
            # Store results in session state and trigger tab switch
            st.session_state.current_results = results_df
            st.session_state.analysis_complete = True
            
            # Show completion message with auto-refresh instruction
            st.balloons()
            st.info("🎉 **Analysis Complete!** The page will automatically reorganize to show your results.")
            
            # Force rerun to switch to results view
            time.sleep(1)
            st.rerun()

def view_previous_results_tab():
    """View previously saved results"""
    st.header("📊 Previous Analysis Results")
    
    # Load existing results
    df = load_analysis_results()
    
    if df is None:
        st.info("📊 No previous analysis results found.")
        st.markdown("""
        **Get started by:**
        1. 📤 Switch to the "Upload & Analyze" tab to analyze new questions
        2. 📓 Or run the Jupyter notebook locally with `make run-notebook`
        3. 🌐 For interactive analysis, use the [Google Colab notebook](https://colab.research.google.com)
        """
        )
    else:
        # Quick action to load previous results
        if st.button("🔄 Load Previous Results", type="primary"):
            st.session_state.current_results = df
            st.session_state.analysis_complete = True
            st.rerun()
        
        # Show basic info about previous results
        st.write(f"**📋 Last Analysis:** {len(df)} questions, {len(df['Topic_Name'].unique())} topics")
        
        # Show summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Questions", len(df))
        
        with col2:
            st.metric("Topics Found", len(df['Topic_Name'].unique()))
        
        with col3:
            categorized = len(df[df['Topic_ID'] != -1])
            st.metric("Categorized", f"{categorized/len(df)*100:.1f}%")

def display_analysis_summary(df):
    """Display analysis summary and key metrics"""
    st.header("📊 Analysis Summary")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Questions", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Topics Found", len(df['Topic_Name'].unique()))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        categorized = len(df[df['Topic_ID'] != -1])
        st.metric("Categorized", f"{categorized} ({categorized/len(df)*100:.1f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_confidence = df['Probability'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
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
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Enhanced topic distribution chart
        fig = px.bar(
            topic_counts.head(15),
            x='Count',
            y='Topic_Name',
            orientation='h',
            title="Top 15 Topics by Question Count",
            color='Count',
            color_continuous_scale='Blues',
            text='Count'
        )
        fig.update_layout(
            height=500,
            showlegend=False,
            yaxis_title=None,
            xaxis_title="Number of Questions"
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Topic confidence analysis
        st.subheader("📈 Topic Confidence Distribution")
        fig_conf = px.box(
            df[df['Topic_ID'] != -1],
            x='Topic_Name',
            y='Probability',
            title="Confidence Distribution by Topic"
        )
        fig_conf.update_layout(
            height=400,
            xaxis_title="Topics",
            yaxis_title="Confidence Score",
            xaxis_tickangle=45
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    
    with col2:
        # Scrollable topic list in its own container
        st.subheader("📋 All Topics")
        
        # Simple topic list without complex containers
        
        # Display topics as simple list items
        for _, row in topic_analysis.iterrows():
            confidence_color = "🟢" if row['Avg_Confidence'] > 0.7 else "🟡" if row['Avg_Confidence'] > 0.5 else "🔴"
            
            st.markdown(f"""
            **{row['Topic_Name']}**  
            📊 {row['Count']} questions | {confidence_color} {row['Avg_Confidence']:.2f} avg confidence | 📈 {row['Std_Confidence']:.2f} std dev
            """)
        
            st.markdown("""
            <style>
            .scrollable-topics {
                height: 380px;
                overflow-y: auto;
                overflow-x: hidden;
                padding: 10px;
                margin-top: -420px;
                margin-left: 16px;
                margin-right: 16px;
                position: relative;
                z-index: 10;
            }
            .topic-card {
                background-color: white;
                padding: 12px;
                margin-bottom: 8px;
                border-radius: 6px;
                border-left: 4px solid #007bff;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .topic-title {
                font-weight: bold;
                color: #333;
                font-size: 14px;
                margin-bottom: 4px;
            }
            .topic-stats {
                font-size: 12px;
                color: #666;
                line-height: 1.4;
            }
            </style>
            <div class="scrollable-topics">
            """, unsafe_allow_html=True)
            
            # Render each topic as individual components
            for _, row in topic_analysis.iterrows():
                confidence_color = "🟢" if row['Avg_Confidence'] > 0.7 else "�" if row['Avg_Confidence'] > 0.5 else "🔴"
                
                st.markdown(f"""
                <div class="topic-card">
                    <div class="topic-title">{row['Topic_Name']}</div>
                    <div class="topic-stats">
                        📊 {row['Count']} questions | 
                        {confidence_color} {row['Avg_Confidence']:.2f} avg confidence |
                        📈 {row['Std_Confidence']:.2f} std dev
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Quick stats
        st.subheader("📊 Quick Stats")
        st.write(f"**Largest Topic:** {topic_counts.iloc[0]['Topic_Name']} ({topic_counts.iloc[0]['Count']} questions)")
        st.write(f"**Smallest Topic:** {topic_counts.iloc[-1]['Topic_Name']} ({topic_counts.iloc[-1]['Count']} questions)")
        st.write(f"**Most Confident Topic:** {topic_analysis.loc[topic_analysis['Avg_Confidence'].idxmax(), 'Topic_Name']} ({topic_analysis['Avg_Confidence'].max():.2f})")
        st.write(f"**Least Confident Topic:** {topic_analysis.loc[topic_analysis['Avg_Confidence'].idxmin(), 'Topic_Name']} ({topic_analysis['Avg_Confidence'].min():.2f})")
    
    # Additional Analysis
    st.divider()
    st.subheader("🔍 Advanced Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Topic size distribution
        st.write("**Topic Size Distribution**")
        size_bins = pd.cut(topic_counts['Count'], bins=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
        size_dist = size_bins.value_counts().sort_index()
        fig_size = px.pie(
            values=size_dist.values,
            names=size_dist.index,
            title="Topics by Size Category"
        )
        fig_size.update_layout(height=300)
        st.plotly_chart(fig_size, use_container_width=True)
    
    with col2:
        # Confidence distribution
        st.write("**Overall Confidence Distribution**")
        fig_hist = px.histogram(
            df[df['Topic_ID'] != -1],
            x='Probability',
            nbins=20,
            title="Question Confidence Distribution"
        )
        fig_hist.update_layout(
            height=300,
            xaxis_title="Confidence Score",
            yaxis_title="Number of Questions"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col3:
        # Topic diversity metrics
        st.write("**Topic Diversity Metrics**")
        total_questions = len(df)
        noise_ratio = len(df[df['Topic_ID'] == -1]) / total_questions
        largest_topic_ratio = topic_counts.iloc[0]['Count'] / total_questions
        
        st.metric("Noise Ratio", f"{noise_ratio:.1%}")
        st.metric("Largest Topic Share", f"{largest_topic_ratio:.1%}")
        st.metric("Topic Balance", f"{1 - largest_topic_ratio:.1%}")
        
        # Diversity score (entropy-like measure)
        topic_probs = topic_counts['Count'] / topic_counts['Count'].sum()
        diversity_score = -(topic_probs * np.log2(topic_probs + 1e-10)).sum()
        st.metric("Diversity Score", f"{diversity_score:.2f}")
    
    # Data Export
    st.divider()
    st.subheader("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df.to_csv(index=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"pathway_questions_complete_{timestamp}.csv"
        
        st.download_button(
            label="📊 Download Complete Analysis",
            data=csv,
            file_name=filename,
            mime="text/csv",
            type="primary"
        )
        
        # Quick stats
        st.subheader("📊 Quick Stats")
        st.write(f"**Largest Topic:** {topic_counts.iloc[0]['Topic_Name']} ({topic_counts.iloc[0]['Count']} questions)")
        st.write(f"**Smallest Topic:** {topic_counts.iloc[-1]['Topic_Name']} ({topic_counts.iloc[-1]['Count']} questions)")
        st.write(f"**Most Confident Topic:** {topic_analysis.loc[topic_analysis['Avg_Confidence'].idxmax(), 'Topic_Name']} ({topic_analysis['Avg_Confidence'].max():.2f})")
        st.write(f"**Least Confident Topic:** {topic_analysis.loc[topic_analysis['Avg_Confidence'].idxmin(), 'Topic_Name']} ({topic_analysis['Avg_Confidence'].min():.2f})")
    
    with col3:
        # Export topic summary
        topic_summary_csv = topic_analysis.to_csv(index=False)
        st.download_button(
            label="📈 Download Topic Summary",
            data=topic_summary_csv,
            file_name=f"topic_summary_{timestamp}.csv",
            mime="text/csv"
        )
        
        if st.button("🔄 Analyze New Questions", type="secondary"):
            st.session_state.analysis_complete = False
            st.session_state.current_results = None
            st.rerun()

def display_question_explorer(df):
    """Display question exploration interface"""
    st.header("🔍 Question Explorer")
    
    # Topic filter for questions
    topic_filter = st.selectbox(
        "Select Topic to Explore",
        ["All Topics"] + sorted(df['Topic_Name'].unique().tolist()),
        key="topic_explorer_selectbox"
    )
    
    if topic_filter == "All Topics":
        questions_to_show = df
    else:
        questions_to_show = df[df['Topic_Name'] == topic_filter]
    
    # Search functionality
    search_term = st.text_input(
        "🔎 Search questions", 
        placeholder="Enter keywords to search...",
        key="question_search_input"
    )
    
    if search_term:
        mask = questions_to_show['Question'].str.contains(search_term, case=False, na=False)
        questions_to_show = questions_to_show[mask]
    
    # Display questions
    st.subheader(f"📝 Questions ({len(questions_to_show)} found)")
    
    if len(questions_to_show) > 0:
        # Pagination
        questions_per_page = 10
        total_pages = (len(questions_to_show) - 1) // questions_per_page + 1
        
        page = st.selectbox(
            f"Page (showing {questions_per_page} per page)",
            range(1, total_pages + 1),
            key="question_pagination_selectbox"
        )
        
        start_idx = (page - 1) * questions_per_page
        end_idx = start_idx + questions_per_page
        page_questions = questions_to_show.iloc[start_idx:end_idx]
        
        for idx, row in page_questions.iterrows():
            with st.expander(f"Q{idx+1}: {row['Question'][:100]}{'...' if len(row['Question']) > 100 else ''}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("**Full Question:**")
                    st.write(row['Question'])
                
                with col2:
                    st.write("**Topic:**")
                    st.write(row['Topic_Name'])
                    st.write("**Confidence:**")
                    st.write(f"{row['Probability']:.2f}")
        
        # Export filtered results
        if st.button("📥 Download Filtered Results", key="download_filtered"):
            csv = questions_to_show.to_csv(index=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"pathway_questions_filtered_{timestamp}.csv"
            
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
    else:
        st.info("No questions found matching your criteria.")

def display_visualizations(df):
    """Display interactive visualizations"""
    st.header("📊 Advanced Visualizations")
    
    # Load the saved topic model for advanced visualizations
    if os.path.exists('data/topic_model.pkl'):
        with open('data/topic_model.pkl', 'rb') as f:
            topic_model = pickle.load(f)
        
        # Visualization tabs
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["🗺️ Topic Map", "🌳 Hierarchy", "📊 Analysis", "🔗 Relationships"])
        
        with viz_tab1:
            st.subheader("Topic Landscape")
            
            # Sidebar filters for all visualizations
            st.sidebar.title("🎛️ Visualization Controls")
            topics = df['Topic_Name'].unique()
            selected_topics = st.sidebar.multiselect(
                "Filter by Topics",
                topics,
                default=topics[:10] if len(topics) > 10 else topics,
                key="viz_topic_filter"
            )
            
            try:
                fig = topic_model.visualize_topics()
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **Tip:** Use the zoom and pan tools to explore topic clusters. Hover over points to see topic details.")
                
            except Exception as e:
                st.error(f"Could not generate topic visualization: {e}")
            
            # Interactive Scatter Plot (if UMAP coordinates available)
            if 'UMAP_X' in df.columns and 'UMAP_Y' in df.columns:
                st.subheader("🗺️ Question Clusters Map")
                
                # Filter data based on sidebar selection
                if selected_topics:
                    filtered_df = df[df['Topic_Name'].isin(selected_topics)]
                else:
                    filtered_df = df
                
                fig = px.scatter(
                    filtered_df,
                    x='UMAP_X',
                    y='UMAP_Y',
                    color='Topic_Name',
                    hover_data={
                        'Question': True,
                        'Topic_Name': True,
                        'Probability': ':.2f',
                        'UMAP_X': False,
                        'UMAP_Y': False
                    },
                    title="Interactive Question Clusters (hover to see questions)",
                    width=800,
                    height=600
                )
                
                fig.update_traces(
                    hovertemplate='<b>%{hovertext}</b><br>' +
                                 'Topic: %{customdata[1]}<br>' +
                                 'Confidence: %{customdata[2]}<br>' +
                                 '<extra></extra>',
                    hovertext=filtered_df['Question'].str[:100] + '...'
                )
                
                fig.update_layout(
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.01
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab2:
            st.subheader("Topic Hierarchy & Clustering")
            try:
                # Hierarchical clustering visualization
                fig_hier = topic_model.visualize_hierarchy()
                fig_hier.update_layout(height=700)
                st.plotly_chart(fig_hier, use_container_width=True)
                
                st.info("💡 **Tip:** This dendrogram shows how topics relate hierarchically. Similar topics cluster together.")
                
                # Topic similarity heatmap
                if hasattr(topic_model, 'topic_embeddings_') and topic_model.topic_embeddings_ is not None:
                    st.subheader("Topic Similarity Heatmap")
                    
                    # Calculate cosine similarity between topics
                    from sklearn.metrics.pairwise import cosine_similarity
                    
                    topic_embeddings = topic_model.topic_embeddings_
                    similarity_matrix = cosine_similarity(topic_embeddings)
                    
                    # Get topic names
                    topic_info = topic_model.get_topic_info()
                    topic_names = [f"Topic {i}: {name[:30]}..." if len(name) > 30 else f"Topic {i}: {name}" 
                                 for i, name in enumerate(topic_info['Name'][:20])]  # Limit to top 20
                    
                    if len(similarity_matrix) > 20:
                        similarity_matrix = similarity_matrix[:20, :20]
                    
                    fig_heatmap = px.imshow(
                        similarity_matrix,
                        labels=dict(x="Topics", y="Topics", color="Similarity"),
                        x=topic_names,
                        y=topic_names,
                        title="Topic Similarity Matrix",
                        color_continuous_scale='RdYlBu_r',
                        aspect="auto"
                    )
                    fig_heatmap.update_layout(
                        height=600,
                        xaxis_tickangle=45
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                
            except Exception as e:
                st.error(f"Could not generate hierarchy visualization: {e}")
        
        with viz_tab3:
            st.subheader("Statistical Analysis")
            
            # Enhanced topic distribution
            topic_counts = df.groupby('Topic_Name').size().reset_index(name='Count')
            topic_counts = topic_counts.sort_values('Count', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart for top topics
                fig_pie = px.pie(
                    topic_counts.head(10),
                    values='Count',
                    names='Topic_Name',
                    title="Top 10 Topics Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_layout(height=500)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Topic size distribution histogram
                fig_hist = px.histogram(
                    topic_counts,
                    x='Count',
                    nbins=15,
                    title="Topic Size Distribution",
                    labels={'Count': 'Questions per Topic', 'count': 'Number of Topics'}
                )
                fig_hist.update_layout(height=500)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Confidence analysis
            st.subheader("🎯 Confidence Score Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Box plot of confidence by topic
                filtered_df = df[df['Topic_ID'] != -1]
                top_topics = topic_counts.head(10)['Topic_Name'].tolist()
                filtered_df = filtered_df[filtered_df['Topic_Name'].isin(top_topics)]
                
                fig_box = px.box(
                    filtered_df,
                    x='Topic_Name',
                    y='Probability',
                    title="Confidence Distribution (Top 10 Topics)",
                    color='Topic_Name',
                    showlegend=False
                )
                fig_box.update_layout(
                    height=500,
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                # Scatter plot: topic size vs average confidence
                topic_stats = df.groupby('Topic_Name').agg({
                    'Probability': 'mean',
                    'Topic_ID': 'count'
                }).reset_index()
                topic_stats.columns = ['Topic_Name', 'Avg_Confidence', 'Question_Count']
                
                fig_scatter = px.scatter(
                    topic_stats,
                    x='Question_Count',
                    y='Avg_Confidence',
                    size='Question_Count',
                    hover_data=['Topic_Name'],
                    title="Topic Size vs Average Confidence",
                    labels={'Question_Count': 'Number of Questions', 'Avg_Confidence': 'Average Confidence'}
                )
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        with viz_tab4:
            st.subheader("Topic Relationships & Keywords")
            
            # Try to create topic barchart if possible
            try:
                # Barchart visualization for top topics
                fig_bar = topic_model.visualize_barchart(top_k_topics=10, n_words=10)
                fig_bar.update_layout(height=800)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                st.info("💡 **Tip:** This chart shows the most important words for each topic, helping understand topic themes.")
                
            except Exception as e:
                st.warning(f"Advanced topic visualization not available: {e}")
            
            # Word frequency analysis
            st.subheader("🔤 Keyword Analysis")
            
            # Get all topic keywords
            try:
                topic_info = topic_model.get_topic_info()
                all_keywords = []
                
                for topic_id in topic_info['Topic'][:10]:  # Top 10 topics
                    if topic_id != -1:
                        keywords = topic_model.get_topic(topic_id)
                        for word, score in keywords[:5]:  # Top 5 words per topic
                            all_keywords.append({'word': word, 'score': score, 'topic_id': topic_id})
                
                if all_keywords:
                    keywords_df = pd.DataFrame(all_keywords)
                    word_freq = keywords_df.groupby('word')['score'].sum().reset_index()
                    word_freq = word_freq.sort_values('score', ascending=False).head(20)
                    
                    fig_words = px.bar(
                        word_freq,
                        x='score',
                        y='word',
                        orientation='h',
                        title="Top 20 Keywords Across All Topics",
                        labels={'score': 'Cumulative Importance Score', 'word': 'Keywords'}
                    )
                    fig_words.update_layout(height=600)
                    st.plotly_chart(fig_words, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Keyword analysis not available: {e}")
                
    else:
        st.warning("⚠️ Topic model not found. Please run analysis first in the 'Upload & Analyze' tab.")
    
    # Quality metrics summary
    if os.path.exists('data') and 'quality_metrics.json' in os.listdir('data'):
        try:
            with open('data/quality_metrics.json', 'r') as f:
                metrics = json.load(f)
            
            st.divider()
            st.subheader("📊 Model Quality Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Coherence Score", f"{metrics.get('coherence_score', 'N/A')}")
            with col2:
                st.metric("Silhouette Score", f"{metrics.get('silhouette_score', 'N/A')}")
            with col3:
                st.metric("Topics Created", f"{metrics.get('n_topics', 'N/A')}")
            with col4:
                st.metric("Noise Ratio", f"{metrics.get('noise_ratio', 'N/A')}")
                
        except Exception as e:
            st.info("Quality metrics not available for this analysis.")

def display_results(df):
    """Legacy function - now handled by separate tab functions"""
    display_analysis_summary(df)

if __name__ == "__main__":
    main()
