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

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 BYU Pathway Questions Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Topic Discovery and Question Clustering</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_analysis_results()
    
    if df is None:
        st.error("📊 No analysis results found. Please run the Jupyter notebook first to generate results.")
        st.info("🚀 **How to get started:**")
        st.code("""
# 1. Run the analysis notebook
make run-notebook

# 2. Execute all cells in the notebook

# 3. Return here to view results
        """)
        return
    
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
        st.subheader("🔥 Top Topics")
        top_topics = topic_counts.sort_values('Count', ascending=False).head(5)
        
        for idx, row in top_topics.iterrows():
            topic_name = row['Topic_Name']
            count = row['Count']
            
            st.markdown(f"""
            <div class="topic-card">
                <strong>{topic_name}</strong><br>
                <small>{count} questions</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Question Explorer
    st.header("🔍 Question Explorer")
    
    # Topic selector
    topic_options = ["All Topics"] + sorted(df['Topic_Name'].unique())
    selected_topic = st.selectbox("Select a topic to explore", topic_options)
    
    if selected_topic == "All Topics":
        questions_to_show = df
    else:
        questions_to_show = df[df['Topic_Name'] == selected_topic]
    
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

if __name__ == "__main__":
    main()
