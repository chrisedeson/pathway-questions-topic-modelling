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
    # Header
    st.markdown('<h1 class="main-header">🎓 BYU Pathway Questions Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Topic Discovery and Question Clustering</p>', unsafe_allow_html=True)
    
    # Check for API key
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        st.error("🔑 OpenAI API key not found. Please add your API key to the .env file.")
        st.code("OPENAI_API_KEY=your_api_key_here")
        st.info("💡 **Need an interactive analysis session?** [Open in Google Colab](https://colab.research.google.com) - Upload the `BYU_Pathway_Topic_Modeling_Colab.ipynb` file")
        return
    
    # Create tabs for different modes
    tab1, tab2 = st.tabs(["📊 View Results", "📤 Upload & Analyze"])
    
    with tab1:
        # Load existing results
        df = load_analysis_results()
        
        if df is None:
            st.info("📊 No previous analysis results found.")
            st.markdown("""
            **Get started by:**
            1. 📤 Switch to the "Upload & Analyze" tab to analyze new questions
            2. 📓 Or run the Jupyter notebook locally with `make run-notebook`
            3. 🌐 For interactive analysis, use the [Google Colab notebook](https://colab.research.google.com)
            """)
        else:
            display_results(df)
    
    with tab2:
        st.header("📤 Upload Questions File")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a text file with questions (one per line)",
            type=['txt'],
            help="Upload a .txt file with one question per line. Recommended: 50+ questions for meaningful analysis."
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
            st.markdown("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) Upload `BYU_Pathway_Topic_Modeling_Colab.ipynb`")
        
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
                
                # Display results
                display_results(results_df)

if __name__ == "__main__":
    main()
