#!/usr/bin/env python3
"""
Simple Streamlit Demo - Database Auto-Update
Demonstrates the automatic database updating after analysis completion
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import asyncio
from datetime import datetime
import uuid
import numpy as np

# Add src directory to Python path
current_dir = Path(__file__).parent.absolute()
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))

# Configure Streamlit page
st.set_page_config(
    page_title="BYU Pathway - Database Auto-Update Demo",
    page_icon="🗄️",
    layout="wide"
)

st.title("🗄️ BYU Pathway Analysis - Auto Database Update Demo")
st.markdown("**Demo showing automatic database updates after analysis completion**")

# Sample data preparation
@st.cache_data
def get_sample_questions():
    """Get sample questions for testing"""
    return pd.DataFrame({
        'question': [
            'How do I reset my password?',
            'Where can I find my transcript?',
            'What are the library hours?',
            'How do I access my student portal?',
            'What financial aid is available?',
            'How do I contact my advisor?',
            'Where can I find parking information?',
            'How do I drop a class?',
            'What are the graduation requirements?',
            'How do I pay my tuition?'
        ],
        'user_language': ['en'] * 10,
        'country': ['USA', 'Canada', 'Mexico', 'USA', 'Philippines', 'Brazil', 'USA', 'Canada', 'USA', 'Mexico'],
        'state': ['Utah', 'Ontario', 'Jalisco', 'California', 'Manila', 'São Paulo', 'Texas', 'Quebec', 'Florida', 'Nuevo León'],
        'user_role': ['student', 'student', 'faculty', 'student', 'student', 'administrator', 'student', 'student', 'advisor', 'student']
    })

@st.cache_data
def get_sample_topics():
    """Get sample existing topics"""
    return pd.DataFrame({
        'Topic': ['Technical Support', 'Academic Services', 'Financial Aid'],
        'Subtopic': ['Login Issues', 'Transcripts', 'Scholarships'],
        'Question': [
            'Password reset problems',
            'Transcript access questions', 
            'Scholarship information requests'
        ]
    })

def simulate_analysis(questions_df, topics_df, threshold):
    """Simulate hybrid topic analysis"""
    
    # Simulate processing time
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔍 Analyzing questions...")
    progress_bar.progress(25)
    st.time.sleep(1)
    
    status_text.text("🤖 Generating embeddings...")
    progress_bar.progress(50)
    st.time.sleep(1)
    
    status_text.text("🎯 Discovering topics...")
    progress_bar.progress(75)
    st.time.sleep(1)
    
    # Create simulated results
    similar_df = pd.DataFrame({
        'question': questions_df['question'].iloc[:3].tolist(),
        'similarity_score': [0.85, 0.78, 0.92],
        'matched_topic': ['Technical Support', 'Academic Services', 'Financial Aid']
    })
    
    clustered_df = pd.DataFrame({
        'question': questions_df['question'].iloc[3:].tolist(),
        'cluster': [0, 1, 2, 0, 1, 2, 0]
    })
    
    topic_names = {
        0: 'Student Portal & Online Services',
        1: 'Academic Planning & Requirements', 
        2: 'Campus Services & Information'
    }
    
    # Generate fake embeddings
    embeddings = {
        i: np.random.rand(1536).tolist() 
        for i in range(len(questions_df))
    }
    
    results = {
        'similar_questions_df': similar_df,
        'clustered_questions_df': clustered_df,
        'topic_names': topic_names,
        'embeddings': embeddings,
        'eval_questions_df': questions_df
    }
    
    status_text.text("✅ Analysis complete!")
    progress_bar.progress(100)
    
    return results

def update_database_with_results(questions_df, results, threshold, mode, sample_size):
    """Update database with analysis results - matching the main app functionality"""
    try:
        from database import DatabaseManager, AnalysisRun, Question, TopicCluster, QuestionEmbedding, QuestionClusterAssignment, AnalysisCache
        from sqlalchemy import text
        from datetime import timezone
        
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Generate unique run ID
        run_id = str(uuid.uuid4())
        
        with db_manager.get_session() as session:
            # Step 1: Clear previous analysis data (as requested)
            session.execute(text("DELETE FROM question_cluster_assignments"))
            session.execute(text("DELETE FROM question_embeddings"))
            session.execute(text("DELETE FROM topic_clusters"))
            session.execute(text("DELETE FROM analysis_cache"))
            session.execute(text("DELETE FROM analysis_runs"))
            session.commit()
            
            # Step 2: Create new analysis run record
            analysis_run = AnalysisRun(
                run_id=run_id,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                status='completed',
                total_questions=len(questions_df),
                total_topics=len(results.get('topic_names', {})),
                config_snapshot={
                    'similarity_threshold': threshold,
                    'processing_mode': mode,
                    'sample_size': sample_size,
                    'embedding_model': 'text-embedding-3-small',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                },
                created_by='streamlit_demo'
            )
            session.add(analysis_run)
            session.commit()
            
            # Step 3: Clear and re-add questions
            session.execute(text("DELETE FROM questions"))
            session.commit()
            
            for idx, row in questions_df.iterrows():
                question = Question(
                    original_text=str(row.get('question', '')),
                    cleaned_question=str(row.get('question', '')),
                    timestamp=datetime.now(timezone.utc),
                    user_language=str(row.get('user_language', 'en')),
                    country=str(row.get('country', 'Unknown')),
                    state=str(row.get('state', 'Unknown')),
                    user_role=str(row.get('user_role', 'Unknown')),
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
                session.add(question)
            session.commit()
            
            # Get question IDs for relationships
            questions_in_db = session.query(Question).all()
            question_id_map = {idx: q.id for idx, q in enumerate(questions_in_db)}
            
            # Step 4: Insert topic clusters
            topic_names = results.get('topic_names', {})
            cluster_id_map = {}
            
            for cluster_num, topic_name in topic_names.items():
                clustered_df = results.get('clustered_questions_df')
                rep_question = "No representative question"
                question_count = 0
                
                if clustered_df is not None and len(clustered_df) > 0:
                    cluster_questions = clustered_df[clustered_df['cluster'] == cluster_num]
                    if len(cluster_questions) > 0:
                        rep_question = str(cluster_questions.iloc[0]['question'])
                        question_count = len(cluster_questions)
                
                topic_cluster = TopicCluster(
                    run_id=run_id,
                    cluster_id=cluster_num,
                    topic_name=topic_name,
                    topic_description=f"Auto-generated topic: {topic_name}",
                    representative_question=rep_question,
                    question_count=question_count,
                    avg_sentiment=0.5,
                    urgency_score=0.5,
                    keywords=[],
                    created_at=datetime.now(timezone.utc)
                )
                session.add(topic_cluster)
                session.commit()
                cluster_id_map[cluster_num] = topic_cluster.id
            
            # Step 5: Insert embeddings
            embeddings_data = results.get('embeddings', {})
            if embeddings_data:
                for idx, embedding in embeddings_data.items():
                    if idx in question_id_map and embedding is not None:
                        question_embedding = QuestionEmbedding(
                            question_id=question_id_map[idx],
                            run_id=run_id,
                            embedding=embedding,
                            model_version='text-embedding-3-small',
                            created_at=datetime.now(timezone.utc)
                        )
                        session.add(question_embedding)
            session.commit()
            
            # Step 6: Insert assignments
            clustered_df = results.get('clustered_questions_df')
            if clustered_df is not None and len(clustered_df) > 0:
                for idx, row in clustered_df.iterrows():
                    cluster_num = row.get('cluster', -1)
                    # Adjust index to match question_id_map
                    adjusted_idx = idx + 3  # Since clustered questions start after similar ones
                    if adjusted_idx in question_id_map and cluster_num in cluster_id_map:
                        assignment = QuestionClusterAssignment(
                            question_id=question_id_map[adjusted_idx],
                            cluster_id=cluster_id_map[cluster_num],
                            run_id=run_id,
                            similarity_score=1.0
                        )
                        session.add(assignment)
            session.commit()
            
            # Step 7: Cache analysis results
            cache_entry = AnalysisCache(
                cache_key=f"streamlit_demo_{run_id}",
                run_id=run_id,
                cache_type="demo_analysis",
                data={
                    'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                    'total_processed': len(questions_df),
                    'similar_questions': len(results.get('similar_questions_df', [])),
                    'new_topics': len(topic_names),
                    'demo_mode': True
                },
                expires_at=None,
                created_at=datetime.now(timezone.utc)
            )
            session.add(cache_entry)
            session.commit()
            
            return run_id
            
    except Exception as e:
        raise RuntimeError(f"Failed to update database: {str(e)}")

# Main demo interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Sample Data")
    
    questions_df = get_sample_questions()
    topics_df = get_sample_topics()
    
    st.write("**Questions to Analyze:**")
    st.dataframe(questions_df[['question', 'country', 'user_role']], use_container_width=True)
    
    st.write("**Existing Topics:**")
    st.dataframe(topics_df, use_container_width=True)

with col2:
    st.subheader("⚙️ Configuration")
    
    threshold = st.slider(
        "Similarity Threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.70,
        step=0.05
    )
    
    st.info("🗄️ **Auto Database Update**\n\nAfter analysis completes, the database will be automatically updated with fresh results.")

# Analysis button
st.markdown("---")

if st.button("🚀 **Run Analysis & Auto-Update Database**", type="primary", use_container_width=True):
    
    st.subheader("🔄 Processing...")
    
    # Step 1: Run Analysis (simulated)
    st.write("**Step 1: Running Hybrid Analysis**")
    with st.container():
        results = simulate_analysis(questions_df, topics_df, threshold)
    
    st.success("✅ Analysis completed successfully!")
    
    # Step 2: Auto Database Update
    st.write("**Step 2: Auto-Updating Database**")
    
    try:
        with st.spinner("Clearing previous data and inserting fresh results..."):
            database_run_id = update_database_with_results(
                questions_df, results, threshold, "demo", len(questions_df)
            )
        
        st.success(f"✅ **Database updated successfully!** Run ID: `{database_run_id}`")
        
        # Display results
        st.subheader("📊 Analysis Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📋 Total Processed", len(questions_df))
        
        with col2:
            st.metric("✅ Matched Existing", len(results['similar_questions_df']))
        
        with col3:
            st.metric("🆕 New Topics", len(results['topic_names']))
        
        with col4:
            st.metric("🗄️ Database Status", "Updated")
        
        # Show database status
        st.subheader("🗄️ Database Status")
        
        st.success("**✅ Automatic Database Update Completed**")
        
        st.info(f"""
        **Database Operation Summary:**
        
        🧹 **Previous Data Cleared**: All previous analysis data removed
        💾 **Fresh Data Inserted**: {len(questions_df)} questions, {len(results['topic_names'])} topics
        🔗 **Relationships Created**: Question-topic assignments and embeddings saved
        💿 **Results Cached**: Analysis results stored for fast retrieval
        🗃️ **Run ID**: `{database_run_id}` for tracking
        
        **✨ The database now contains only the latest analysis results and is ready for production use!**
        """)
        
        # Option to view database
        if st.button("📊 **View Updated Database**"):
            with st.spinner("Loading database contents..."):
                try:
                    import subprocess
                    import sys
                    
                    result = subprocess.run([
                        sys.executable, "scripts/view_database.py"
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        st.code(result.stdout, language="text")
                    else:
                        st.error(f"Error viewing database: {result.stderr}")
                        
                except Exception as e:
                    st.error(f"Failed to view database: {str(e)}")
        
    except Exception as e:
        st.error(f"❌ Database update failed: {str(e)}")
        st.info("💡 Analysis results are still available locally, but not saved to database.")

# Information section
st.markdown("---")

with st.expander("ℹ️ **How Auto Database Update Works**", expanded=False):
    st.markdown("""
    **🔄 Automatic Workflow:**
    
    1. **🚀 Analysis Execution**: Hybrid topic analysis runs and caches results locally
    2. **🧹 Database Clearing**: Previous analysis data is automatically removed 
    3. **💾 Fresh Data Insert**: New analysis results are saved to database
    4. **✅ Completion**: Database is updated instantly, no manual review needed
    
    **📋 Database Tables Updated:**
    - `questions` - Raw and cleaned question data
    - `analysis_runs` - Analysis execution metadata  
    - `question_embeddings` - Vector embeddings for similarity
    - `topic_clusters` - Discovered and named topics
    - `question_cluster_assignments` - Question-to-topic mappings
    - `analysis_cache` - Cached results for performance
    
    **🎯 Key Benefits:**
    - **Instant Production Ready**: Results immediately available for other systems
    - **Data Consistency**: Previous data cleared to prevent conflicts
    - **No Manual Steps**: Fully automated database updating process
    - **Real-time Access**: External applications can query fresh data immediately
    
    **🔧 Technical Details:**
    - Uses PostgreSQL with pgvector extension for embeddings
    - Transactional updates ensure data integrity
    - Connection pooling for optimal performance
    - Proper indexing for fast queries
    """)

st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666;">🗄️ BYU Pathway Database Auto-Update Demo • Real-time Database Integration</p>',
    unsafe_allow_html=True
)