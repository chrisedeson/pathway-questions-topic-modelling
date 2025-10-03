"""
Demo: Smart Streamlit Cache + Background Database Updates

This demo shows how the new system works:
1. Fast Streamlit caching for immediate UI response
2. Background database updates (non-blocking)
3. Automatic startup data loading
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import time

# Add src directory to path
current_dir = Path(__file__).parent.absolute()
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))

from streamlit_data_manager import data_manager

def main():
    st.set_page_config(
        page_title="Smart Cache Demo",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Smart Streamlit Cache + Database Demo")
    st.markdown("**Demonstrates fast UI response with background database updates**")
    
    # Sidebar with database status
    with st.sidebar:
        st.subheader("🗄️ Database Status")
        data_manager.display_background_update_status()
        
        # Show database summary
        db_summary = data_manager.get_database_summary()
        if db_summary['status'] == 'connected':
            st.success("✅ Database Connected")
            st.metric("Questions", db_summary.get('questions', 0))
            st.metric("Topics", db_summary.get('topics', 0))
            st.metric("Analysis Runs", db_summary.get('analysis_runs', 0))
        else:
            st.error(f"❌ Database Error: {db_summary.get('error', 'Unknown')}")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Quick Demo", "📊 Previous Analysis", "🔧 Technical Details"])
    
    with tab1:
        st.subheader("Quick Analysis Demo")
        
        # Create sample data
        if st.button("📝 **Create Sample Analysis**", type="primary"):
            # Simulate analysis with immediate cache
            sample_questions = pd.DataFrame({
                'question': [
                    "How do I reset my password?",
                    "Where can I find my grades?",
                    "What financial aid is available?",
                    "How do I contact my advisor?",
                    "When is the deadline for enrollment?"
                ],
                'user_language': ['en'] * 5,
                'country': ['US', 'UK', 'CA', 'AU', 'US'],
                'user_role': ['student'] * 5
            })
            
            # Simulate analysis results (immediate Streamlit cache)
            analysis_results = {
                'eval_questions_df': sample_questions,
                'similar_questions_df': pd.DataFrame(),
                'clustered_questions_df': pd.DataFrame(),
                'topic_names': {0: "Account Access", 1: "Academic Information", 2: "Financial Support"},
                'output_files': [],
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'embeddings': {}  # Empty for demo
            }
            
            # Cache in Streamlit immediately
            st.session_state['demo_results'] = analysis_results
            st.success("✅ **Analysis Complete!** Results cached in Streamlit.")
            
            # Start background database update
            config = {
                'demo': True,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            try:
                run_id = data_manager.update_database_background(
                    sample_questions, analysis_results, config
                )
                st.info(f"🗄️ **Database update started** in background (Run ID: `{run_id[:8]}...`)")
                st.info("💡 **Key Feature**: You can view results immediately while database updates in background!")
                
            except Exception as e:
                st.warning(f"⚠️ Database update failed to start: {str(e)}")
        
        # Show cached results if available
        if 'demo_results' in st.session_state:
            st.subheader("📊 Cached Results (Immediate Access)")
            results = st.session_state['demo_results']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Questions", len(results['eval_questions_df']))
            with col2:
                st.metric("Topics", len(results['topic_names']))
            with col3:
                st.metric("Cache Status", "✅ Active")
            
            # Show questions
            st.dataframe(results['eval_questions_df'], use_container_width=True)
            
            # Clear cache button
            if st.button("🗑️ **Clear Cache**"):
                if 'demo_results' in st.session_state:
                    del st.session_state['demo_results']
                st.rerun()
    
    with tab2:
        st.subheader("📊 Previous Analysis Loading")
        
        # Show startup data loading
        data_manager.display_startup_data_loading()
    
    with tab3:
        st.subheader("🔧 Technical Implementation")
        
        st.markdown("""
        **🚀 Smart Caching Architecture:**
        
        **1. Analysis Workflow:**
        ```
        User clicks "Run Analysis"
              ↓
        Run analysis → Cache in Streamlit (immediate UI response)
              ↓
        Start background database update (non-blocking)
              ↓
        User can view results while DB updates
        ```
        
        **2. Startup Workflow:**
        ```
        App starts → Check database for previous analysis
              ↓
        Load into Streamlit cache (if available)
              ↓
        Fast startup with recent data
        ```
        
        **3. Performance Benefits:**
        - ⚡ **No UI blocking**: Database operations in background threads
        - 🏃 **Fast responses**: Streamlit cache provides immediate access
        - 🔄 **Data consistency**: Fresh data replaces old data automatically
        - 💪 **Robust operation**: Works even if database temporarily unavailable
        
        **4. Cache Strategy:**
        - **L1 Cache**: Streamlit session state (immediate access)
        - **L2 Cache**: Streamlit @st.cache_data (shared across sessions)
        - **L3 Cache**: Database cache tables (persistent)
        """)
        
        # Show current cache status
        st.subheader("💾 Current Cache Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Streamlit Cache:**")
            if 'demo_results' in st.session_state:
                st.success("✅ Demo results cached")
            else:
                st.info("📭 No demo results")
            
            if 'hybrid_results' in st.session_state:
                st.success("✅ Hybrid results cached")
            else:
                st.info("📭 No hybrid results")
        
        with col2:
            st.markdown("**Background Update:**")
            status = data_manager.get_background_update_status()
            
            if status['running']:
                st.info("🔄 Update in progress...")
            elif status['last_sync']:
                st.success(f"✅ Last sync: {status['last_sync'][:19]}")
            else:
                st.info("📭 No recent updates")
        
        # Database test
        st.subheader("🧪 Database Connection Test")
        
        if st.button("Test Connection"):
            try:
                from database import DatabaseManager
                from sqlalchemy import text
                
                db_manager = DatabaseManager()
                with db_manager.get_session() as session:
                    result = session.execute(text("SELECT 1")).scalar()
                    if result == 1:
                        st.success("✅ Database connection successful!")
                        
                        # Show table counts
                        counts = {}
                        tables = ['questions', 'analysis_runs', 'topic_clusters', 'question_embeddings']
                        for table in tables:
                            count = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                            counts[table] = count
                        
                        st.json(counts)
                    else:
                        st.error("❌ Unexpected response")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")


if __name__ == "__main__":
    main()