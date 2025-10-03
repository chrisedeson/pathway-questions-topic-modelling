#!/usr/bin/env python3
"""
Simple Admin App - Sync Google Sheets to Database
Clean, straightforward, no complexity.
"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging

# Setup
st.set_page_config(
    page_title="Pathway Questions Admin",
    page_icon="⚙️",
    layout="wide"
)

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple password protection
ADMIN_PASSWORD = "pathway_admin_2025"

def check_password():
    """Simple password check"""
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.title("🔐 Admin Login")
        password = st.text_input("Enter admin password:", type="password")
        if st.button("Login"):
            if password == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("❌ Invalid password")
        st.stop()

def main():
    """Main admin interface"""
    check_password()
    
    st.title("⚙️ Pathway Questions Admin")
    st.markdown("---")
    
    # Logout button
    col1, col2, col3 = st.columns([3, 1, 1])
    with col3:
        if st.button("🚪 Logout"):
            st.session_state.admin_authenticated = False
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📥 Sync from Sheets", "📊 Database Status", "🧹 Maintenance"])
    
    with tab1:
        show_sync_tab()
    
    with tab2:
        show_database_tab()
    
    with tab3:
        show_maintenance_tab()

def show_sync_tab():
    """Tab for syncing from Google Sheets"""
    st.header("📥 Sync from Google Sheets")
    
    try:
        from google_sheets_utils import GoogleSheetsManager
        from data_cleaning import QuestionCleaner
        from database import get_db_manager
        from config import QUESTIONS_SHEET_ID, TOPICS_SHEET_ID
        from sqlalchemy import text
        
        # Sync Questions
        st.subheader("1️⃣ Questions Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Load from Google Sheets", key="load_questions", type="primary"):
                with st.spinner("Loading questions from Google Sheets..."):
                    try:
                        # Initialize Google Sheets manager
                        sheets_manager = GoogleSheetsManager()
                        
                        # Get sheet URL from ID
                        sheet_url = f"https://docs.google.com/spreadsheets/d/{QUESTIONS_SHEET_ID}"
                        
                        # Load from sheets
                        questions_df, error = sheets_manager.read_questions_from_sheet(sheet_url)
                        
                        if error:
                            st.error(f"❌ Error loading from Google Sheets: {error}")
                        elif questions_df is not None and not questions_df.empty:
                            st.success(f"✅ Loaded {len(questions_df)} questions from Google Sheets")
                            
                            # Store in session for preview
                            st.session_state['questions_preview'] = questions_df
                            
                            # Show preview
                            st.dataframe(questions_df.head(10), use_container_width=True)
                        else:
                            st.error("❌ No questions loaded from Google Sheets")
                            
                    except Exception as e:
                        st.error(f"❌ Error loading from Google Sheets: {e}")
                        logger.error(f"Error loading questions: {e}", exc_info=True)
        
        with col2:
            if st.button("💾 Save to Database", key="save_questions", type="primary"):
                if 'questions_preview' not in st.session_state:
                    st.warning("⚠️ Load questions from Google Sheets first")
                else:
                    with st.spinner("Saving questions to database..."):
                        try:
                            questions_df = st.session_state['questions_preview']
                            
                            # Clean the data
                            cleaner = QuestionCleaner()
                            cleaned_df = cleaner.clean_dataframe(questions_df)
                            
                            # Get database manager
                            db = get_db_manager()
                            
                            # Clear existing questions
                            with db.get_session() as session:
                                session.execute(text("TRUNCATE TABLE questions CASCADE"))
                                st.info("🧹 Cleared existing questions")
                            
                            # Insert new questions
                            from data_service import get_data_service
                            data_service = get_data_service()
                            added, updated, skipped = data_service.store_questions(
                                cleaned_df, 
                                source="google_sheets_admin"
                            )
                            
                            st.success(f"""
                            ✅ **Questions Saved Successfully!**
                            - Added: {added}
                            - Updated: {updated}
                            - Skipped: {skipped}
                            """)
                            
                            # Clear preview
                            del st.session_state['questions_preview']
                            
                        except Exception as e:
                            st.error(f"❌ Error saving to database: {e}")
                            logger.error(f"Error saving questions: {e}", exc_info=True)
        
        st.markdown("---")
        
        # Sync Topics
        st.subheader("2️⃣ Topics Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Load from Google Sheets", key="load_topics", type="primary"):
                with st.spinner("Loading topics from Google Sheets..."):
                    try:
                        # Initialize Google Sheets manager
                        sheets_manager = GoogleSheetsManager()
                        
                        # Get sheet URL from ID
                        sheet_url = f"https://docs.google.com/spreadsheets/d/{TOPICS_SHEET_ID}"
                        
                        # Load from sheets
                        topics_df, error = sheets_manager.read_topics_from_sheet(sheet_url)
                        
                        if error:
                            st.error(f"❌ Error loading from Google Sheets: {error}")
                        elif topics_df is not None and not topics_df.empty:
                            st.success(f"✅ Loaded {len(topics_df)} topic entries from Google Sheets")
                            
                            # Store in session for preview
                            st.session_state['topics_preview'] = topics_df
                            
                            # Show preview
                            st.dataframe(topics_df.head(10), use_container_width=True)
                        else:
                            st.error("❌ No topics loaded from Google Sheets")
                            
                    except Exception as e:
                        st.error(f"❌ Error loading from Google Sheets: {e}")
                        logger.error(f"Error loading topics: {e}", exc_info=True)
        
        with col2:
            st.info("""
            📋 **Topics Reference**
            
            Topics data is used during analysis to match questions to predefined categories.
            This doesn't store in database tables but is loaded during analysis.
            """)
        
    except ImportError as e:
        st.error(f"❌ Missing dependencies: {e}")
        st.info("Make sure all required modules are installed")

def show_database_tab():
    """Tab for viewing database status"""
    st.header("📊 Database Status")
    
    try:
        from database import get_db_manager
        from sqlalchemy import text
        
        db = get_db_manager()
        
        # Test connection
        if db.test_connection():
            st.success("✅ Database connected")
        else:
            st.error("❌ Database connection failed")
            return
        
        # Get counts
        with db.get_session() as session:
            # Questions
            result = session.execute(text("SELECT COUNT(*) FROM questions"))
            questions_count = result.scalar()
            
            # Embeddings
            result = session.execute(text("SELECT COUNT(*) FROM question_embeddings"))
            embeddings_count = result.scalar()
            
            # Clusters
            result = session.execute(text("SELECT COUNT(*) FROM topic_clusters"))
            clusters_count = result.scalar()
            
            # Assignments
            result = session.execute(text("SELECT COUNT(*) FROM question_cluster_assignments"))
            assignments_count = result.scalar()
            
            # Analysis runs
            result = session.execute(text("SELECT COUNT(*) FROM analysis_runs"))
            runs_count = result.scalar()
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📝 Questions", questions_count)
        
        with col2:
            st.metric("🔢 Embeddings", embeddings_count)
        
        with col3:
            st.metric("🏷️ Clusters", clusters_count)
        
        with col4:
            st.metric("🔗 Assignments", assignments_count)
        
        with col5:
            st.metric("📈 Analysis Runs", runs_count)
        
        st.markdown("---")
        
        # Recent questions
        st.subheader("Recent Questions")
        with db.get_session() as session:
            result = session.execute(text("""
                SELECT cleaned_question, timestamp, user_language, country 
                FROM questions 
                ORDER BY timestamp DESC 
                LIMIT 10
            """))
            rows = result.fetchall()
            
            if rows:
                df = pd.DataFrame(rows, columns=['Question', 'Timestamp', 'Language', 'Country'])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No questions in database")
        
        # Recent analysis runs
        st.subheader("Recent Analysis Runs")
        with db.get_session() as session:
            result = session.execute(text("""
                SELECT run_id, status, started_at, completed_at, total_questions, total_topics
                FROM analysis_runs 
                ORDER BY started_at DESC 
                LIMIT 5
            """))
            rows = result.fetchall()
            
            if rows:
                df = pd.DataFrame(rows, columns=['Run ID', 'Status', 'Started', 'Completed', 'Questions', 'Topics'])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No analysis runs in database")
        
    except Exception as e:
        st.error(f"❌ Error getting database status: {e}")
        logger.error(f"Error in database tab: {e}", exc_info=True)

def show_maintenance_tab():
    """Tab for maintenance operations"""
    st.header("🧹 Maintenance")
    
    st.warning("⚠️ **Caution:** These operations will modify or delete data")
    
    try:
        from database import get_db_manager
        from sqlalchemy import text
        
        db = get_db_manager()
        
        # Clear operations
        st.subheader("Clear Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Questions Only", type="secondary"):
                if st.checkbox("Confirm clear questions", key="confirm_clear_questions"):
                    with st.spinner("Clearing questions..."):
                        try:
                            with db.get_session() as session:
                                session.execute(text("TRUNCATE TABLE questions CASCADE"))
                            st.success("✅ Questions cleared")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
        
        with col2:
            if st.button("🗑️ Clear Analysis Data", type="secondary"):
                if st.checkbox("Confirm clear analysis", key="confirm_clear_analysis"):
                    with st.spinner("Clearing analysis data..."):
                        try:
                            with db.get_session() as session:
                                session.execute(text("TRUNCATE TABLE question_cluster_assignments CASCADE"))
                                session.execute(text("TRUNCATE TABLE question_embeddings CASCADE"))
                                session.execute(text("TRUNCATE TABLE topic_clusters CASCADE"))
                                session.execute(text("TRUNCATE TABLE analysis_runs CASCADE"))
                                session.execute(text("TRUNCATE TABLE analysis_cache CASCADE"))
                            st.success("✅ Analysis data cleared")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
        
        with col3:
            if st.button("🗑️ Clear ALL Data", type="secondary"):
                if st.checkbox("Confirm clear ALL", key="confirm_clear_all"):
                    with st.spinner("Clearing all data..."):
                        try:
                            if db.clear_all_data():
                                st.success("✅ All data cleared")
                            else:
                                st.error("❌ Failed to clear data")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
        
        st.markdown("---")
        
        # Database info
        st.subheader("Database Information")
        
        with db.get_session() as session:
            # Get database size
            result = session.execute(text("""
                SELECT 
                    pg_size_pretty(pg_database_size(current_database())) as db_size,
                    current_database() as db_name
            """))
            row = result.fetchone()
            
            if row:
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"📊 Database: {row[1]}")
                with col2:
                    st.info(f"💾 Size: {row[0]}")
        
    except Exception as e:
        st.error(f"❌ Error in maintenance tab: {e}")
        logger.error(f"Error in maintenance tab: {e}", exc_info=True)

if __name__ == "__main__":
    main()
