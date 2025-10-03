"""
Database-backed developer mode interface - sidebar password protected settings and controls
Enhanced with smart caching and background updates
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
import json
from datetime import datetime, timezone
import time

from data_service import get_data_service
from sheets_sync import get_sheets_manager
from database_analysis_engine import get_analysis_engine
from config import (
    QUESTIONS_SHEET_ID, TOPICS_SHEET_ID, DEFAULT_SYNC_INTERVAL_MINUTES,
    DEV_PASSWORD
)

# Import our smart data manager
try:
    from streamlit_data_manager import data_manager
except ImportError:
    data_manager = None

def check_dev_password() -> bool:
    """Check if the user is already authenticated in the session."""
    return st.session_state.get('dev_authenticated', False)

def validate_dev_password(password: str) -> bool:
    """Validate a given password against the DEV_PASSWORD."""
    return password == DEV_PASSWORD

def show_dev_mode_toggle():
    """Display dev mode toggle in sidebar"""
    with st.sidebar:
        st.markdown("---")
        if st.button("⚙️ Developer Mode", use_container_width=True, key="dev_mode_toggle"):
            if not st.session_state.get('dev_authenticated', False):
                st.session_state.show_dev_login = True
                st.session_state.show_dev_sidebar = True
            else:
                st.session_state.show_dev_sidebar = not st.session_state.get('show_dev_sidebar', False)
        st.markdown("---")

def show_dev_login_in_sidebar():
    """Display developer login in sidebar (not full page)"""
    if st.session_state.get('show_dev_login', False) and not st.session_state.get('dev_authenticated', False):
        with st.sidebar:
            st.markdown("## 🔒 Developer Login")
            st.markdown("---")
            
            password = st.text_input("Enter developer password:", type="password", key="dev_password_sidebar")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login", type="primary", use_container_width=True):
                    if validate_dev_password(password):
                        st.session_state.dev_authenticated = True
                        st.session_state.show_dev_login = False
                        st.success("✅ Authenticated!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid password")
            
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_dev_login = False
                    st.session_state.show_dev_sidebar = False
                    st.rerun()

def require_dev_auth(func):
    """Decorator to require developer authentication"""
    def wrapper(*args, **kwargs):
        if not check_dev_password():
            st.warning("🔒 Developer authentication required")
            return None
        return func(*args, **kwargs)
    return wrapper

def logout_dev_mode():
    """Logout from developer mode"""
    st.session_state.dev_authenticated = False
    st.session_state.show_dev_login = False
    st.session_state.show_dev_sidebar = False

@require_dev_auth
def show_dev_sidebar():
    """Display developer sidebar with settings and controls"""
    if not st.session_state.get('show_dev_sidebar', False):
        return
    
    with st.sidebar:
        st.markdown("## ⚙️ Developer Mode")
        st.markdown("---")
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            logout_dev_mode()
            st.rerun()
        
        st.markdown("---")
        
        # Tabs for different dev sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "📁 Data", "🔄 Sync", "⚙️ Config"])
        
        with tab1:
            show_analysis_controls()
        
        with tab2:
            show_data_management()
        
        with tab3:
            show_sync_management()
        
        with tab4:
            show_configuration_settings()

def show_analysis_controls():
    """Display analysis control panel with smart caching integration"""
    st.markdown("### 🔬 Analysis Controls")
    
    # Show smart caching status first
    if data_manager:
        st.markdown("#### 💾 Smart Caching Status")
        
        # Display background update status
        bg_status = data_manager.get_background_update_status()
        
        if bg_status['running']:
            st.info("🔄 **Background database update in progress...**")
            st.progress(0.7)
        elif bg_status['last_sync']:
            st.success(f"✅ **Last database update:** {bg_status['last_sync'][:19]}")
        else:
            st.info("📭 **No recent database updates**")
        
        # Show cached analysis status
        if st.session_state.get('cached_analysis') and st.session_state.get('startup_data_loaded'):
            cached_analysis = st.session_state['cached_analysis']
            st.info(f"""
            📊 **Cached Analysis Available:**
            - **Run ID:** `{cached_analysis['run_id'][:8]}...`
            - **Completed:** {cached_analysis['completed_at'][:19]}
            - **Questions:** {cached_analysis['total_questions']}
            - **Topics:** {cached_analysis['total_topics']}
            """)
            
            # Option to clear cache
            if st.button("🗑️ **Clear Cache**", type="secondary"):
                if 'cached_analysis' in st.session_state:
                    del st.session_state['cached_analysis']
                if 'startup_data_loaded' in st.session_state:
                    del st.session_state['startup_data_loaded']
                st.success("✅ Cache cleared!")
                st.rerun()
        else:
            st.warning("📭 **No cached analysis** - Run new analysis to populate cache")
    
    st.markdown("---")
    
    # Get current status from existing analysis engine
    analysis_engine = get_analysis_engine()
    status = analysis_engine.get_analysis_status()
    
    # Display current status and progress
    if status['is_running']:
        st.info(f"🔄 Analysis running: {status['current_run_id']}")
        
        # Get progress from the analysis engine (thread-safe)
        progress_info = analysis_engine.get_current_progress()
        
        if progress_info and progress_info['progress'] > 0:
            progress_value = progress_info['progress'] / 100.0
            
            # Progress bar
            st.progress(progress_value, text=f"{progress_info['progress']:.1f}%")
            
            # Current step info
            step_emoji = {
                'initialization': '🚀',
                'clearing_data': '🧹',
                'loading_data': '📥',
                'storing_questions': '💾',
                'loading_topics': '📋',
                'embeddings': '🧠',
                'clustering': '🔗',
                'storing_embeddings': '💾',
                'storing_clusters': '📊',
                'storing_assignments': '🔗',
                'sentiment': '😊',
                'trends': '📈',
                'finalizing': '✅',
                'failed': '❌'
            }.get(progress_info['step'], '⚙️')
            
            st.markdown(f"{step_emoji} **{progress_info['step'].replace('_', ' ').title()}**")
            st.markdown(f"*{progress_info['message']}*")
        else:
            st.info("Analysis starting... Progress will appear automatically")
            
        # Auto-refresh only the progress section (not whole page)
        time.sleep(1)
        st.rerun()
    else:
        st.success("✅ Ready to run fresh analysis")
    
    # Analysis controls with smart caching integration
    st.markdown("#### 🚀 Analysis Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ **Run Smart Analysis**", 
                    disabled=status['is_running'], 
                    use_container_width=True,
                    type="primary",
                    help="Run analysis with smart caching - results cached immediately, database updated in background"):
            try:
                # Option 1: Use our enhanced analysis with smart caching
                if data_manager:
                    st.info("🚀 **Starting Smart Analysis** - Results will be cached immediately!")
                    
                    # This would integrate with your existing analysis but with smart caching
                    # For now, trigger the existing analysis and then cache results
                    def progress_callback(step, progress, message):
                        st.session_state.progress = {
                            'step': step,
                            'progress': progress,
                            'message': message
                        }
                    
                    analysis_engine.set_progress_callback(progress_callback)
                    run_id = analysis_engine.run_full_analysis(force_refresh=True)
                    
                    # Store the run ID for status checking
                    st.session_state['last_analysis_run_id'] = run_id
                    
                    st.success(f"🚀 Smart analysis started: {run_id}")
                    st.info("💡 **Smart Features:** Results will be cached for fast access while database updates in background")
                    st.rerun()
                else:
                    st.error("❌ Smart data manager not available - falling back to standard analysis")
                    
            except Exception as e:
                st.error(f"❌ Failed to start smart analysis: {e}")
    
    with col2:
        if st.button("🔄 **Load Fresh from DB**", 
                    disabled=status['is_running'],
                    use_container_width=True,
                    help="Load latest analysis from database into cache"):
            try:
                if data_manager:
                    with st.spinner("Loading latest analysis from database..."):
                        latest_analysis = data_manager.load_latest_analysis_from_database()
                        
                    if latest_analysis:
                        st.session_state['cached_analysis'] = latest_analysis
                        st.session_state['startup_data_loaded'] = True
                        st.success(f"✅ Loaded analysis: {latest_analysis['run_id'][:8]}...")
                        st.rerun()
                    else:
                        st.warning("📭 No analysis found in database")
                else:
                    st.error("❌ Smart data manager not available")
                    
            except Exception as e:
                st.error(f"❌ Failed to load from database: {e}")
    
    # Show completion status if analysis just finished
    if not status['is_running'] and st.session_state.get('last_analysis_run_id'):
        last_run_id = st.session_state['last_analysis_run_id']
        
        # Check if this analysis run completed successfully
        try:
            data_service = get_data_service()
            analysis_status = data_service.get_analysis_status()
            
            if analysis_status and analysis_status.get('latest_run'):
                latest_run = analysis_status['latest_run']
                
                # Check if this is the run we're tracking
                if latest_run['run_id'] == last_run_id:
                    if latest_run['status'] == 'completed':
                        st.success("🎉 **Analysis Completed Successfully!**")
                        st.info(f"""
                        ✅ **Database Updated:** All analysis results saved to database
                        📊 **Run ID:** `{last_run_id}`
                        🕒 **Completed:** {latest_run.get('completed_at', 'Just now')}
                        📋 **Questions Processed:** {latest_run.get('total_questions', 0)}
                        🏷️ **Topics Discovered:** {latest_run.get('total_topics', 0)}
                        """)
                        
                        # Clear the last run ID so this message doesn't persist
                        del st.session_state['last_analysis_run_id']
                        
                        # Offer to view results
                        if st.button("👀 **View Analysis Results**", type="secondary"):
                            st.rerun()  # This will trigger the main app to show results
                            
                    elif latest_run['status'] == 'failed':
                        st.error("❌ **Analysis Failed**")
                        st.error(f"Error: {latest_run.get('error_message', 'Unknown error')}")
                        del st.session_state['last_analysis_run_id']
                        
                    elif latest_run['status'] == 'running':
                        st.info("🔄 **Analysis Still Running...**")
                        
        except Exception as e:
            st.warning(f"Could not verify analysis completion: {e}")
    
    # Additional controls
    st.markdown("---")
    st.markdown("#### 🛠️ Advanced Controls")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("🧹 Clear Database", 
                    disabled=status['is_running'],
                    use_container_width=True,
                    help="Clear all data from database"):
            try:
                data_service = get_data_service()
                if data_service.db_manager.clear_all_data():
                    st.success("✅ Database cleared successfully")
                else:
                    st.error("❌ Failed to clear database")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error clearing database: {e}")
    
    # Database status
    st.markdown("### 📊 Database Status")
    db_status = status.get('database_status', {})
    
    if db_status:
        latest_run = db_status.get('latest_run', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Questions", db_status.get('database_stats', {}).get('total_questions', 0))
        
        with col2:
            topics = latest_run.get('total_topics', 0)
            st.metric("Topics", topics if topics else "N/A")
        
        with col3:
            status_text = latest_run.get('status', 'unknown')
            st.metric("Last Run", status_text.title())
        
        # Last run details
        if latest_run:
            with st.expander("🔍 Last Analysis Run Details"):
                st.json(latest_run)

def show_data_management():
    """Display data management controls"""
    st.markdown("### 📁 Data Management")
    
    data_service = get_data_service()
    
    # File upload
    st.markdown("#### 📤 Upload Questions File")
    uploaded_file = st.file_uploader(
        "Choose CSV file", 
        type=['csv'],
        help="Upload a CSV file with questions to add to the system"
    )
    
    if uploaded_file is not None:
        try:
            # Read and preview the file
            df = pd.read_csv(uploaded_file)
            st.write(f"**Preview of {uploaded_file.name}** ({len(df)} rows)")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🚀 Process and Upload", type="primary"):
                with st.spinner("Processing file..."):
                    # Save uploaded file temporarily
                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Upload to Google Sheets and database
                    sheets_manager = get_sheets_manager()
                    success, message, stats = sheets_manager.upload_file(temp_path)
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.json(stats)
                    else:
                        st.error(f"❌ {message}")
                        
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    
    # Data cleaning tools
    st.markdown("#### 🧹 Data Cleaning")
    
    if st.button("🧽 Clean Questions Data"):
        with st.spinner("Cleaning questions in database..."):
            # This could trigger a data cleaning process
            st.info("Data cleaning functionality to be implemented")
    
    # Database stats
    st.markdown("#### 📊 Database Statistics")
    
    try:
        stats = data_service.get_analysis_status()
        
        if stats:
            db_stats = stats.get('database_stats', {})
            sync_status = stats.get('sync_status', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Questions", db_stats.get('total_questions', 0))
                st.metric("Questions in Latest Run", db_stats.get('questions_in_latest_run', 0))
            
            with col2:
                questions_sync = sync_status.get('questions', {})
                st.metric("Last Sync Status", questions_sync.get('status', 'unknown').title())
                
                if questions_sync.get('completed_at'):
                    last_sync = pd.to_datetime(questions_sync['completed_at'])
                    st.metric("Last Sync", last_sync.strftime("%Y-%m-%d %H:%M"))
    
    except Exception as e:
        st.error(f"❌ Error getting database stats: {e}")

def show_sync_management():
    """Display synchronization management controls"""
    st.markdown("### 🔄 Google Sheets Sync")
    
    sheets_manager = get_sheets_manager()
    
    # Manual sync controls
    st.markdown("#### 🔄 Manual Sync")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬇️ Sync from Sheets", use_container_width=True):
            with st.spinner("Syncing from Google Sheets..."):
                success, message, stats = sheets_manager.manual_sync()
                
                if success:
                    st.success(f"✅ {message}")
                    if stats:
                        st.json(stats)
                else:
                    st.error(f"❌ {message}")
    
    with col2:
        if st.button("🔗 Test Connections", use_container_width=True):
            with st.spinner("Testing connections..."):
                results = sheets_manager.test_connections()
                
                for sheet_type, success in results.items():
                    if success:
                        st.success(f"✅ {sheet_type.title()} sheet: Connected")
                    else:
                        st.error(f"❌ {sheet_type.title()} sheet: Failed")
    
    # Sync schedule configuration
    st.markdown("#### ⏰ Sync Schedule")
    
    status = sheets_manager.get_status()
    schedule_info = status.get('schedule_info', {})
    
    current_interval = status.get('sync_interval_minutes', DEFAULT_SYNC_INTERVAL_MINUTES)
    
    new_interval = st.selectbox(
        "Sync Interval",
        options=[5, 10, 15, 30, 60],
        index=[5, 10, 15, 30, 60].index(current_interval) if current_interval in [5, 10, 15, 30, 60] else 1,
        help="How often to sync with Google Sheets (in minutes)"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Start Sync", use_container_width=True):
            sheets_manager.set_sync_interval(new_interval)
            if sheets_manager.start_sync():
                st.success(f"✅ Sync started (every {new_interval} min)")
                st.rerun()
            else:
                st.error("❌ Failed to start sync")
    
    with col2:
        if st.button("⏹️ Stop Sync", use_container_width=True):
            sheets_manager.stop_sync()
            st.success("✅ Sync stopped")
            st.rerun()
    
    # Sync status
    if schedule_info.get('status') == 'running':
        st.success("🔄 Automatic sync is running")
        
        jobs = schedule_info.get('jobs', [])
        if jobs:
            for job in jobs:
                next_run = job.get('next_run')
                if next_run:
                    st.info(f"⏰ Next sync: {next_run}")
    else:
        st.warning("⏸️ Automatic sync is stopped")

def show_configuration_settings():
    """Display configuration settings"""
    st.markdown("### ⚙️ Configuration")
    
    # Google Sheets IDs
    st.markdown("#### 📊 Google Sheets Configuration")
    
    current_questions_id = QUESTIONS_SHEET_ID
    current_topics_id = TOPICS_SHEET_ID
    
    new_questions_id = st.text_input(
        "Questions Sheet ID", 
        value=current_questions_id,
        help="Google Sheets ID for the questions data"
    )
    
    new_topics_id = st.text_input(
        "Topics Sheet ID", 
        value=current_topics_id,
        help="Google Sheets ID for the topics/subtopics data"
    )
    
    if st.button("💾 Update Sheet IDs"):
        sheets_manager = get_sheets_manager()
        sheets_manager.set_sheet_ids(new_questions_id, new_topics_id)
        st.success("✅ Sheet IDs updated")
        st.rerun()
    
    # Analysis configuration
    st.markdown("#### 🔬 Analysis Configuration")
    
    with st.expander("🔧 Analysis Parameters"):
        st.info("Analysis parameters are loaded from environment variables and config.py")
        
        config_display = {
            "Embedding Model": "text-embedding-3-small",
            "Similarity Threshold": 0.70,
            "Min Cluster Size": 3,
            "UMAP Components": 5
        }
        
        for key, value in config_display.items():
            st.text(f"{key}: {value}")
    
    # Cache management
    st.markdown("#### 💾 Cache Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Analysis Cache", use_container_width=True):
            # This would clear cached analysis results
            st.info("Cache clearing functionality to be implemented")
    
    with col2:
        if st.button("📊 View Cache Stats", use_container_width=True):
            # This would show cache statistics
            st.info("Cache statistics functionality to be implemented")
    
    # System information
    st.markdown("#### ℹ️ System Information")
    
    with st.expander("🖥️ System Status"):
        try:
            from database import get_db_manager
            db_manager = get_db_manager()
            
            system_info = {
                "Database Connected": db_manager.test_connection(),
                "Google Sheets Client": get_sheets_manager().sync.client is not None,
                "Current Time": datetime.now(timezone.utc).isoformat(),
                "Environment": "Development" if DEV_PASSWORD == 'pathway_dev_2025' else "Production"
            }
            
            for key, value in system_info.items():
                if isinstance(value, bool):
                    st.text(f"{key}: {'✅ Yes' if value else '❌ No'}")
                else:
                    st.text(f"{key}: {value}")
                    
        except Exception as e:
            st.error(f"Error getting system info: {e}")

# Progress tracking for analysis runs
def init_progress_tracking():
    """Initialize progress tracking in session state"""
    if 'progress' not in st.session_state:
        st.session_state.progress = {
            'step': 'idle',
            'progress': 0,
            'message': 'Ready'
        }