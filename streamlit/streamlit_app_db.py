"""
BYU Pathway Student Questions Dashboard - Database-Backed Professional Edition
Instant loading with pre-computed database results and developer mode for configuration
"""

import streamlit as st
import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Add src directory to Python path
current_dir = Path(__file__).parent.absolute()
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))

# Import configuration and initialize database
try:
    from config import PAGE_TITLE, PAGE_ICON, LAYOUT, CUSTOM_CSS
    from database import init_database
    from sheets_sync import get_sheets_manager
    from config import QUESTIONS_SHEET_ID, TOPICS_SHEET_ID, DEFAULT_SYNC_INTERVAL_MINUTES
    from dashboard_components_db import display_full_dashboard
    from dev_mode_db import (
        show_dev_mode_toggle, show_dev_login_in_sidebar, 
        show_dev_sidebar, init_progress_tracking
    )
    
    # Import our new smart data manager
    from streamlit_data_manager import data_manager
    
    # Initialize database on startup
    init_database()
    
except ImportError as e:
    st.error(f"❌ Failed to import required modules: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to initialize database: {e}")
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Streamlit page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Apply custom CSS (safe fallback if empty)
if CUSTOM_CSS and CUSTOM_CSS.strip():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def initialize_app():
    """Initialize the application and configure services with smart caching"""
    try:
        # Initialize progress tracking for dev mode
        init_progress_tracking()
        
        # Initialize smart data manager for caching and background updates
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = True
            
            # Load previous analysis from database on startup (smart caching)
            try:
                with st.spinner("🔄 Loading previous analysis from database..."):
                    latest_analysis = data_manager.load_latest_analysis_from_database()
                    
                if latest_analysis:
                    # Cache in Streamlit for fast access
                    st.session_state['cached_analysis'] = latest_analysis
                    st.session_state['startup_data_loaded'] = True
                    st.success(f"✅ Loaded analysis from {latest_analysis['completed_at'][:19]}")
                else:
                    st.session_state['startup_data_loaded'] = False
                    st.info("ℹ️ No previous analysis found. Ready for new analysis.")
                    
            except Exception as e:
                st.warning(f"⚠️ Could not load previous analysis: {str(e)}")
                st.session_state['startup_data_loaded'] = False
        
        # Configure Google Sheets manager with default IDs
        sheets_manager = get_sheets_manager()
        sheets_manager.set_sheet_ids(QUESTIONS_SHEET_ID, TOPICS_SHEET_ID)
        
        # Start background sync if enabled
        if not sheets_manager.sync.scheduler or not sheets_manager.sync.scheduler.running:
            try:
                sheets_manager.start_sync()
                logging.info("Background sync started")
            except Exception as e:
                logging.warning(f"Failed to start background sync: {e}")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to initialize app: {e}")
        return False

def display_system_status():
    """Display system status in sidebar with smart caching info"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🖥️ System Status")
        
        try:
            from database import get_db_manager
            from sheets_sync import get_sheets_manager
            
            # Database status
            db_manager = get_db_manager()
            db_connected = db_manager.test_connection()
            
            if db_connected:
                st.success("🟢 Database Connected")
            else:
                st.error("🔴 Database Disconnected")
            
            # Smart caching status
            if st.session_state.get('startup_data_loaded', False):
                st.success("💾 Previous Analysis Cached")
            else:
                st.info("📭 No Cached Analysis")
                st.caption("Tip: Run analysis from Developer Mode to generate fresh results.")
            
            # Background update status
            data_manager.display_background_update_status()
            
            # Sheets sync status
            sheets_manager = get_sheets_manager()
            sync_status = sheets_manager.get_status()
            schedule_info = sync_status.get('schedule_info', {})
            
            if schedule_info.get('status') == 'running':
                st.success("🔄 Auto-sync Active")
            else:
                st.warning("⏸️ Auto-sync Stopped")
            
            # Last sync info
            from data_service import get_data_service
            data_service = get_data_service()
            status = data_service.get_analysis_status()
            sync_status_details = status.get('sync_status', {})
            
            questions_sync = sync_status_details.get('questions', {})
            if questions_sync.get('completed_at'):
                last_sync = pd.to_datetime(questions_sync['completed_at'])
                st.info(f"📊 Last sync: {last_sync.strftime('%H:%M')}")
        
        except Exception as e:
            st.error("❌ Status check failed")
            # Fallback to basic database summary
            try:
                db_summary = data_manager.get_database_summary()
                if db_summary['status'] == 'connected':
                    st.success("🟢 Database Connected")
                    st.metric("Questions", db_summary.get('questions', 0))
                    st.metric("Topics", db_summary.get('topics', 0))
                else:
                    st.error("🔴 Database Error")
            except Exception:
                st.error("❌ Could not retrieve database summary")

def main():
    """Main application entry point"""
    
    # Initialize the application
    if not initialize_app():
        st.stop()
    
    # Sidebar components (always visible)
    with st.sidebar:
        st.markdown("# 🎓 BYU Pathway")
        st.markdown("## Questions Analytics")
        st.markdown("---")
    
    # Developer mode components (in sidebar)
    show_dev_mode_toggle()
    show_dev_login_in_sidebar()
    show_dev_sidebar()
    
    # System status (in sidebar)
    display_system_status()
    
    # Main dashboard content (instant loading)
    try:
        display_full_dashboard()
        
    except Exception as e:
        st.error(f"❌ Dashboard Error: {e}")
        st.markdown("""
        ### 🔧 Troubleshooting
        
        If you're seeing this error:
        1. **Check if analysis has been run**: Use Developer Mode to run initial analysis
        2. **Verify database connection**: Check system status in sidebar
        3. **Sync data**: Use Developer Mode to sync from Google Sheets
        4. **Contact administrator**: If problems persist
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #666; font-size: 0.9rem;">
        📊 BYU Pathway Questions Analytics Dashboard | 
        🔄 Data updates automatically | 
        ⚙️ Developer mode available in sidebar
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
