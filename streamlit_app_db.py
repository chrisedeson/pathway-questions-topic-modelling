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

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def initialize_app():
    """Initialize the application and configure services"""
    try:
        # Initialize progress tracking for dev mode
        init_progress_tracking()
        
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
    """Display system status in sidebar"""
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
            st.error(f"❌ Status check failed")

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