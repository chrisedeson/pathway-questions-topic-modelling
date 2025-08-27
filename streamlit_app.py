"""
BYU Pathway Questions Analysis - Streamlined Main Application

This is the main Streamlit application that uses modular components
to provide topic modeling and analysis of BYU Pathway student questions.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent.absolute()
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))

# Now import modular components
try:
    from config import CUSTOM_CSS, PAGE_TITLE, PAGE_ICON, LAYOUT
    from components import (
        display_header, check_api_key, display_analysis_summary, 
        display_question_explorer, upload_and_analyze_tab, 
        view_previous_results_tab, display_app_footer
    )
    from visualizations import display_visualization_tabs
    from utils import create_session_state_defaults
    from enhanced_metrics import create_enhanced_metrics_tab
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main():
    """Main application function"""
    # Configure page
    configure_page()
    
    # Display header
    display_header()
    
    # Check API key
    if not check_api_key():
        return
    
    # Initialize session state
    create_session_state_defaults()
    
    # Create tabs based on analysis state
    if st.session_state.analysis_complete and st.session_state.current_results is not None:
        # After analysis - show results-focused tabs with enhanced metrics tab first
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Enhanced Metrics",
            "📊 Analysis Results", 
            "🔍 Explore Questions", 
            "📈 Visualizations", 
            "📤 Upload New File"
        ])
        
        with tab1:
            # Enhanced metrics tab
            create_enhanced_metrics_tab(
                st.session_state.current_results,
                st.session_state.get('current_topic_model'),
                st.session_state.get('clustering_metrics')
            )
        
        with tab2:
            display_analysis_summary(
                st.session_state.current_results,
                st.session_state.get('clustering_metrics')
            )
        
        with tab3:
            display_question_explorer(st.session_state.current_results)
        
        with tab4:
            display_visualization_tabs(
                st.session_state.current_results,
                st.session_state.get('current_topic_model'),
                st.session_state.get('current_embeddings')
            )
        
        with tab5:
            upload_and_analyze_tab()
            
    else:
        # Before analysis - focus on upload and previous results  
        tab1, tab2 = st.tabs(["📤 Upload & Analyze", "📊 View Previous Results"])
        
        with tab1:
            upload_and_analyze_tab()
        
        with tab2:
            view_previous_results_tab()
    
    # Display footer
    display_app_footer()


if __name__ == "__main__":
    main()
