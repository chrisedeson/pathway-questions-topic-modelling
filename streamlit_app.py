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
    
    # Add dynamic theme detection and selectbox styling
    st.markdown("""
    <script>
    function fixSelectboxTheme() {
        // Wait for elements to load
        setTimeout(() => {
            // Detect if we're in dark mode by checking background color
            const app = window.parent.document.querySelector('.stApp');
            const computedStyle = getComputedStyle(app);
            const bgColor = computedStyle.backgroundColor;
            const isDark = bgColor.includes('14, 17, 23') || bgColor.includes('11, 13, 18');
            
            // Apply theme-specific styles to all selectboxes
            const selectboxes = window.parent.document.querySelectorAll('.stSelectbox');
            selectboxes.forEach(selectbox => {
                const selectElements = selectbox.querySelectorAll('div, span, *');
                selectElements.forEach(element => {
                    if (isDark) {
                        element.style.color = '#fafafa !important';
                        element.style.setProperty('color', '#fafafa', 'important');
                    } else {
                        element.style.color = '#262730 !important';
                        element.style.setProperty('color', '#262730', 'important');
                    }
                });
                
                // Fix select container backgrounds
                const selectContainers = selectbox.querySelectorAll('div[data-baseweb="select"] > div');
                selectContainers.forEach(container => {
                    if (isDark) {
                        container.style.backgroundColor = '#262730 !important';
                        container.style.borderColor = '#30363d !important';
                    } else {
                        container.style.backgroundColor = 'white !important';
                        container.style.borderColor = '#d1d5db !important';
                    }
                });
            });
        }, 100);
    }
    
    // Run the fix immediately and on theme changes
    fixSelectboxTheme();
    
    // Re-run when DOM changes (theme switching)
    const observer = new MutationObserver(fixSelectboxTheme);
    observer.observe(window.parent.document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['class', 'style']
    });
    
    // Also run on window resize/focus events
    window.parent.addEventListener('focus', fixSelectboxTheme);
    window.parent.addEventListener('resize', fixSelectboxTheme);
    </script>
    
    <style>
    /* Backup CSS for selectbox visibility */
    .stSelectbox div[data-baseweb="select"] * {
        color: inherit !important;
        opacity: 1 !important;
    }
    
    /* Ensure selectbox text contrasts are readable */
    @media (prefers-color-scheme: dark) {
        .stSelectbox * {
            color: #fafafa !important;
        }
        .stSelectbox > div > div,
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #262730 !important;
            border-color: #30363d !important;
        }
    }
    
    @media (prefers-color-scheme: light) {
        .stSelectbox * {
            color: #262730 !important;
        }
        .stSelectbox > div > div,
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: white !important;
            border-color: #d1d5db !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)


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
