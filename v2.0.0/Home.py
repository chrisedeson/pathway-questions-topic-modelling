"""
BYU Pathway Topic Analysis Dashboard v2.0.0
Main Streamlit Application

A professional, scalable dashboard for analyzing student questions and topics.
"""

import streamlit as st
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from config import PAGE_CONFIG, CUSTOM_CSS
from utils.data_loader import load_data_from_s3, merge_data_for_dashboard, calculate_kpis, get_latest_file_info


def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(**PAGE_CONFIG)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main():
    """Main application entry point"""
    configure_page()
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("BYU Pathway Missionary Question Analysis Dashboard")
        st.markdown("*Insights into missionary questions and topic discovery*")
    
    with col2:
        st.image(
            "https://byu-pathway.brightspotcdn.com/42/2e/4d4c7b10498c84233ae51179437c/byu-pw-icon-gold-rgb-1-1.svg",
            width=100
        )
    
    st.markdown("---")
    
    # Load data
    with st.spinner("🔄 Loading data from AWS S3..."):
        data = load_data_from_s3()
    
    if not data:
        st.error("❌ **No data available.** Please ensure the notebook has uploaded files to S3.")
        st.info("""
        **💡 Tip:** Run the Jupyter notebook to process questions and upload results to S3.
        The dashboard will automatically load the most recent data.
        """)
        st.stop()
    
    # Merge data for dashboard
    merged_df = merge_data_for_dashboard(data)
    
    if merged_df.empty:
        st.warning("⚠️ No question data available in the loaded files.")
        st.stop()
    
    # Calculate KPIs
    kpis = calculate_kpis(merged_df, data)
    
    # Store in session state for use in other pages
    if 'merged_df' not in st.session_state:
        st.session_state['merged_df'] = merged_df
    if 'raw_data' not in st.session_state:
        st.session_state['raw_data'] = data
    if 'kpis' not in st.session_state:
        st.session_state['kpis'] = kpis
    
    # Success message
    file_info = get_latest_file_info()
    if file_info and 'timestamp' in file_info:
        st.success(f"✅ **Data loaded successfully!** Processing {kpis['total_questions']:,} questions from S3.")
        st.caption(f"📅 Data timestamp: {file_info['timestamp']}")
    else:
        st.success(f"✅ **Data loaded successfully!** Processing {kpis['total_questions']:,} questions from S3.")
    
    # Quick overview
    st.markdown("### 📋 Quick Overview")
    
    from utils.visualizations import create_kpi_cards
    create_kpi_cards(kpis)
    
    st.markdown("---")
    
    # Navigation info
    st.info("""
    ### 🧭 Navigation
    
    Use the sidebar to navigate between different sections:
    
    - **📊 Dashboard** (Home): Overview and key metrics
    - **📋 Questions Table**: Interactive table with filters and search
    - **📈 Trends & Analytics**: Detailed visualizations and insights
    - **🆕 New Topics**: Explore newly discovered topics
    - **📥 Export Data**: Download processed data
    
    💡 **Tip:** All filters and sorting happen instantly without page refresh!
    """)
    
    # Sidebar - Refresh button at the bottom
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data", help="Clear cache and reload data from S3", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>BYU Pathway Worldwide</strong> | Topic Analysis Dashboard v2.0.0</p>
        <p>Powered by AWS S3, OpenAI, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
