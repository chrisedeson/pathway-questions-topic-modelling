"""
Questions Table Page - Interactive data table with filters
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CLASSIFICATION_OPTIONS
from utils.data_loader import filter_dataframe, export_to_csv


def main():
    st.title("📋 Questions Table")
    st.markdown("*Interactive table with advanced filtering*")
    st.markdown("---")
    
    # Check if data is loaded
    if 'merged_df' not in st.session_state:
        st.error("❌ No data loaded. Please return to the home page.")
        st.stop()
    
    df = st.session_state['merged_df'].copy()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Classification filter
    classification = st.sidebar.selectbox(
        "Classification",
        CLASSIFICATION_OPTIONS,
        help="Filter by question classification"
    )
    
    # Date range filter
    st.sidebar.subheader("📅 Date Range")
    if 'timestamp' in df.columns:
        min_date = df['timestamp'].min().date() if not df['timestamp'].isna().all() else datetime.now().date()
        max_date = df['timestamp'].max().date() if not df['timestamp'].isna().all() else datetime.now().date()
        
        date_range = st.sidebar.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Filter questions by date range"
        )
        
        if len(date_range) == 2:
            date_filter = date_range
        else:
            date_filter = None
    else:
        date_filter = None
        st.sidebar.info("No timestamp data available")
    
    # Country filter
    if 'country' in df.columns:
        countries = sorted(df['country'].dropna().unique().tolist())
        selected_countries = st.sidebar.multiselect(
            "🌍 Countries",
            countries,
            help="Filter by country (leave empty for all)"
        )
        country_filter = selected_countries if selected_countries else None
    else:
        country_filter = None
    
    # Similarity filter
    if 'similarity_score' in df.columns:
        min_similarity = st.sidebar.slider(
            "📊 Minimum Similarity Score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Filter by minimum similarity score (for existing topics)"
        )
    else:
        min_similarity = None
    
    # Search box
    st.sidebar.subheader("🔎 Search")
    search_query = st.sidebar.text_input(
        "Search in questions",
        placeholder="Enter keywords...",
        help="Search for specific text in questions"
    )
    
    # Clear filters button
    if st.sidebar.button("🔄 Clear All Filters"):
        st.rerun()
    
    # Apply filters
    filtered_df = filter_dataframe(
        df,
        classification=classification,
        date_range=date_filter,
        countries=country_filter,
        search_query=search_query if search_query else None,
        min_similarity=min_similarity
    )
    
    # Main content
    st.markdown(f"### 📊 Showing {len(filtered_df):,} of {len(df):,} questions")
    
    # Export options
    st.markdown("####  Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = export_to_csv(filtered_df)
        st.download_button(
            label="📄 Download Filtered Data (CSV)",
            data=csv_data,
            file_name=f"pathway_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download the currently filtered data"
        )
    
    with col2:
        csv_all = export_to_csv(df)
        st.download_button(
            label="📄 Download All Data (CSV)",
            data=csv_all,
            file_name=f"pathway_questions_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download all questions (ignoring filters)"
        )
    
    st.markdown("---")
    
    # Display table with Streamlit's native interactive dataframe
    if not filtered_df.empty:
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=600,
            hide_index=True
        )
        
        # Summary statistics
        with st.expander("📊 Summary Statistics"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rows", f"{len(filtered_df):,}")
            
            with col2:
                if 'country' in filtered_df.columns:
                    st.metric("Unique Countries", filtered_df['country'].nunique())
            
            with col3:
                if 'similarity_score' in filtered_df.columns:
                    avg_sim = filtered_df['similarity_score'].mean()
                    st.metric("Avg Similarity", f"{avg_sim:.3f}")
            
            with col4:
                if 'classification' in filtered_df.columns:
                    new_topic_pct = (filtered_df['classification'] == 'new').sum() / len(filtered_df) * 100
                    st.metric("New Topics %", f"{new_topic_pct:.1f}%")
    
    else:
        st.info("ℹ️ No data to display with current filters. Try adjusting your filters.")
    
    # Tips
    st.markdown("---")
    st.info("""
    ### 💡 Tips for Using the Table
    
    - **Filter** questions by classification, date, country, or similarity score
    - **Search** for specific keywords in questions
    - **Sort** columns by clicking on the column headers
    - **Resize** columns by dragging the column borders
    - **Export** your filtered results to CSV for external analysis
    - All operations happen **instantly** without page refresh!
    """)


if __name__ == "__main__":
    main()
