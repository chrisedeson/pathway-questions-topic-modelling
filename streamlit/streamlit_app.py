"""
BYU Pathway Hybrid Topic Analysis - Enhanced Streamlit Application

This application combines similarity-based classification with clustering-based topic discovery
to help Elder Edwards and his team analyze student questions and manage topic taxonomies.
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
    from config import (
        CUSTOM_CSS, PAGE_TITLE, PAGE_ICON, LAYOUT,
        EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, CHAT_MODEL,
        UMAP_N_COMPONENTS, MIN_CLUSTER_SIZE, SIMILARITY_THRESHOLD,
        CACHE_EMBEDDINGS
    )
    from enhanced_components import (
        display_header, check_api_key, create_hybrid_processing_tab,
        display_hybrid_results, display_app_footer
    )
    from utils import create_session_state_defaults
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure all required dependencies are installed. Run: pip install -r requirements.txt")
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
    
    # Enhanced styling for the hybrid interface
    st.markdown("""
    <style>
    /* Enhanced styling for hybrid interface */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .topic-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .similarity-threshold {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Improved button styling */
    .stButton > button {
        border-radius: 20px;
        border: none;
        font-weight: 500;
    }
    
    /* Enhanced expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    </style>
    """, unsafe_allow_html=True)


def create_sidebar():
    """Create enhanced sidebar with configuration options"""
    with st.sidebar:
        st.image("https://byu-pathway.brightspotcdn.com/42/2e/4d4c7b10498c84233ae51179437c/byu-pw-icon-gold-rgb-1-1.svg", width=100)
        
        st.markdown("### BYU Pathway")
        st.markdown("### Hybrid Topic Analysis")
        
        st.markdown("---")
        
        # Quick stats if data is loaded
        if 'hybrid_results' in st.session_state:
            results = st.session_state['hybrid_results']
            similar_df = results.get('similar_questions_df', [])
            clustered_df = results.get('clustered_questions_df')
            
            st.markdown("### 📊 Current Analysis")
            st.metric("Total Questions", len(results.get('eval_questions_df', [])))
            st.metric("Matched Existing", len(similar_df))
            
            # Show both new topics count and questions in new topics for clarity
            if clustered_df is not None and len(clustered_df) > 0:
                n_new_topics = len(clustered_df['cluster_id'].unique())
                n_questions_in_new_topics = len(clustered_df)
                st.metric("New Topics Found", n_new_topics)
                st.metric("Questions in New Topics", n_questions_in_new_topics)
            else:
                st.metric("New Topics Found", 0)
                st.metric("Questions in New Topics", 0)
        
        st.markdown("---")
        
        # Help section
        with st.expander("ℹ️ **How to Use**", expanded=False):
            st.markdown("""
            **Step 1**: Choose data source (file or Google Sheets)
            
            **Step 2**: Configure similarity threshold
            
            **Step 3**: Run hybrid analysis
            
            **Step 4**: Review and approve new topics
            
            **Step 5**: Manage topic taxonomy
            """)
        
        # Configuration section
        with st.expander("⚙️ **Configuration**", expanded=False):
            # Show current configuration
            from config import EMBEDDING_MODEL, CHAT_MODEL, SIMILARITY_THRESHOLD
            
            st.write("**Current Settings:**")
            st.write(f"• Embedding Model: {EMBEDDING_MODEL}")
            st.write(f"• Chat Model: {CHAT_MODEL}")
            st.write(f"• Default Threshold: {SIMILARITY_THRESHOLD}")
        
        st.markdown("---")
        st.markdown("*Powered by OpenAI GPT & Embeddings*")


def main():
    """Main application function"""
    # Configure page
    configure_page()
    
    # Create sidebar
    create_sidebar()
    
    # Display header
    display_header()
    
    # Check API key
    if not check_api_key():
        return
    
    # Initialize session state
    create_session_state_defaults()
    
    # Main application tabs
    if 'hybrid_results' in st.session_state and st.session_state['hybrid_results']:
        # Post-analysis: Show results and management tools
        tab1, tab2 = st.tabs([
            "📊 **Analysis Results**",
            "🚀 **New Analysis**"
        ])
        
        with tab1:
            display_hybrid_results(st.session_state['hybrid_results'])
        
        with tab2:
            create_hybrid_processing_tab()
    
    else:
        # Pre-analysis: Focus on data input and processing
        tab1, tab2 = st.tabs([
            "🚀 **Hybrid Analysis**",
            "📖 **About**"
        ])
        
        with tab1:
            create_hybrid_processing_tab()
        
        with tab2:
            display_about_tab()
    
    # Display footer
    display_app_footer()


def display_about_tab():
    """Display information about the hybrid approach"""
    st.header("📖 About Hybrid Topic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("What is Hybrid Processing?")
        st.markdown("""
        **Hybrid Topic Analysis** combines two powerful approaches:
        
        **1. Similarity-Based Classification**
        - Compares new questions to existing topics using AI embeddings
        - Questions above similarity threshold → classified to existing topics
        - Fast and accurate for questions similar to known topics
        
        **2. Clustering-Based Topic Discovery**
        - Groups remaining questions using machine learning
        - Discovers new topic patterns automatically
        - Generates topic names using GPT
        
        This approach ensures both efficiency and discovery!
        """)
    
    with col2:
        st.subheader("How Elder Edwards Uses This")
        st.markdown("""
        **Weekly Workflow:**
        
        1. **Upload** new student questions
        2. **Connect** to Google Sheets with existing topics
        3. **Configure** similarity threshold (usually 0.70)
        4. **Run** hybrid analysis
        5. **Review** discovered topics
        6. **Approve/Reject** new topics for inclusion
        7. **Update** master topic list
        8. **Download** reports for administration
        
        The system learns and improves over time!
        """)
    
    st.subheader("🚀 Key Features")
    
    features_col1, features_col2, features_col3 = st.columns(3)
    
    with features_col1:
        st.markdown("""
        **📊 Data Sources**
        - File uploads (CSV/TXT)
        - Google Sheets integration
        - Permission management
        """)
    
    with features_col2:
        st.markdown("""
        **🤖 AI-Powered**
        - GPT-5 nano/mini models
        - Advanced embeddings
        - Semantic similarity
        - Automated topic naming
        """)
    
    with features_col3:
        st.markdown("""
        **📝 Management Tools**
        - View and download results
        - Export to CSV
        - Analysis statistics
        - Real-time processing
        """)
    
    st.subheader("📁 Output Files")
    st.markdown("""
    The system generates **three important files** for Elder Edwards:
    
    1. **Similar Questions File**: Questions matched to existing topics with similarity scores
    2. **New Topics File**: Newly discovered topics with representative questions
    3. **Complete Review File**: All questions with topic assignments for comprehensive review
    
    These files help Elder Edwards make informed decisions about topic management.
    """)
    
    # Model information
    with st.expander("**Technical Details**", expanded=False):
        st.markdown(f"""
        **Models Used:**
        - **Embeddings**: {EMBEDDING_MODEL} ({EMBEDDING_DIMENSIONS} dimensions)
        - **Chat**: {CHAT_MODEL} (for topic naming)
        - **Clustering**: HDBSCAN with UMAP dimensionality reduction
        
        **Configuration:**
        - **UMAP Components**: {UMAP_N_COMPONENTS}
        - **Min Cluster Size**: {MIN_CLUSTER_SIZE} (minimum questions per topic)
        - **Default Similarity Threshold**: {SIMILARITY_THRESHOLD}
        - **Caching**: {"Enabled" if CACHE_EMBEDDINGS else "Disabled"} for embeddings
        
        **Performance:**
        - Handles thousands of questions efficiently
        - Caching reduces API calls by ~80%
        - Auto-retry logic for robust operation
        """)


if __name__ == "__main__":
    main()