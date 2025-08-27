"""
Configuration settings for BYU Pathway Questions Analysis App
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Model Configuration  
EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', "text-embedding-3-large")
CHAT_MODEL = os.getenv('OPENAI_CHAT_MODEL', "gpt-4o-mini")

# Clustering Configuration
MIN_CLUSTER_SIZE = 15
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5
UMAP_METRIC = 'cosine'
HDBSCAN_METRIC = 'euclidean'
HDBSCAN_CLUSTER_SELECTION_METHOD = 'eom'

# Vectorizer Configuration
MAX_FEATURES = 1000
STOP_WORDS = "english"

# File paths
RESULTS_DIR = "results"
DATA_DIR = "data"

# UI Configuration
PAGE_TITLE = "BYU Pathway Missionary Questions Analysis"
PAGE_ICON = "🎓"
LAYOUT = "wide"

# Custom CSS styles
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
    }
    .topic-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Fix select dropdown styling */
    .stSelectbox > div > div {
        background-color: white !important;
        color: #262730 !important;
    }
    
    .stSelectbox > div > div > div {
        background-color: white !important;
        color: #262730 !important;
    }
    
    .stSelectbox label {
        color: #262730 !important;
    }
    
    /* Fix select dropdown options */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: white !important;
        color: #262730 !important;
    }
    
    /* Ensure dropdown text is visible */
    .stSelectbox div[data-baseweb="select"] div {
        color: #262730 !important;
    }
    
    .stSelectbox div[role="option"] {
        background-color: white !important;
        color: #262730 !important;
    }
</style>
"""
