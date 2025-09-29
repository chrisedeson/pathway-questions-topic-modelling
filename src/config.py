"""
Configuration settings for BYU Pathway Questions Analysis App
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Model Configuration - Updated for latest OpenAI models
EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv('OPENAI_EMBEDDING_DIMENSIONS', "1536"))
CHAT_MODEL = os.getenv('OPENAI_CHAT_MODEL', "gpt-5-nano")

# Hybrid Processing Configuration
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', "0.70"))
REPRESENTATIVE_QUESTION_METHOD = os.getenv('REPRESENTATIVE_QUESTION_METHOD', "centroid")
PROCESSING_MODE = os.getenv('PROCESSING_MODE', "sample")
SAMPLE_SIZE = int(os.getenv('SAMPLE_SIZE', "2000"))

# Clustering Configuration - Updated based on hybrid notebook
MIN_CLUSTER_SIZE = int(os.getenv('HDBSCAN_MIN_CLUSTER_SIZE', "8"))  # Increased to reduce over-clustering
UMAP_N_COMPONENTS = int(os.getenv('UMAP_N_COMPONENTS', "5"))

# Concurrent Processing Configuration
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', "5"))  # OpenAI rate limit friendly
ENABLE_ASYNC_PROCESSING = os.getenv('ENABLE_ASYNC_PROCESSING', "true").lower() == "true"
RANDOM_SEED = int(os.getenv('RANDOM_SEED', "42"))

# Caching Configuration
CACHE_EMBEDDINGS = os.getenv('CACHE_EMBEDDINGS', 'true').lower() == 'true'
CACHE_DIR = os.getenv('CACHE_DIR', "embeddings_cache/")

# Google Sheets Configuration
GOOGLE_SHEETS_CREDENTIALS_PATH = os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH', "credentials/byu-pathway-chatbot-service-account.json")

# File paths
RESULTS_DIR = "results"
DATA_DIR = "data"

# UI Configuration
PAGE_TITLE = "BYU Pathway Hybrid Topic Analysis"
PAGE_ICON = ""
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
    
    /* Enhanced selectbox and form control styling for dark mode compatibility */
    
    /* Selectbox container and input */
    .stSelectbox > div > div[data-baseweb="select"] {
        background-color: transparent !important;
        border: 1px solid #d1d5db !important;
        border-radius: 4px !important;
    }
    
    /* Selectbox text and selected value */
    .stSelectbox div[data-baseweb="select"] div,
    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="select"] {
        color: inherit !important;
        background-color: transparent !important;
    }
    
    /* Dark mode selectbox styling */
    [data-theme="dark"] .stSelectbox > div > div[data-baseweb="select"] {
        border: 1px solid #4a5568 !important;
        background-color: #2d3748 !important;
    }
    
    [data-theme="dark"] .stSelectbox div[data-baseweb="select"] div,
    [data-theme="dark"] .stSelectbox div[data-baseweb="select"] span {
        color: #ffffff !important;
    }
    
    /* Dropdown menu styling */
    .stSelectbox div[role="listbox"] {
        background-color: white !important;
        border: 1px solid #d1d5db !important;
        border-radius: 4px !important;
    }
    
    [data-theme="dark"] .stSelectbox div[role="listbox"] {
        background-color: #2d3748 !important;
        border: 1px solid #4a5568 !important;
    }
    
    /* Dropdown options */
    .stSelectbox div[role="option"] {
        color: #1a202c !important;
        background-color: transparent !important;
    }
    
    .stSelectbox div[role="option"]:hover {
        background-color: #f7fafc !important;
    }
    
    [data-theme="dark"] .stSelectbox div[role="option"] {
        color: #ffffff !important;
    }
    
    [data-theme="dark"] .stSelectbox div[role="option"]:hover {
        background-color: #4a5568 !important;
    }
    
    /* Number input dark mode fixes */
    [data-theme="dark"] .stNumberInput > div > div > input {
        background-color: #2d3748 !important;
        color: #ffffff !important;
        border: 1px solid #4a5568 !important;
    }
    
    [data-theme="dark"] .stNumberInput button {
        background-color: #4a5568 !important;
        color: #ffffff !important;
        border: 1px solid #4a5568 !important;
    }
</style>
"""
