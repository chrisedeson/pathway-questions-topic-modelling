"""
Configuration settings for BYU Pathway Questions Analysis App
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Model Configuration - Updated for GPT-5 support
EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv('OPENAI_EMBEDDING_DIMENSIONS', "1536"))
CHAT_MODEL = os.getenv('OPENAI_CHAT_MODEL', "gpt-5-nano")

# Hybrid Processing Configuration
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', "0.70"))
REPRESENTATIVE_QUESTION_METHOD = os.getenv('REPRESENTATIVE_QUESTION_METHOD', "centroid")
PROCESSING_MODE = os.getenv('PROCESSING_MODE', "sample")
SAMPLE_SIZE = int(os.getenv('SAMPLE_SIZE', "2000"))

# Clustering Configuration - Updated based on hybrid notebook
MIN_CLUSTER_SIZE = int(os.getenv('HDBSCAN_MIN_CLUSTER_SIZE', "3"))  # Tighter clusters
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = int(os.getenv('UMAP_N_COMPONENTS', "5"))
UMAP_METRIC = 'cosine'
HDBSCAN_METRIC = 'euclidean'
HDBSCAN_CLUSTER_SELECTION_METHOD = 'eom'
RANDOM_SEED = int(os.getenv('RANDOM_SEED', "42"))

# Caching Configuration
CACHE_EMBEDDINGS = os.getenv('CACHE_EMBEDDINGS', 'true').lower() == 'true'
CACHE_DIR = os.getenv('CACHE_DIR', "embeddings_cache/")

# Google Sheets Configuration
GOOGLE_SHEETS_CREDENTIALS_PATH = os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH', "credentials/byu-pathway-chatbot-service-account.json")
GOOGLE_SHEETS_AUTO_REFRESH_INTERVAL = int(os.getenv('GOOGLE_SHEETS_AUTO_REFRESH_INTERVAL', "10"))

# Vectorizer Configuration
MAX_FEATURES = 1000
STOP_WORDS = "english"

# File paths
RESULTS_DIR = "results"
DATA_DIR = "data"

# UI Configuration
PAGE_TITLE = "BYU Pathway Hybrid Topic Analysis"
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
    
    /* Enhanced selectbox styling for both themes with simplified approach */
    
    /* Force selectbox text to be visible in all cases */
    .stSelectbox label {
        color: var(--text-color, #262730) !important;
    }
    
    /* Main selectbox container */
    .stSelectbox > div > div,
    .stSelectbox div[data-baseweb="select"] > div,
    .stSelectbox div[data-baseweb="select"] {
        background-color: var(--background-color, white) !important;
        color: var(--text-color, #262730) !important;
        border: 1px solid var(--border-color, #d1d5db) !important;
    }
    
    /* All text inside selectbox */
    .stSelectbox div[data-baseweb="select"] *,
    .stSelectbox div[data-baseweb="select"] div,
    .stSelectbox div[data-baseweb="select"] span {
        color: var(--text-color, #262730) !important;
        opacity: 1 !important;
    }
    
    /* Dropdown options */
    .stSelectbox div[role="listbox"] {
        background-color: var(--background-color, white) !important;
        border: 1px solid var(--border-color, #d1d5db) !important;
    }
    
    .stSelectbox div[role="option"] {
        background-color: var(--background-color, white) !important;
        color: var(--text-color, #262730) !important;
    }
    
    .stSelectbox div[role="option"]:hover {
        background-color: var(--secondary-background-color, #f8f9fa) !important;
        color: var(--text-color, #262730) !important;
    }
    
    /* Nuclear option - force ALL selectbox text to be visible */
    .stSelectbox * {
        color: var(--text-color, #262730) !important;
        opacity: 1 !important;
    }
    
    /* Specifically target the selected value display */
    .stSelectbox [data-baseweb="select"] [data-testid] {
        color: var(--text-color, #262730) !important;
    }
    
    /* Override any inherited dark styles */
    [data-theme="dark"] .stSelectbox *,
    .stApp[data-testid="stApp"] .stSelectbox * {
        color: #fafafa !important;
    }
    
    [data-theme="dark"] .stSelectbox > div > div,
    [data-theme="dark"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: #262730 !important;
        color: #fafafa !important;
        border: 1px solid #30363d !important;
    }
    
    [data-theme="dark"] .stSelectbox div[role="listbox"] {
        background-color: #262730 !important;
        border: 1px solid #30363d !important;
    }
    
    [data-theme="dark"] .stSelectbox div[role="option"] {
        background-color: #262730 !important;
        color: #fafafa !important;
    }
</style>
"""
