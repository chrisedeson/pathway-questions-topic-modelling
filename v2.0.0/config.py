"""
Configuration settings for BYU Pathway Topic Analysis Dashboard v2.0.0
"""

import os
from pathlib import Path

# ============ AWS S3 Configuration ============
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "byupathway-public")
AWS_S3_PREFIX = os.getenv("AWS_S3_PREFIX", "topic-modeling-data")
AWS_REGION = "us-east-1"

# ============ File Naming Patterns ============
# The notebook outputs these files with timestamp suffixes
FILE_PATTERNS = {
    "similar_questions": "similar_questions_*.parquet",
    "new_topics": "new_topics_*.parquet",
    "pathway_questions_review": "pathway_questions_review_*.parquet",
    "topic_distribution": "topic_distribution_*.parquet",
    "error_log": "error_log_*.json"
}

# ============ Streamlit Page Configuration ============
PAGE_CONFIG = {
    "page_title": "BYU Pathway - Topic Analysis Dashboard",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# ============ Dashboard Default Columns ============
# Columns shown by default in the main table
DEFAULT_VISIBLE_COLUMNS = [
    "input",  # The question
    "timestamp",
    "country",
    "state"
]

# All available columns (user can show/hide these)
ALL_AVAILABLE_COLUMNS = [
    "input",
    "timestamp",
    "country",
    "state",
    "city",
    "matched_topic",
    "matched_subtopic",
    "similarity_score",
    "classification",
    "output",
    "user_feedback",
    "user_language",
    "ip_address",
    "is_suspicious",
    "cluster_id",
    "topic_keywords"
]

# Column display names (for better UX)
COLUMN_DISPLAY_NAMES = {
    "input": "Question",
    "timestamp": "Timestamp",
    "country": "Country",
    "state": "State/Province",
    "city": "City",
    "matched_topic": "Matched Topic",
    "matched_subtopic": "Matched Subtopic",
    "similarity_score": "Similarity Score",
    "classification": "Classification",
    "output": "Response",
    "user_feedback": "User Feedback",
    "user_language": "Language",
    "ip_address": "IP Address",
    "is_suspicious": "Suspicious",
    "cluster_id": "Cluster ID",
    "topic_keywords": "Keywords"
}

# ============ Data Type Mappings ============
COLUMN_TYPES = {
    "similarity_score": "float",
    "timestamp": "datetime",
    "is_suspicious": "bool"
}

# ============ Filter Options ============
CLASSIFICATION_OPTIONS = ["All", "Existing Topic", "New Topic"]
SORT_OPTIONS = {
    "Timestamp (Newest First)": ("timestamp", False),
    "Timestamp (Oldest First)": ("timestamp", True),
    "Similarity Score (High to Low)": ("similarity_score", False),
    "Similarity Score (Low to High)": ("similarity_score", True),
    "Country (A-Z)": ("country", True),
    "Country (Z-A)": ("country", False)
}

# ============ Cache Settings ============
CACHE_TTL = 3600  # Cache data for 1 hour (in seconds)

# ============ Styling ============
CUSTOM_CSS = """
<style>
    /* Main container */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }
    
    /* Headers */
    h1 {
        color: #002E5D;
        font-weight: 700;
    }
    
    h2 {
        color: #002E5D;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: #666;
        font-weight: 500;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #002E5D;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    /* Warning boxes */
    .warning-box {
        background-color: #fff4e5;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    /* Success boxes */
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #002E5D;
        color: white;
        font-weight: 500;
        border-radius: 4px;
        border: none;
        padding: 0.5rem 2rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #004080;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #002E5D;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""

# ============ Chart Colors ============
BYU_COLORS = {
    "primary": "#002E5D",  # BYU Navy
    "secondary": "#FFB933",  # BYU Gold
    "accent1": "#0062B8",  # BYU Blue
    "accent2": "#C5050C",  # BYU Red
    "neutral": "#666666",
    "light": "#F5F5F5"
}

CHART_COLOR_PALETTE = [
    "#002E5D", "#FFB933", "#0062B8", "#C5050C", 
    "#5D7B9D", "#FFD700", "#4A90E2", "#E74C3C",
    "#95A5A6", "#F39C12", "#3498DB", "#E67E22"
]
