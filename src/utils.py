"""
Utility functions for the BYU Pathway Questions Analysis App
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional, List
import streamlit as st


def calculate_clustering_metrics(df: pd.DataFrame, embeddings: Optional[np.ndarray] = None) -> dict:
    """Calculate detailed clustering metrics"""
    total_questions = len(df)
    
    # Count clusters (excluding noise/ungrouped)
    clusters_found = df['Topic'].nunique() - (1 if -1 in df['Topic'].values else 0)
    
    # Count clustered vs unclustered questions
    questions_clustered = len(df[df['Topic'] != -1])
    questions_not_clustered = len(df[df['Topic'] == -1])
    noise_points = questions_not_clustered
    
    # Calculate percentages
    noise_percentage = (noise_points / total_questions) * 100 if total_questions > 0 else 0
    categorized_percentage = (questions_clustered / total_questions) * 100 if total_questions > 0 else 0
    
    metrics = {
        'total_questions': total_questions,
        'clusters_found': clusters_found,
        'questions_clustered': questions_clustered,
        'questions_not_clustered': questions_not_clustered,
        'noise_points': noise_points,
        'noise_percentage': noise_percentage,
        'categorized_percentage': categorized_percentage,
        'min_cluster_size': 3  # From config
    }
    
    if embeddings is not None:
        metrics['embeddings_shape'] = embeddings.shape
    
    return metrics


def validate_questions_file(content: str) -> Tuple[bool, List[str], str]:
    """Validate uploaded questions file and return status, questions, and message"""
    questions = [line.strip() for line in content.split('\n') if line.strip()]
    
    if len(questions) < 10:
        return False, questions, f"❌ Not enough questions. Found {len(questions)}, need at least 10 for meaningful analysis."
    
    if len(questions) < 50:
        return True, questions, f"⚠️ Warning: Only {len(questions)} questions found. For better results, consider using 50+ questions."
    
    return True, questions, f"✅ Found {len(questions)} questions. Ready for analysis!"


def create_session_state_defaults():
    """Initialize session state with default values"""
    defaults = {
        'analysis_complete': False,
        'current_results': None,
        'current_topic_model': None, 
        'current_embeddings': None,
        'clustering_metrics': None,
        'uploaded_file_name': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value