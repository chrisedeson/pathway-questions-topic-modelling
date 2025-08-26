"""
Utility functions for the BYU Pathway Questions Analysis App
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
from typing import Tuple, Optional, List
import streamlit as st


def load_analysis_results() -> Optional[pd.DataFrame]:
    """Load the latest analysis results from the results directory"""
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    
    # Find the most recent results file
    csv_files = list(results_dir.glob("pathway_questions_analysis_*.csv"))
    if not csv_files:
        return None
    
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    return pd.read_csv(latest_file)


def load_topic_model():
    """Load saved topic model if available"""
    model_files = list(Path(".").glob("topic_model_*.pkl"))
    if not model_files:
        return None
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    with open(latest_model, 'rb') as f:
        return pickle.load(f)


def save_results(df: pd.DataFrame, topic_model=None) -> str:
    """Save analysis results and optionally the topic model"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save CSV results
    csv_path = results_dir / f"pathway_questions_analysis_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # Save topic model if provided
    if topic_model is not None:
        model_path = f"topic_model_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(topic_model, f)
    
    return str(csv_path)


def create_download_csv(df: pd.DataFrame, include_representation: bool = False) -> str:
    """Create CSV for download with specified columns"""
    if include_representation:
        # Create the format requested in tasks - representation and question columns
        download_df = df[['Topic_Name', 'Question']].copy()
        download_df = download_df.rename(columns={'Topic_Name': 'representation'})
        # Sort by representation (topic) and then by question alphabetically
        download_df = download_df.sort_values(['representation', 'Question'])
    else:
        download_df = df
    
    return download_df.to_csv(index=False)


def calculate_clustering_metrics(df: pd.DataFrame, embeddings: Optional[np.ndarray] = None) -> dict:
    """Calculate detailed clustering metrics as specified in tasks"""
    total_questions = len(df)
    clustered_questions = len(df[df['Topic_ID'] != -1])
    unclustered_questions = len(df[df['Topic_ID'] == -1])
    
    # Count unique clusters (excluding noise cluster -1)
    unique_clusters = len(df[df['Topic_ID'] != -1]['Topic_ID'].unique())
    noise_points = unclustered_questions
    noise_percentage = (noise_points / total_questions) * 100
    categorized_percentage = (clustered_questions / total_questions) * 100
    
    metrics = {
        'total_questions': total_questions,
        'clusters_found': unique_clusters,
        'questions_clustered': clustered_questions,
        'questions_not_clustered': unclustered_questions,
        'noise_points': noise_points,
        'noise_percentage': noise_percentage,
        'categorized_percentage': categorized_percentage,
        'min_cluster_size': 15  # As requested in tasks
    }
    
    # Add embeddings shape if available
    if embeddings is not None:
        metrics['embeddings_shape'] = embeddings.shape
    
    return metrics


def format_metrics_display(metrics: dict) -> str:
    """Format metrics for display as specified in the tasks"""
    display_text = f"""
**Clustering Results:**
• Number of clusters found: {metrics['clusters_found']}
• Number of questions clustered: {metrics['questions_clustered']}
• Number of questions not clustered: {metrics['questions_not_clustered']}
• Clusters found: {metrics['clusters_found']}
• Noise points: {metrics['noise_points']} ({metrics['noise_percentage']:.1f}%)
• Questions categorized: {metrics['categorized_percentage']:.1f}%
• Min Cluster Size: {metrics['min_cluster_size']}

**Configuration:**
• Embedding Model: text-embedding-3-large
• Chat Model: gpt-4o-mini
"""
    
    if 'embeddings_shape' in metrics:
        display_text += f"\n**Embeddings:**\n• Shape: {metrics['embeddings_shape']}"
    
    return display_text


def validate_questions_file(content: str) -> Tuple[bool, List[str], str]:
    """Validate uploaded questions file and return status, questions, and message"""
    questions = [line.strip() for line in content.split('\n') if line.strip()]
    
    if len(questions) < 10:
        return False, questions, f"❌ Not enough questions. Found {len(questions)}, need at least 10 for meaningful analysis."
    
    if len(questions) < 50:
        return True, questions, f"⚠️ Warning: Only {len(questions)} questions found. For better results, consider using 50+ questions."
    
    return True, questions, f"✅ File loaded: {len(questions)} questions found"


def create_session_state_defaults():
    """Initialize default session state values"""
    defaults = {
        'analysis_complete': False,
        'current_results': None,
        'current_topic_model': None,
        'current_embeddings': None,  # Store embeddings for enhanced metrics
        'clustering_metrics': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
