"""
Topic modeling functionality for BYU Pathway Questions Analysis
"""
import pandas as pd
import numpy as np
from openai import OpenAI, AsyncOpenAI
import umap.umap_ as umap
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from bertopic import BERTopic
import streamlit as st
from typing import Tuple, Optional, List, Dict
import time
import asyncio

from config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, CHAT_MODEL, MIN_CLUSTER_SIZE,
    UMAP_N_NEIGHBORS, UMAP_N_COMPONENTS, UMAP_METRIC, HDBSCAN_METRIC,
    HDBSCAN_CLUSTER_SELECTION_METHOD, MAX_FEATURES, STOP_WORDS,
    MAX_CONCURRENT_REQUESTS, ENABLE_ASYNC_PROCESSING
)


def get_openai_embeddings(questions: List[str], client: OpenAI) -> np.ndarray:
    """Generate embeddings for questions using OpenAI API with batching"""
    # Removed duplicate "Creating embeddings..." message - it's shown in main process
    
    # Process in batches to avoid API limits
    batch_size = 1000
    all_embeddings = []
    
    total_batches = (len(questions) + batch_size - 1) // batch_size
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        try:
            response = client.embeddings.create(
                input=batch,
                model=EMBEDDING_MODEL
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # Small delay to respect rate limits
            time.sleep(0.1)
            
        except Exception as e:
            st.error(f"Error processing batch {batch_num}: {str(e)}")
            raise
    
    return np.array(all_embeddings)


def perform_clustering(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform UMAP + HDBSCAN clustering"""
    
    # UMAP for clustering - reduce to 5D for clustering
    umap_model = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS, 
        n_components=UMAP_N_COMPONENTS, 
        random_state=42, 
        metric=UMAP_METRIC
    )
    umap_embeddings = umap_model.fit_transform(embeddings)
    
    # HDBSCAN clustering with updated min_cluster_size
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        metric=HDBSCAN_METRIC, 
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD
    )
    cluster_labels = hdbscan_model.fit_predict(umap_embeddings)
    
    return umap_embeddings, cluster_labels


def create_topic_model(questions: List[str], embeddings: np.ndarray) -> BERTopic:
    """Create and fit BERTopic model"""
    
    # Create UMAP model for BERTopic (2D for visualization)
    umap_model = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS, 
        n_components=UMAP_N_COMPONENTS, 
        random_state=42, 
        metric=UMAP_METRIC
    )
    
    # Create HDBSCAN model with updated min_cluster_size
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        metric=HDBSCAN_METRIC, 
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD
    )
    
    # Create vectorizer
    vectorizer_model = CountVectorizer(stop_words=STOP_WORDS, max_features=MAX_FEATURES)
    
    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model=None,  # We provide embeddings directly
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        verbose=True
    )
    
    # Fit the model
    topics, probs = topic_model.fit_transform(questions, embeddings)
    
    return topic_model, topics, probs


async def generate_single_topic_label(async_client: AsyncOpenAI, topic_id: int, keywords: List[Tuple[str, float]], semaphore: asyncio.Semaphore) -> Tuple[int, str]:
    """Generate a single topic label asynchronously with rate limiting"""
    async with semaphore:
        keyword_str = ", ".join([word for word, _ in keywords])
        
        prompt = f"""Based on these keywords from student questions: {keyword_str}

Create a clear, concise topic label (2-4 words) that describes the main theme.
Focus on what students are asking about. Examples: "Course Registration", "Financial Aid", "Technical Support"

Topic label:"""
        
        try:
            response = await async_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=20,
                temperature=0.3
            )
            
            label = response.choices[0].message.content.strip().strip('"')
            return topic_id, label
            
        except Exception as e:
            st.warning(f"Could not enhance label for topic {topic_id}: {str(e)}")
            return topic_id, f"Topic {topic_id}"


async def enhance_topic_labels_async(topic_model: BERTopic, async_client: AsyncOpenAI) -> Dict[int, str]:
    """Enhance topic labels using concurrent OpenAI API calls"""
    
    topic_info = topic_model.get_topic_info()
    enhanced_labels = {}
    
    # Skip noise topic
    valid_topics = [tid for tid in topic_info['Topic'].unique() if tid != -1]
    
    if not valid_topics:
        return enhanced_labels
    
    # Create semaphore to limit concurrent requests (configurable, optimal for OpenAI rate limits)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Create tasks for all topics
    tasks = []
    for topic_id in valid_topics:
        keywords = topic_model.get_topic(topic_id)[:10]
        task = generate_single_topic_label(async_client, topic_id, keywords, semaphore)
        tasks.append(task)
    
    # Execute all tasks concurrently
    st.write(f"🔄 Generating {len(tasks)} topic labels concurrently (max {MAX_CONCURRENT_REQUESTS} parallel requests)...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for result in results:
        if isinstance(result, Exception):
            st.warning(f"Error in topic labeling: {result}")
        else:
            topic_id, label = result
            enhanced_labels[topic_id] = label
    
    return enhanced_labels


def enhance_topic_labels(topic_model: BERTopic, client: OpenAI) -> dict:
    """Enhanced topic labels using concurrent processing (wrapper for async function)"""
    # Create async client from the sync client's API key
    async_client = AsyncOpenAI(api_key=client.api_key)
    
    # Run the async function
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # If we're already in an event loop (like in Jupyter/Streamlit)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, enhance_topic_labels_async(topic_model, async_client))
            return future.result()
    else:
        # If no event loop is running
        return asyncio.run(enhance_topic_labels_async(topic_model, async_client))


def create_results_dataframe(questions: List[str], topics: List[int], 
                           probs: List[float], enhanced_labels: dict) -> pd.DataFrame:
    """Create results DataFrame with all necessary columns"""
    
    # Create the main results DataFrame
    results_df = pd.DataFrame({
        'Question': questions,
        'Topic_ID': topics,
        'Probability': probs,
        'Topic_Name': [enhanced_labels.get(topic_id, f"Topic {topic_id}") 
                      if topic_id != -1 else "Uncategorized" 
                      for topic_id in topics]
    })
    
    return results_df


def process_questions_file(uploaded_file) -> Optional[Tuple[pd.DataFrame, BERTopic, np.ndarray]]:
    """Main processing function that orchestrates the entire analysis pipeline"""
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Read and validate file
        content = uploaded_file.read().decode('utf-8')
        questions = [line.strip() for line in content.split('\n') if line.strip()]
        
        if len(questions) < 10:
            st.error(f"❌ Not enough questions. Found {len(questions)}, need at least 10 for meaningful analysis.")
            return None
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Generate embeddings
        status_text.text("🔄 Creating embeddings...")
        progress_bar.progress(10)
        embeddings = get_openai_embeddings(questions, client)
        progress_bar.progress(50)
        
        # Step 2: Create topic model
        status_text.text("🔄 Creating topic model...")
        progress_bar.progress(60)
        topic_model, topics, probs = create_topic_model(questions, embeddings)
        progress_bar.progress(80)
        
        # Step 3: Enhance labels
        status_text.text("🔄 Enhancing topic labels...")
        progress_bar.progress(90)
        enhanced_labels = enhance_topic_labels(topic_model, client)
        
        # Step 4: Create results DataFrame
        results_df = create_results_dataframe(questions, topics, probs, enhanced_labels)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        return results_df, topic_model, embeddings
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        return None
