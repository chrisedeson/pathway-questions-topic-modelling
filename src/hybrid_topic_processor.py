"""
Hybrid Topic Discovery and Classification utilities for BYU Pathway Questions Analysis
Integrates similarity-based classification with clustering-based topic discovery
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
import openai
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from hdbscan import HDBSCAN
import umap
from bertopic import BERTopic
import asyncio
import backoff
from openai import AsyncOpenAI, OpenAI
from openai import APIStatusError
from tqdm import tqdm
import hashlib
import pickle
import os
from tqdm import tqdm
from pathlib import Path
import time

from config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, CHAT_MODEL,
    SIMILARITY_THRESHOLD, REPRESENTATIVE_QUESTION_METHOD, PROCESSING_MODE,
    SAMPLE_SIZE, MIN_CLUSTER_SIZE, UMAP_N_COMPONENTS, MAX_CONCURRENT_REQUESTS,
    ENABLE_ASYNC_PROCESSING, CACHE_EMBEDDINGS, CACHE_DIR, RANDOM_SEED
)

logger = logging.getLogger(__name__)

def clean_question(text: str) -> str:
    """Remove ACM prefixes and clean question text for better processing"""
    if not text:
        return ""
    
    # Remove common prefixes that might interfere with embeddings
    prefixes_to_remove = [
        "ACM ", "acm ", "ACM: ", "acm: ",
        "Q: ", "q: ", "Question: ", "question: "
    ]
    
    cleaned = text
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    
    # Clean up whitespace and normalize
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()

class HybridTopicProcessor:
    """Optimized hybrid topic processor with advanced caching and batch processing"""
    
    def __init__(self):
        """Initialize the processor with OpenAI clients and caching setup"""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Setup caching
        if CACHE_EMBEDDINGS and CACHE_DIR:
            Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
            logger.info(f"Caching enabled: {CACHE_DIR}")
        else:
            logger.info("Caching disabled")
        
        # Initialize batch cache
        self._batch_cache = None
        self._batch_cache_path = os.path.join(CACHE_DIR, "batch_embeddings_cache.pkl")
        self._cache_modified = False
    
    def _load_batch_cache(self):
        """Load the batch cache from disk"""
        if self._batch_cache is not None:
            return
        
        if os.path.exists(self._batch_cache_path):
            try:
                with open(self._batch_cache_path, 'rb') as f:
                    self._batch_cache = pickle.load(f)
                logger.info(f"Loaded batch cache with {sum(len(model_cache) for model_cache in self._batch_cache.values())} embeddings")
            except Exception as e:
                logger.warning(f"Failed to load batch cache: {e}")
                self._batch_cache = {}
        else:
            self._batch_cache = {}
    
    def _save_batch_cache(self):
        """Save the batch cache to disk if modified"""
        if not self._cache_modified or not CACHE_EMBEDDINGS:
            return
        
        try:
            with open(self._batch_cache_path, 'wb') as f:
                pickle.dump(self._batch_cache, f)
            self._cache_modified = False
            logger.debug("Batch cache saved")
        except Exception as e:
            logger.warning(f"Failed to save batch cache: {e}")
    
    def get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
    
    def load_cached_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Load embedding from batch cache"""
        if not CACHE_EMBEDDINGS:
            return None
        
        self._load_batch_cache()
        cache_key = self.get_cache_key(text, model)
        
        if model in self._batch_cache and cache_key in self._batch_cache[model]:
            return self._batch_cache[model][cache_key]
        
        return None
    
    def save_embedding_to_cache(self, text: str, model: str, embedding: List[float]):
        """Save embedding to batch cache"""
        if not CACHE_EMBEDDINGS:
            return
        
        self._load_batch_cache()
        cache_key = self.get_cache_key(text, model)
        
        if model not in self._batch_cache:
            self._batch_cache[model] = {}
        
        self._batch_cache[model][cache_key] = embedding
        self._cache_modified = True
        
        # Save periodically to avoid memory issues
        if len(self._batch_cache.get(model, {})) % 100 == 0:
            self._save_batch_cache()
    
    def clean_question(self, question: str) -> str:
        """Remove ACM question prefix from questions before processing"""
        if not isinstance(question, str):
            return str(question) if question else ""
        
        # Define patterns to match ACM prefixes (case-insensitive)
        import re
        acm_patterns = [
            r'^\s*\(ACMs?\s+[Qq]uestion\)\s*:?\s*',
            r'^\s*\(ACMs?\s+[Qq]uestions?\)\s*:?\s*',
            r'^\s*ACMs?\s+[Qq]uestion:?\s*',
            r'^\s*ACMs?\s+[Qq]uestions?:?\s*'
        ]
        
        cleaned = question
        for pattern in acm_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()
    
    def preprocess_questions_dataframe(self, df: pd.DataFrame, question_column: str = 'question') -> pd.DataFrame:
        """Preprocess questions DataFrame"""
        df = df.copy()
        
        # Clean questions
        if question_column in df.columns:
            df[question_column] = df[question_column].apply(self.clean_question)
        
        # Remove empty questions
        df = df[df[question_column].notna() & (df[question_column].str.strip() != '')]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=[question_column])
        
        return df.reset_index(drop=True)
    
    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APITimeoutError, openai.APIError),
        max_tries=3,
        base=2,
        max_value=60
    )
    def get_embedding(self, text: str, model: str = EMBEDDING_MODEL) -> List[float]:
        """Get embedding for text with caching and retry logic"""
        # Clean question text before processing
        cleaned_text = clean_question(text)
        
        # Check cache first (using cleaned text)
        cached_embedding = self.load_cached_embedding(cleaned_text, model)
        if cached_embedding is not None:
            return cached_embedding
        
        try:
            response = self.client.embeddings.create(
                input=cleaned_text.replace("\n", " "),  # Clean text further
                model=model,
                dimensions=EMBEDDING_DIMENSIONS if "3-small" in model or "3-large" in model else None
            )
            embedding = response.data[0].embedding
            
            # Cache the result using cleaned text
            self.save_embedding_to_cache(cleaned_text, model, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str], model: str = EMBEDDING_MODEL, batch_size: int = 1000) -> List[List[float]]:
        """Get embeddings for multiple texts with true batch processing, caching, and question preprocessing"""
        
        # Clean all texts first (remove ACM prefixes)
        cleaned_texts = [clean_question(text) for text in texts]
        
        embeddings = []
        cache_hits = 0
        api_calls = 0
        batch_count = 0
        
        # Create Streamlit progress components
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(f"🔄 Generating embeddings for {len(cleaned_texts)} texts...")
        
        # Process in batches for API efficiency
        for i in range(0, len(cleaned_texts), batch_size):
            batch_texts = cleaned_texts[i:i+batch_size]
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            # Check cache for each text in batch
            for j, text in enumerate(batch_texts):
                cached_embedding = self.load_cached_embedding(text, model)
                if cached_embedding is not None:
                    batch_embeddings.append(cached_embedding)
                    cache_hits += 1
                else:
                    batch_embeddings.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(j)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                try:
                    # TRUE BATCH API CALL - multiple texts in single request
                    response = self.client.embeddings.create(
                        model=model,
                        input=uncached_texts
                    )
                    
                    new_embeddings = [data.embedding for data in response.data]
                    api_calls += len(uncached_texts)
                    batch_count += 1
                    
                    # Fill in the uncached embeddings and save to cache
                    for idx, embedding in zip(uncached_indices, new_embeddings):
                        batch_embeddings[idx] = embedding
                        self.save_embedding_to_cache(batch_texts[idx], model, embedding)
                    
                    # Rate limiting for batch API calls
                    if batch_count % 5 == 0:  # Brief pause every 5 batches
                        time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch: {e}")
                    # Fill with None for failed embeddings (will be handled downstream)
                    for idx in uncached_indices:
                        batch_embeddings[idx] = None
            
            embeddings.extend(batch_embeddings)
            
            # Update progress
            progress = (i + len(batch_texts)) / len(cleaned_texts)
            progress_bar.progress(progress, text=f"Processing: {i + len(batch_texts)}/{len(cleaned_texts)}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show summary with cache efficiency stats
        cache_efficiency = (cache_hits / len(embeddings) * 100) if embeddings else 0
        st.success(f"✅ Embedding generation complete! Cache hits: {cache_hits}, API calls: {api_calls}, Cache efficiency: {cache_efficiency:.1f}%")
        
        return embeddings
        
        return embeddings
    
    def find_best_topic_match(self, question_embedding: List[float], topic_embeddings_df: pd.DataFrame) -> Optional[Dict]:
        """Find the best matching topic for a question using cosine similarity (optimized from insights)"""
        
        if not question_embedding or len(question_embedding) != EMBEDDING_DIMENSIONS:
            logger.error(f"Invalid question embedding - expected dimension {EMBEDDING_DIMENSIONS}, got {len(question_embedding) if question_embedding else 0}")
            return None
            
        if len(topic_embeddings_df) == 0:
            return None

        best_distance = float('inf')
        best_match = None

        for idx, row in topic_embeddings_df.iterrows():
            topic_embedding = row['embedding']

            # Error handling for topic embeddings
            if not topic_embedding:
                logger.error(f"No topic embedding for topic '{row.get('Topic', 'Unknown')}' - skipping")
                continue

            if len(topic_embedding) != EMBEDDING_DIMENSIONS:
                logger.error(f"Invalid topic embedding dimension for topic '{row.get('Topic', 'Unknown')}' - expected {EMBEDDING_DIMENSIONS}, got {len(topic_embedding)} - skipping")
                continue

            try:
                # Calculate cosine distance (1 - cosine similarity) - more efficient than sklearn
                distance = cosine(question_embedding, topic_embedding)

                if distance < best_distance:
                    best_distance = distance
                    best_match = {
                        'topic': row.get('Topic', row.get('topic', 'Unknown')),
                        'subtopic': row.get('Subtopic', row.get('subtopic', '')),
                        'topic_question': row.get('Question', row.get('question', '')),
                        'distance': distance,
                        'similarity': 1 - distance  # Convert distance to similarity
                    }
            except Exception as e:
                logger.error(f"Failed to calculate cosine distance for topic '{row.get('Topic', 'Unknown')}': {e} - skipping")
                continue

        return best_match
    
    def classify_by_similarity(self, 
                               questions_df: pd.DataFrame, 
                               topic_questions_df: pd.DataFrame, 
                               threshold: float = SIMILARITY_THRESHOLD) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Classify questions by similarity to existing topics (optimized from insights)"""
        
        st.info(f"🔍 **Step 1: Similarity-Based Classification** (threshold: {threshold})")
        
        # Generate embeddings for existing topic questions
        st.write(f"📊 Generating embeddings for {len(topic_questions_df)} existing topic questions...")
        topic_questions_list = topic_questions_df['Question'].tolist()
        topic_embeddings_list = self.get_embeddings_batch(topic_questions_list)
        topic_embeddings = np.array(topic_embeddings_list)
        
        # Add embeddings to dataframe
        topic_questions_with_embeddings = topic_questions_df.copy()
        topic_questions_with_embeddings['embedding'] = topic_embeddings_list

        # Generate embeddings for student questions
        st.write(f"📊 Generating embeddings for {len(questions_df)} student questions...")
        student_questions_list = questions_df['question'].tolist()
        student_embeddings_list = self.get_embeddings_batch(student_questions_list)
        student_embeddings = np.array(student_embeddings_list)

        # Classify each student question
        similar_questions = []
        remaining_questions = []

        st.write(f"🔍 Classifying {len(questions_df)} questions against existing topics...")

        # Use streamlit progress bar instead of tqdm for better UX
        progress_bar = st.progress(0)
        
        for i, (question, embedding) in enumerate(zip(student_questions_list, student_embeddings_list)):
            if embedding is not None and len(embedding) == EMBEDDING_DIMENSIONS:
                best_match = self.find_best_topic_match(embedding, topic_questions_with_embeddings)

                if best_match and best_match['similarity'] >= threshold:
                    # Question matches existing topic - capture both topic and subtopic
                    similar_questions.append({
                        'question': question,
                        'matched_topic': best_match['topic'],
                        'matched_subtopic': best_match['subtopic'],
                        'matched_topic_question': best_match['topic_question'],
                        'similarity_score': best_match['similarity']
                    })
                else:
                    # Question doesn't match - add to clustering queue
                    remaining_questions.append({
                        'question': question,
                        'embedding': embedding
                    })
            else:
                # Handle failed embedding with error logging
                logger.error(f"Invalid student question embedding - expected dimension {EMBEDDING_DIMENSIONS}, got {len(embedding) if embedding else 0} for question: '{question[:50]}...' - adding to clustering queue with zero vector")
                remaining_questions.append({
                    'question': question,
                    'embedding': [0.0] * EMBEDDING_DIMENSIONS
                })

            # Update progress
            progress = (i + 1) / len(questions_df)
            progress_bar.progress(progress, text=f"Classifying: {i+1}/{len(questions_df)}")

        progress_bar.empty()

        # Convert to DataFrames
        similar_questions_df = pd.DataFrame(similar_questions)
        remaining_questions_df = pd.DataFrame(remaining_questions)

        # Summary statistics
        total_questions = len(questions_df)
        similar_count = len(similar_questions_df)
        remaining_count = len(remaining_questions_df)

        st.success(f"✅ Similarity classification complete!")
        st.write(f"📊 **Results Summary:**")
        st.write(f"   • Total questions processed: **{total_questions}**")
        st.write(f"   • Similar to existing topics (≥{threshold}): **{similar_count}** ({similar_count/total_questions*100:.1f}%)")
        st.write(f"   • Remaining for clustering (<{threshold}): **{remaining_count}** ({remaining_count/total_questions*100:.1f}%)")

        return similar_questions_df, remaining_questions_df
    
    def perform_clustering_analysis(self, remaining_questions_df: pd.DataFrame) -> tuple[Optional[pd.DataFrame], Optional[BERTopic]]:
        """Perform clustering analysis on remaining questions - returns tuple like insights"""
        
        if len(remaining_questions_df) == 0:
            return None, None
            
        st.info("🎯 **Step 2: Clustering-Based Topic Discovery**")
        st.write("Finding new topics in unmatched questions...")
        
        # Get embeddings for remaining questions
        questions = remaining_questions_df['question'].tolist()
        embeddings_list = remaining_questions_df['embedding'].tolist() if 'embedding' in remaining_questions_df.columns else self.get_embeddings_batch(questions)
        
        # Convert to numpy array for BERTopic
        import numpy as np
        embeddings = np.array(embeddings_list)
        
        # Create BERTopic model for better visualizations
        topic_model = None
        try:
            from bertopic import BERTopic
            from sklearn.cluster import HDBSCAN
            import umap.umap_ as umap
            
            # Validate embeddings
            if len(embeddings) == 0 or embeddings.ndim != 2:
                st.warning("⚠️ Could not create BERTopic model: Invalid embeddings format. Falling back to manual clustering.")
                raise ValueError("Invalid embeddings format")
            
            # Initialize UMAP and HDBSCAN models
            umap_model = umap.UMAP(
                n_components=UMAP_N_COMPONENTS,
                random_state=RANDOM_SEED,
                metric='cosine'
            )
            
            hdbscan_model = HDBSCAN(
                min_cluster_size=MIN_CLUSTER_SIZE,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            
            # Create BERTopic model
            topic_model = BERTopic(
                embedding_model=None,  # Use our precomputed embeddings
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                verbose=False
            )
            
            # Fit the model
            topics, probs = topic_model.fit_transform(questions, embeddings)
            
            # Get topic information like insights
            topic_info = topic_model.get_topic_info()
            
            # Create results DataFrame exactly like insights
            clustered_df = pd.DataFrame({
                'question': questions,
                'cluster_id': topics,  # Use topics as cluster_id
                'topic_id': topics,
                'cluster_probability': probs if probs is not None else [1.0] * len(questions)
            })
            
            # Filter out noise points (cluster_id == -1) like insights
            clustered_df = clustered_df[clustered_df['cluster_id'] != -1]
            
            # Add topic keywords like insights
            topic_map = topic_info.set_index("Topic")["Representation"].to_dict()
            
            def get_topic_keywords(topic_id):
                """Extract keywords from topic representation"""
                rep = topic_map.get(topic_id, [])
                if isinstance(rep, list) and len(rep) > 0:
                    return ", ".join(rep[:5])  # Top 5 keywords
                return "Unknown"
            
            clustered_df['topic_keywords'] = clustered_df['topic_id'].apply(get_topic_keywords)
            
        except Exception as e:
            st.warning(f"Could not create BERTopic model: {e}. Falling back to manual clustering.")
            
            # Fallback to manual clustering
            umap_model = umap.UMAP(
                n_components=UMAP_N_COMPONENTS,
                random_state=RANDOM_SEED,
                metric='cosine'
            )
            reduced_embeddings = umap_model.fit_transform(embeddings)
            
            # Clustering with HDBSCAN
            hdbscan_model = HDBSCAN(
                min_cluster_size=MIN_CLUSTER_SIZE,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            cluster_labels = hdbscan_model.fit_predict(reduced_embeddings)
            
            # Add cluster information to DataFrame
            clustered_df = pd.DataFrame({
                'question': questions,
                'cluster_id': cluster_labels,
                'topic_id': cluster_labels,
                'cluster_probability': [0.5] * len(cluster_labels),  # Default probability
                'topic_keywords': ['unknown'] * len(cluster_labels)
            })
            
            # Filter out noise points
            clustered_df = clustered_df[clustered_df['cluster_id'] != -1]
        
        cluster_labels = clustered_df['cluster_id'].tolist() if len(clustered_df) > 0 else []
        n_clusters = len(set(cluster_labels)) if cluster_labels else 0
        
        st.success(f"✅ Found {n_clusters} new topic clusters")
        
        return (
            clustered_df if len(clustered_df) > 0 else None,
            topic_model
        )
    
    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APITimeoutError, openai.APIError),
        max_tries=3,
        base=2,
        max_value=60
    )
    @backoff.on_exception(
        backoff.expo,
        (APIStatusError, asyncio.TimeoutError),
        max_tries=3,
        base=2,
        max_value=60
    )
    async def generate_topic_name(self, questions: List[str], keywords: str = "") -> str:
        """Generate a topic name using GPT for a cluster of questions"""
        
        # Limit to top 10 questions for context (exactly like insights)
        sample_questions = questions[:10]
        questions_text = "\n".join([f"- {q}" for q in sample_questions])

        prompt = f"""
    Based on the following student questions and keywords, generate a concise, descriptive topic name.

QUESTIONS:
{questions_text}

KEYWORDS: {keywords}

Instructions:
- Your answer must be ONLY the topic name (2–8 words), no extra text.
- It should clearly describe the shared theme of the questions.
- Avoid generic labels like "General Questions" or "Miscellaneous."
- Do not include "Topic name:" or quotation marks.
- Use simple, natural English that sounds clear to a student or teacher.

Example:
Questions:
- When does registration open?
- What are the fall 2025 enrollment deadlines?
Keywords: registration, deadlines

Topic name: Fall 2025 Registration Deadlines

Now generate the topic name for the questions above:
"""

        try:
            messages = [
                {"role": "system", "content": "You are an expert at creating clear, descriptive topic names for student question categories."},
                {"role": "user", "content": prompt}
            ]
            
            print(f"Calling GPT with model: {CHAT_MODEL}, keywords: '{keywords[:50]}...'")
            
            response = await self.async_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                max_completion_tokens=1000  # Match insights
            )
            
            topic_name = response.choices[0].message.content.strip()
            print(f"Raw GPT response: '{topic_name}'")
            
            # Clean up the response (exactly like insights)
            topic_name = topic_name.replace("Topic name:", "").strip()
            topic_name = topic_name.strip('"\'')
            
            if not topic_name:
                topic_name = f"Topic: {keywords[:50]}" if keywords else f"Question Group {hash(str(questions[:3])) % 1000}"
                print(f"Empty response, using fallback: '{topic_name}'")
            
            print(f"Final topic name: '{topic_name}' for {len(questions)} questions")
            return topic_name
            
        except Exception as e:
            logger.error(f"Error generating topic name: {e}")
            print(f"Exception occurred: {str(e)}")
            fallback_name = f"Topic: {keywords[:50]}" if keywords else f"Topic Group {hash(str(questions[:3])) % 1000}"
            print(f"Using fallback topic name: '{fallback_name}'")
            return fallback_name
    
    async def generate_topic_names_for_clusters(self, clustered_questions_df: pd.DataFrame) -> Dict[int, str]:
        """Generate topic names for all clusters using topic_keywords from BERTopic like insights"""
        
        st.write("🤖 Generating topic names with GPT...")
        
        topic_names = {}
        cluster_groups = clustered_questions_df.groupby('cluster_id')
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)
        
        async def generate_for_cluster(cluster_id, cluster_data):
            async with semaphore:
                questions = cluster_data['question'].tolist()
                # Use topic_keywords from BERTopic if available (like insights)
                keywords = cluster_data['topic_keywords'].iloc[0] if 'topic_keywords' in cluster_data.columns else ""
                
                topic_name = await self.generate_topic_name(questions, keywords)
                return cluster_id, topic_name
        
        # Generate all topic names concurrently
        tasks = [
            generate_for_cluster(cluster_id, group)
            for cluster_id, group in cluster_groups
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error generating topic name: {result}")
            else:
                cluster_id, topic_name = result
                topic_names[cluster_id] = topic_name
        
        return topic_names
    
    def select_representative_questions(self, 
                                       clustered_questions_df: pd.DataFrame,
                                       method: str = REPRESENTATIVE_QUESTION_METHOD) -> Dict[int, str]:
        """Select representative questions for each cluster"""
        
        representative_questions = {}
        
        for cluster_id, group in clustered_questions_df.groupby('cluster_id'):
            questions = group['question'].tolist()
            
            if method == "centroid":
                # For now, just select the first question
                # In a full implementation, we'd calculate the centroid
                representative_questions[cluster_id] = questions[0]
            else:  # frequent
                # Select the most common question (or first if all unique)
                representative_questions[cluster_id] = questions[0]
        
        return representative_questions
    
    def create_output_files(self,
                           similar_questions_df: pd.DataFrame,
                           clustered_questions_df: Optional[pd.DataFrame],
                           topic_names: Dict[int, str],
                           representative_questions: Dict[int, str]) -> Tuple[str, str, str]:
        """Generate the three required output files"""
        
        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # File 1: Questions similar to existing topics
        file1_name = f"results/similar_questions_{timestamp}.csv"
        
        if len(similar_questions_df) > 0:
            file1_data = similar_questions_df[[
                'question', 'matched_topic', 'matched_subtopic', 'similarity_score'
            ]].copy()
            file1_data.columns = ['question', 'existing_topic', 'existing_subtopic', 'similarity_score']
            file1_data = file1_data.sort_values('similarity_score', ascending=False)
        else:
            file1_data = pd.DataFrame(columns=['question', 'existing_topic', 'existing_subtopic', 'similarity_score'])
        
        file1_data.to_csv(file1_name, index=False)
        
        # File 2: New topics discovered through clustering
        file2_name = f"results/new_topics_{timestamp}.csv"
        
        if clustered_questions_df is not None and len(clustered_questions_df) > 0:
            cluster_summary = []
            
            for cluster_id, group in clustered_questions_df.groupby('cluster_id'):
                topic_name = topic_names.get(cluster_id, f"Cluster_{cluster_id}")
                rep_question = representative_questions.get(cluster_id, group['question'].iloc[0])
                question_count = len(group)
                
                cluster_summary.append({
                    'topic_name': topic_name,
                    'representative_question': rep_question,
                    'question_count': question_count
                })
            
            file2_data = pd.DataFrame(cluster_summary)
            file2_data = file2_data.sort_values('question_count', ascending=False)
        else:
            file2_data = pd.DataFrame(columns=['topic_name', 'representative_question', 'question_count'])
        
        file2_data.to_csv(file2_name, index=False)
        
        # File 3: pathway_questions_review - all questions with topic assignments
        file3_name = f"results/pathway_questions_review_{timestamp}.csv"
        
        pathway_review_data = []
        
        # Add similar questions
        if len(similar_questions_df) > 0:
            for _, row in similar_questions_df.iterrows():
                pathway_review_data.append({
                    'question': row['question'],
                    'topic_name': f"{row['matched_topic']} | {row['matched_subtopic']}"
                })
        
        # Add clustered questions
        if clustered_questions_df is not None and len(clustered_questions_df) > 0:
            for _, row in clustered_questions_df.iterrows():
                topic_name = topic_names.get(row['cluster_id'], f"Cluster_{row['cluster_id']}")
                pathway_review_data.append({
                    'question': row['question'],
                    'topic_name': topic_name
                })
        
        file3_data = pd.DataFrame(pathway_review_data) if pathway_review_data else pd.DataFrame(columns=['question', 'topic_name'])
        file3_data.to_csv(file3_name, index=False)
        
        return file1_name, file2_name, file3_name
    

    async def process_hybrid_analysis(self, 
                                      questions_df: pd.DataFrame,
                                      topic_questions_df: pd.DataFrame,
                                      threshold: float = SIMILARITY_THRESHOLD,
                                      processing_mode: str = "sample",
                                      sample_size: int = SAMPLE_SIZE) -> Dict[str, Any]:
        """Run complete hybrid topic analysis - matches insights flow exactly"""
        
        st.header("🚀 Hybrid Topic Analysis")
        
        # Step 1: Prepare evaluation dataset (like insights)
        if processing_mode == "sample":
            eval_questions_df = questions_df.sample(
                n=min(sample_size, len(questions_df)),
                random_state=RANDOM_SEED
            ).copy()
            st.info(f"🎯 **Sample Mode**: Processing {len(eval_questions_df)} questions (random sample)")
        else:
            eval_questions_df = questions_df.copy()
            st.info(f"🎯 **Full Mode**: Processing all {len(eval_questions_df)} questions")
        
        # Step 2: Similarity-based classification
        similar_questions_df, remaining_questions_df = self.classify_by_similarity(
            eval_questions_df, topic_questions_df, threshold
        )
        
        # Step 3: Clustering-based topic discovery
        clustering_result = self.perform_clustering_analysis(remaining_questions_df)
        if clustering_result[0] is not None:
            clustered_questions_df, topic_model = clustering_result
        else:
            clustered_questions_df = None
            topic_model = None
        
        # Step 4: Generate topic names
        topic_names = {}
        if clustered_questions_df is not None:
            st.info("🤖 **Generating topic names...**")
            topic_names = await self.generate_topic_names_for_clusters(clustered_questions_df)
        
        # Step 5: Select representative questions  
        representative_questions = {}
        if clustered_questions_df is not None:
            # Add embeddings for centroid method like insights
            if REPRESENTATIVE_QUESTION_METHOD == "centroid" and 'embedding' not in clustered_questions_df.columns:
                question_to_embedding = dict(zip(
                    remaining_questions_df['question'],
                    remaining_questions_df['embedding']
                ))
                clustered_questions_df['embedding'] = clustered_questions_df['question'].map(question_to_embedding)
            
            representative_questions = self.select_representative_questions(clustered_questions_df)
        
        # Step 6: Create output files
        st.info("📁 **Creating output files...**")
        file1, file2, file3 = self.create_output_files(
            similar_questions_df, clustered_questions_df, 
            topic_names, representative_questions
        )
        
        # Ensure batch cache is saved
        self._save_batch_cache()
        
        return {
            'similar_questions_df': similar_questions_df,
            'clustered_questions_df': clustered_questions_df,
            'topic_names': topic_names,
            'representative_questions': representative_questions,
            'output_files': [file1, file2, file3],
            'eval_questions_df': eval_questions_df,
            'topic_model': topic_model,
            'similarity_threshold': threshold,
            'processing_mode': processing_mode
        }
    
    def __del__(self):
        """Cleanup: ensure cache is saved when object is destroyed"""
        if hasattr(self, '_cache_modified') and self._cache_modified:
            self._save_batch_cache()