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
from hdbscan import HDBSCAN
import umap
from bertopic import BERTopic
import asyncio
import backoff
from openai import AsyncOpenAI
import hashlib
import pickle
import os
from pathlib import Path
import time

from config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, CHAT_MODEL,
    SIMILARITY_THRESHOLD, REPRESENTATIVE_QUESTION_METHOD, PROCESSING_MODE,
    SAMPLE_SIZE, MIN_CLUSTER_SIZE, UMAP_N_COMPONENTS, RANDOM_SEED,
    CACHE_EMBEDDINGS, CACHE_DIR
)

logger = logging.getLogger(__name__)

class HybridTopicProcessor:
    """Handles hybrid topic discovery and classification"""
    
    def __init__(self):
        """Initialize the hybrid processor"""
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Create cache directory if needed
        if CACHE_EMBEDDINGS:
            Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    
    def clean_question(self, question: str) -> str:
        """Remove ACM question prefix from questions before processing"""
        if not isinstance(question, str):
            return str(question) if question else ""
        
        # Define patterns to match ACM prefixes (case-insensitive)
        import re
        acm_patterns = [
            r'^\(ACM[s]?\s+[Qq]uestion\):?\s*',
            r'^\(ACM[s]?\s+[Qq]uestions?\):?\s*',
            r'^ACM[s]?\s+[Qq]uestion:?\s*',
            r'^ACM[s]?\s+[Qq]uestions?:?\s*'
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
    
    def get_cache_path(self, text: str, model: str) -> str:
        """Generate cache path for embedding"""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        model_clean = model.replace('/', '_')
        return os.path.join(CACHE_DIR, f"{model_clean}_{text_hash}.pkl")
    
    def load_cached_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Load embedding from cache"""
        if not CACHE_EMBEDDINGS:
            return None
            
        cache_path = self.get_cache_path(text, model)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        return None
    
    def save_embedding_to_cache(self, text: str, model: str, embedding: List[float]):
        """Save embedding to cache"""
        if not CACHE_EMBEDDINGS:
            return
            
        cache_path = self.get_cache_path(text, model)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding to cache: {e}")
    
    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APITimeoutError, openai.APIError),
        max_tries=3,
        base=2,
        max_value=60
    )
    def get_embedding(self, text: str, model: str = EMBEDDING_MODEL) -> List[float]:
        """Get embedding for text with caching and retry logic"""
        # Check cache first
        cached_embedding = self.load_cached_embedding(text, model)
        if cached_embedding is not None:
            return cached_embedding
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=model,
                dimensions=EMBEDDING_DIMENSIONS if "3-small" in model or "3-large" in model else None
            )
            embedding = response.data[0].embedding
            
            # Cache the result
            self.save_embedding_to_cache(text, model, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str], model: str = EMBEDDING_MODEL, batch_size: int = 100) -> List[List[float]]:
        """Get embeddings for multiple texts with optimized progress tracking"""
        embeddings = []
        
        # Create single progress bar for the entire process
        progress_bar = st.progress(0)
        
        total_texts = len(texts)
        processed = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Process each text in the batch
            for text in batch:
                embedding = self.get_embedding(text, model)
                embeddings.append(embedding)
                processed += 1
                
                # Update progress every 10 items or at end of batch
                if processed % 10 == 0 or processed == total_texts:
                    progress = processed / total_texts
                    progress_bar.progress(progress, text=f"Processing: {processed}/{total_texts}")
        
        progress_bar.empty()
        
        return embeddings
    
    def find_best_topic_match(self, question_embedding: List[float], topic_embeddings_df: pd.DataFrame) -> Optional[Dict]:
        """Find the best matching topic for a question using cosine similarity"""
        if len(topic_embeddings_df) == 0:
            return None
        
        # Calculate similarities with all topic questions
        similarities = []
        for _, row in topic_embeddings_df.iterrows():
            topic_embedding = row['embedding']
            similarity = cosine_similarity([question_embedding], [topic_embedding])[0][0]
            similarities.append({
                'topic': row['Topic'],
                'subtopic': row['Subtopic'],
                'question': row['Question'],
                'similarity': similarity
            })
        
        # Find best match
        best_match = max(similarities, key=lambda x: x['similarity'])
        
        # Return match if above threshold
        if best_match['similarity'] >= SIMILARITY_THRESHOLD:
            return best_match
        
        return None
    
    def classify_by_similarity(self, 
                               questions_df: pd.DataFrame, 
                               topic_questions_df: pd.DataFrame, 
                               threshold: float = SIMILARITY_THRESHOLD) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Classify questions by similarity to existing topics"""
        
        st.info("🔍 **Step 1: Similarity-Based Classification**")
        st.write("Matching questions to existing topics...")
        
        # Get embeddings for topic questions
        if 'embedding' not in topic_questions_df.columns:
            st.write("Processing topic embeddings...")
            topic_questions = topic_questions_df['Question'].tolist()
            topic_embeddings = self.get_embeddings_batch(topic_questions)
            topic_questions_df = topic_questions_df.copy()
            topic_questions_df['embedding'] = topic_embeddings
        
        # Get embeddings for new questions
        st.write("Processing question embeddings...")
        new_questions = questions_df['question'].tolist()
        question_embeddings = self.get_embeddings_batch(new_questions)
        
        # Classify questions
        similar_questions = []
        remaining_questions = []
        
        progress_bar = st.progress(0)
        
        for i, (_, row) in enumerate(questions_df.iterrows()):
            question = row['question']
            question_embedding = question_embeddings[i]
            
            # Find best match
            match = self.find_best_topic_match(question_embedding, topic_questions_df)
            
            if match:
                similar_questions.append({
                    'question': question,
                    'matched_topic': match['topic'],
                    'matched_subtopic': match['subtopic'],
                    'similarity_score': match['similarity']
                })
            else:
                remaining_questions.append({'question': question})
            
            # Update progress
            progress = (i + 1) / len(questions_df)
            progress_bar.progress(progress, text=f"Classifying: {i+1}/{len(questions_df)}")
        
        progress_bar.empty()
        
        similar_df = pd.DataFrame(similar_questions)
        remaining_df = pd.DataFrame(remaining_questions)
        
        st.success(f"✅ {len(similar_df)} matched, {len(remaining_df)} for clustering")
        
        return similar_df, remaining_df
    
    def perform_clustering_analysis(self, remaining_questions_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Perform clustering analysis on remaining questions"""
        
        if len(remaining_questions_df) == 0:
            return None
            
        st.info("🎯 **Step 2: Clustering-Based Topic Discovery**")
        st.write("Finding new topics in unmatched questions...")
        
        # Get embeddings for remaining questions
        questions = remaining_questions_df['question'].tolist()
        embeddings = self.get_embeddings_batch(questions)
        
        # Dimensionality reduction with UMAP
        st.write("Applying UMAP and HDBSCAN clustering...")
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
        clustered_df = remaining_questions_df.copy()
        clustered_df['cluster_id'] = cluster_labels
        clustered_df['umap_x'] = reduced_embeddings[:, 0]
        clustered_df['umap_y'] = reduced_embeddings[:, 1]
        
        # Filter out noise points (cluster_id == -1)
        valid_clusters_df = clustered_df[clustered_df['cluster_id'] != -1]
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = sum(1 for label in cluster_labels if label == -1)
        
        st.success(f"✅ Found {n_clusters} new topic clusters ({n_noise} noise points)")
        
        return valid_clusters_df if len(valid_clusters_df) > 0 else None
    
    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APITimeoutError, openai.APIError),
        max_tries=3,
        base=2,
        max_value=60
    )
    async def generate_topic_name(self, questions: List[str]) -> str:
        """Generate a topic name using GPT for a cluster of questions"""
        
        # Select representative questions (up to 5)
        sample_questions = questions[:5] if len(questions) > 5 else questions
        questions_text = "\n".join([f"- {q}" for q in sample_questions])
        
        prompt = f"""Analyze these student questions and generate a concise, descriptive topic name (2-4 words max).

Questions:
{questions_text}

Generate a topic name that captures the common theme. Response format: just the topic name, nothing else.

Examples of good topic names:
- "Technical Support"
- "Course Registration"  
- "Academic Policies"
- "Financial Aid"
- "Study Resources"

Topic name:"""
        
        try:
            response = self.client.chat.completions.create(
                model=TOPIC_GENERATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=20,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating topic name: {e}")
            return f"Topic Cluster"
    
    async def generate_topic_names_for_clusters(self, clustered_questions_df: pd.DataFrame) -> Dict[int, str]:
        """Generate topic names for all clusters"""
        
        st.write("🤖 Generating topic names with GPT...")
        
        topic_names = {}
        cluster_groups = clustered_questions_df.groupby('cluster_id')
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)
        
        async def generate_for_cluster(cluster_id, questions):
            async with semaphore:
                topic_name = await self.generate_topic_name(questions)
                return cluster_id, topic_name
        
        # Generate all topic names concurrently
        tasks = [
            generate_for_cluster(cluster_id, group['question'].tolist())
            for cluster_id, group in cluster_groups
        ]
        
        results = await asyncio.gather(*tasks)
        
        for cluster_id, topic_name in results:
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
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # File 1: Questions similar to existing topics
        file1_name = f"similar_questions_{timestamp}.csv"
        
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
        file2_name = f"new_topics_{timestamp}.csv"
        
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
        file3_name = f"pathway_questions_review_{timestamp}.csv"
        
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
    
    def prepare_evaluation_dataset(self, questions_df: pd.DataFrame, 
                                   mode: str = PROCESSING_MODE, 
                                   sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
        """Prepare evaluation dataset based on processing mode"""
        
        if mode == "sample" and len(questions_df) > sample_size:
            st.info(f"🎯 **Sample Mode**: Processing {sample_size} questions out of {len(questions_df)}")
            return questions_df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
        else:
            st.info(f"🎯 **Full Mode**: Processing all {len(questions_df)} questions")
            return questions_df
    
    async def process_hybrid_analysis(self, 
                                      questions_df: pd.DataFrame,
                                      topic_questions_df: pd.DataFrame,
                                      threshold: float = SIMILARITY_THRESHOLD) -> Dict[str, Any]:
        """Run complete hybrid topic analysis"""
        
        st.header("🚀 Hybrid Topic Analysis")
        
        # Prepare evaluation dataset
        eval_questions_df = self.prepare_evaluation_dataset(questions_df)
        
        # Step 1: Similarity-based classification
        similar_questions_df, remaining_questions_df = self.classify_by_similarity(
            eval_questions_df, topic_questions_df, threshold
        )
        
        # Step 2: Clustering-based topic discovery
        clustered_questions_df = self.perform_clustering_analysis(remaining_questions_df)
        
        # Step 3: Generate topic names
        topic_names = {}
        if clustered_questions_df is not None:
            st.info("🤖 **Generating topic names...**")
            topic_names = await self.generate_topic_names_for_clusters(clustered_questions_df)
        
        # Step 4: Select representative questions  
        representative_questions = {}
        if clustered_questions_df is not None:
            representative_questions = self.select_representative_questions(clustered_questions_df)
        
        # Step 5: Create output files
        st.info("📁 **Creating output files...**")
        file1, file2, file3 = self.create_output_files(
            similar_questions_df, clustered_questions_df, 
            topic_names, representative_questions
        )
        
        return {
            'similar_questions_df': similar_questions_df,
            'clustered_questions_df': clustered_questions_df,
            'topic_names': topic_names,
            'representative_questions': representative_questions,
            'output_files': [file1, file2, file3],
            'eval_questions_df': eval_questions_df
        }