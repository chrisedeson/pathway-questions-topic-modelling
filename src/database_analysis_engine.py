"""
Database-backed analysis engine that orchestrates the hybrid topic analysis
Handles background analysis runs and coordinates all components
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import logging
import asyncio
from pathlib import Path
import json

from hybrid_topic_processor import HybridTopicProcessor
# from sentiment_analyzer import SentimentAnalyzer, detect_topic_urgency
from trend_analyzer import TrendAnalyzer
from data_service import get_data_service
from sheets_sync import get_sheets_manager
from config import (
    SIMILARITY_THRESHOLD, MIN_CLUSTER_SIZE, UMAP_N_COMPONENTS,
    EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, CHAT_MODEL
)

logger = logging.getLogger(__name__)


class DatabaseAnalysisEngine:
    """
    Database-backed analysis engine that coordinates all analysis components.
    Handles background processing and database storage of results.
    """
    
    def __init__(self):
        """Initialize analysis engine with all components"""
        self.hybrid_processor = HybridTopicProcessor()
        # self.sentiment_analyzer = SentimentAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.data_service = get_data_service()
        self.sheets_manager = get_sheets_manager()
        
        self.current_run_id = None
        self.is_running = False
        self.progress_callback = None
        self.progress_state = {
            'step': 'idle',
            'progress': 0.0,
            'message': 'Ready'
        }
    
    def set_progress_callback(self, callback):
        """Set a callback function to report progress updates"""
        self.progress_callback = callback
    
    def _report_progress(self, step: str, progress: float, message: str = ""):
        """Report progress to callback and internal state"""
        # Update internal state (always works)
        self.progress_state = {
            'step': step,
            'progress': progress,
            'message': message
        }
        
        # Try callback (may fail in threads, but that's ok)
        if self.progress_callback:
            try:
                self.progress_callback(step, progress, message)
            except Exception as e:
                # Ignore Streamlit context errors in background threads
                logger.debug(f"Progress callback failed (expected in background thread): {e}")
    
    def get_current_progress(self):
        """Get current progress state (thread-safe)"""
        return self.progress_state.copy()
    
    def run_full_analysis(
        self,
        force_refresh: bool = False,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Run complete analysis pipeline in background
        
        Args:
            force_refresh: If True, bypass cache and recompute everything
            filters: Optional filters for question selection
            
        Returns:
            run_id for tracking the analysis
        """
        if self.is_running:
            raise RuntimeError("Analysis is already running")
        
        self.is_running = True
        
        try:
            # Create analysis run record
            config = {
                'similarity_threshold': SIMILARITY_THRESHOLD,
                'min_cluster_size': MIN_CLUSTER_SIZE,
                'umap_n_components': UMAP_N_COMPONENTS,
                'embedding_model': EMBEDDING_MODEL,
                'embedding_dimensions': EMBEDDING_DIMENSIONS,
                'chat_model': CHAT_MODEL,
                'force_refresh': force_refresh,
                'filters': filters
            }
            
            self.current_run_id = self.data_service.create_analysis_run(config)
            
            # Run analysis in separate thread for Streamlit compatibility
            import threading
            thread = threading.Thread(
                target=self._run_analysis_sync,
                args=(force_refresh, filters),
                daemon=True
            )
            thread.start()
            
            return self.current_run_id
            
        except Exception as e:
            self.is_running = False
            logger.error(f"Failed to start analysis: {e}")
            raise
    
    def _run_analysis_sync(self, force_refresh: bool = False, filters: Optional[Dict[str, Any]] = None):
        """Synchronous wrapper to run async analysis"""
        try:
            # Create new event loop for this thread
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._run_analysis_async())
            finally:
                loop.close()
        except Exception as e:
            self.is_running = False
            logger.error(f"Analysis failed: {e}")
            # Update analysis run status
            if self.current_run_id:
                self.data_service.fail_analysis_run(
                    self.current_run_id, str(e)
                )
    
    async def _run_analysis_async(self):
        """Run the actual analysis asynchronously - Fresh data flow"""
        try:
            self._report_progress("initialization", 0.0, "Starting fresh analysis...")
            
            # Step 1: Clear all existing data
            self._report_progress("clearing_data", 5.0, "Clearing existing database data...")
            db_manager = self.data_service.db_manager
            if not db_manager.clear_all_data():
                raise RuntimeError("Failed to clear existing database data")
            
            # Step 2: Load fresh questions data from Google Sheets
            self._report_progress("loading_data", 10.0, "Loading questions from Google Sheets...")
            questions_df = await self._load_questions_from_sheets()
            
            if questions_df.empty:
                raise ValueError("No questions found in Google Sheets")
            
            self._report_progress("loading_data", 20.0, f"Loaded {len(questions_df)} questions from sheets")
            
            # Step 3: Store fresh questions in database
            self._report_progress("storing_questions", 22.0, "Storing questions in database...")
            added, updated, skipped = self.data_service.store_questions(questions_df, "analysis_run")
            self._report_progress("storing_questions", 25.0, f"Stored {added} questions in database")
            
            # Step 4: Load topics data from Google Sheets
            self._report_progress("loading_topics", 28.0, "Loading topics from Google Sheets...")
            topics_df = await self._load_topics_data()
            
            self._report_progress("loading_topics", 30.0, f"Loaded {len(topics_df)} topic references")
            
            # Step 5: Run hybrid topic processing
            self._report_progress("embeddings", 35.0, "Preparing data for embeddings...")
            
            # Prepare DataFrame for hybrid processor (needs 'question' column)
            processing_df = questions_df.copy()
            if 'cleaned_question' in processing_df.columns and 'question' not in processing_df.columns:
                processing_df['question'] = processing_df['cleaned_question']
            
            self._report_progress("embeddings", 40.0, f"Generating embeddings for {len(processing_df)} questions...")
            hybrid_results = await self.hybrid_processor.process_hybrid_analysis(
                questions_df=processing_df,
                topic_questions_df=topics_df
            )
            
            self._report_progress("clustering", 55.0, "Clustering and topic discovery completed")
            
            # Step 6: Store embeddings
            self._report_progress("storing_embeddings", 60.0, "Storing embeddings in database...")
            await self._store_embeddings(processing_df, hybrid_results)
            
            # Step 7: Process and store clusters
            self._report_progress("storing_clusters", 70.0, "Processing topic clusters...")
            cluster_id_map = await self._store_clusters(hybrid_results)
            
            # Step 8: Store question-cluster assignments
            self._report_progress("storing_assignments", 80.0, "Storing question assignments...")
            await self._store_assignments(processing_df, hybrid_results, cluster_id_map)
            
            # Step 9: Run trend analysis
            self._report_progress("trends", 85.0, "Analyzing trends...")
            trend_results = self._analyze_trends(processing_df, hybrid_results)
            
            # Step 10: Cache dashboard data (skip sentiment for now)
            self._report_progress("caching", 90.0, "Caching analysis results...")
            sentiment_results = {
                'status': 'skipped',
                'cluster_sentiments': [],  # Empty list to match expected schema
                'individual_sentiments': []
            }
            await self._cache_dashboard_data(questions_df, hybrid_results, sentiment_results, trend_results)
            
            # Step 11: Finalize
            self._report_progress("finalizing", 95.0, "Finalizing analysis...")
            
            # Update analysis run as completed
            total_clusters = len(hybrid_results.get('clusters', []))
            self.data_service.complete_analysis_run(
                self.current_run_id, 
                len(processing_df), 
                total_clusters
            )
            
            self._report_progress("completed", 100.0, f"Analysis completed successfully! 🎉 Found {total_clusters} topics")
            
            logger.info(f"Analysis run {self.current_run_id} completed successfully")
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg)
            
            if self.current_run_id:
                self.data_service.fail_analysis_run(self.current_run_id, error_msg)
            
            self._report_progress("failed", 0.0, error_msg)
            
        finally:
            self.is_running = False
            self.current_run_id = None
    
    async def _load_questions_from_sheets(self) -> pd.DataFrame:
        """Load questions data directly from Google Sheets"""
        try:
            from config import QUESTIONS_SHEET_ID
            
            # Fetch fresh data from Google Sheets
            questions_df = self.sheets_manager.sync.fetch_questions_data(QUESTIONS_SHEET_ID)
            
            if questions_df.empty:
                logger.warning("No questions data found in Google Sheets")
                return pd.DataFrame()
            
            # Clean the questions data
            cleaned_df = self.data_service.cleaner.clean_questions_dataframe(questions_df)
            logger.info(f"Cleaned {len(cleaned_df)} questions from Google Sheets")
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Failed to load questions from Google Sheets: {e}")
            return pd.DataFrame()
    
    async def _load_topics_data(self) -> pd.DataFrame:
        """Load topics data from Google Sheets"""
        try:
            # Get topics sheet ID from configuration
            from config import TOPICS_SHEET_ID
            
            topics_df = self.sheets_manager.sync.fetch_topics_data(TOPICS_SHEET_ID)
            
            if topics_df.empty:
                logger.warning("No topics data found, creating empty DataFrame")
                return pd.DataFrame(columns=['Topic', 'Subtopic', 'Question'])
            
            return topics_df
            
        except Exception as e:
            logger.warning(f"Failed to load topics data: {e}")
            return pd.DataFrame(columns=['Topic', 'Subtopic', 'Question'])
    
    async def _store_embeddings(self, questions_df: pd.DataFrame, hybrid_results: Dict[str, Any]):
        """Store question embeddings in database"""
        embeddings = hybrid_results.get('embeddings', [])
        
        if not embeddings:
            logger.warning("No embeddings found in hybrid results")
            return
        
        # Create question_id to embedding mapping
        question_embeddings = []
        for i, (_, row) in enumerate(questions_df.iterrows()):
            if i < len(embeddings):
                question_embeddings.append((row['id'], embeddings[i]))
        
        self.data_service.store_embeddings(self.current_run_id, question_embeddings)
    
    async def _store_clusters(self, hybrid_results: Dict[str, Any]) -> Dict[int, int]:
        """Store topic clusters and return mapping"""
        clusters = hybrid_results.get('clusters', [])
        
        clusters_data = []
        for cluster in clusters:
            clusters_data.append({
                'cluster_id': cluster.get('cluster_id', -1),
                'topic_name': cluster.get('topic_name', 'Unknown Topic'),
                'topic_description': cluster.get('description'),
                'representative_question': cluster.get('representative_question', ''),
                'question_count': len(cluster.get('questions', [])),
                'avg_sentiment': cluster.get('avg_sentiment'),
                'urgency_score': cluster.get('urgency_score', 0.0),
                'keywords': cluster.get('keywords', [])
            })
        
        return self.data_service.store_topic_clusters(self.current_run_id, clusters_data)
    
    async def _store_assignments(self, questions_df: pd.DataFrame, 
                               hybrid_results: Dict[str, Any], 
                               cluster_id_map: Dict[int, int]):
        """Store question-to-cluster assignments"""
        assignments = []
        
        # Get cluster assignments from hybrid results
        question_clusters = hybrid_results.get('question_clusters', [])
        
        for i, (_, row) in enumerate(questions_df.iterrows()):
            if i < len(question_clusters):
                cluster_id = question_clusters[i]
                
                if cluster_id in cluster_id_map:
                    assignments.append({
                        'question_id': row['id'],
                        'cluster_db_id': cluster_id_map[cluster_id],
                        'similarity_score': None,  # Could be computed if needed
                        'is_representative': False  # Could be determined from hybrid results
                    })
        
        self.data_service.store_question_assignments(self.current_run_id, assignments)
    
    def _analyze_sentiment(self, questions_df: pd.DataFrame, 
                          hybrid_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run sentiment analysis on questions and clusters"""
        try:
            # Analyze individual questions
            questions = questions_df['question'].tolist() if 'question' in questions_df.columns else questions_df['cleaned_question'].tolist()
            
            sentiments = []
            for question in questions:
                # sentiment = self.sentiment_analyzer.analyze_sentiment(question)
                sentiment = {'sentiment': 'neutral', 'confidence': 0.5}  # Default placeholder
                sentiments.append(sentiment)
            
            # Aggregate by clusters
            clusters = hybrid_results.get('clusters', [])
            cluster_sentiments = []
            
            for cluster in clusters:
                cluster_questions = cluster.get('questions', [])
                if cluster_questions:
                    # cluster_sentiment_scores = [
                    #     self.sentiment_analyzer.analyze_sentiment(q) for q in cluster_questions
                    # ]
                    cluster_sentiment_scores = [{'compound': 0.0}] * len(cluster_questions)  # Default placeholder
                    avg_sentiment = np.mean([s['compound'] for s in cluster_sentiment_scores])
                    # urgency = detect_topic_urgency(cluster_questions, cluster_sentiment_scores)
                    urgency = 'medium'  # Default placeholder
                    
                    cluster_sentiments.append({
                        'cluster_id': cluster.get('cluster_id'),
                        'avg_sentiment': avg_sentiment,
                        'urgency_score': urgency
                    })
            
            return {
                'individual_sentiments': sentiments,
                'cluster_sentiments': cluster_sentiments
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_trends(self, questions_df: pd.DataFrame, 
                       hybrid_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run trend analysis on questions over time"""
        try:
            # Prepare data with topic assignments for trend analysis
            analysis_df = questions_df.copy()
            
            # Add topic information from hybrid results if available
            similar_questions = hybrid_results.get('similar_questions_df')
            clustered_questions = hybrid_results.get('clustered_questions_df')
            topic_names = hybrid_results.get('topic_names', {})
            
            # Create a topic mapping for questions
            question_to_topic = {}
            
            # Map similar questions to their matched topics
            if similar_questions is not None and len(similar_questions) > 0:
                for _, row in similar_questions.iterrows():
                    question_to_topic[row['question']] = row.get('matched_topic', 'Unknown')
            
            # Map clustered questions to their generated topic names
            if clustered_questions is not None and len(clustered_questions) > 0:
                for _, row in clustered_questions.iterrows():
                    cluster_id = row.get('cluster_id', -1)
                    topic_name = topic_names.get(cluster_id, f"Cluster_{cluster_id}")
                    question_to_topic[row['question']] = topic_name
            
            # Add topic column to analysis dataframe
            analysis_df['topic'] = analysis_df['question'].map(question_to_topic).fillna('Uncategorized')
            
            # Use the correct method name from TrendAnalyzer
            trends = self.trend_analyzer.analyze_temporal_trends(
                analysis_df, 
                time_column='timestamp'
            )
            
            # Also get trend summary
            summary = self.trend_analyzer.generate_trend_summary(analysis_df)
            
            # Combine results
            result = {
                'temporal_trends': trends,
                'summary': summary
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {'error': str(e)}
    
    def _make_json_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format"""
        if isinstance(data, dict):
            return {key: self._make_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        elif isinstance(data, (np.integer, np.int64, np.int32)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64, np.float32)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, pd.Timestamp):
            return data.isoformat()
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            return data.to_dict()
        elif hasattr(data, 'isoformat'):  # datetime objects
            return data.isoformat()
        else:
            return data
    
    async def _cache_dashboard_data(self, questions_df: pd.DataFrame,
                                  hybrid_results: Dict[str, Any],
                                  sentiment_results: Dict[str, Any],
                                  trend_results: Dict[str, Any]):
        """Cache dashboard data for different time periods"""
        
        time_periods = ['all', '30d', '7d', '24h']
        
        for period in time_periods:
            try:
                # Filter data for time period
                filtered_df = self._filter_by_time_period(questions_df, period)
                
                # Compile dashboard data
                dashboard_data = {
                    'run_id': self.current_run_id,
                    'time_period': period,
                    'last_updated': datetime.now(timezone.utc).isoformat(),
                    'metrics': {
                        'total_questions': len(filtered_df),
                        'total_topics': len(hybrid_results.get('clusters', [])),
                        'date_range': {
                            'start': filtered_df['timestamp'].min().isoformat() if not filtered_df.empty else None,
                            'end': filtered_df['timestamp'].max().isoformat() if not filtered_df.empty else None
                        }
                    },
                    'language_distribution': self._get_language_distribution(filtered_df),
                    'topic_distribution': self._get_topic_distribution(hybrid_results),
                    'sentiment_summary': self._summarize_sentiment(sentiment_results),
                    'trend_summary': self._summarize_trends(trend_results),
                    'clusters': hybrid_results.get('clusters', [])
                }
                
                # Convert to JSON-serializable format
                dashboard_data = self._make_json_serializable(dashboard_data)
                
                # Cache the data
                cache_key = f"dashboard_{period}_{self.current_run_id}"
                self.data_service.cache_analysis_result(
                    run_id=self.current_run_id,
                    cache_type='dashboard',
                    cache_key=cache_key,
                    data=dashboard_data,
                    expires_hours=24  # Cache for 24 hours
                )
                
            except Exception as e:
                logger.error(f"Failed to cache dashboard data for {period}: {e}")
    
    def _filter_by_time_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Filter DataFrame by time period"""
        if period == 'all' or df.empty:
            return df
        
        now = datetime.now(timezone.utc)
        
        if period == '24h':
            cutoff = now - pd.Timedelta(hours=24)
        elif period == '7d':
            cutoff = now - pd.Timedelta(days=7)
        elif period == '30d':
            cutoff = now - pd.Timedelta(days=30)
        else:
            return df
        
        # Ensure timestamp is timezone-aware
        df_copy = df.copy()
        if 'timestamp' in df_copy.columns:
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], utc=True)
            return df_copy[df_copy['timestamp'] >= cutoff]
        
        return df_copy
    
    def _get_language_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get language distribution from DataFrame"""
        if df.empty or 'user_language' not in df.columns:
            return {}
        
        return df['user_language'].value_counts().to_dict()
    
    def _get_topic_distribution(self, hybrid_results: Dict[str, Any]) -> Dict[str, int]:
        """Get topic distribution from hybrid results"""
        clusters = hybrid_results.get('clusters', [])
        
        topic_dist = {}
        for cluster in clusters:
            topic_name = cluster.get('topic_name', 'Unknown')
            question_count = len(cluster.get('questions', []))
            topic_dist[topic_name] = question_count
        
        return topic_dist
    
    def _summarize_sentiment(self, sentiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize sentiment analysis results"""
        if 'error' in sentiment_results:
            logger.warning(f"Sentiment analysis degraded: {sentiment_results['error']}")
            return {'error': sentiment_results['error']}
        
        if sentiment_results.get('status') == 'skipped':
            logger.info("Sentiment analysis skipped - using neutral defaults")
            return {'avg_sentiment': 0.0, 'urgency_topics': [], 'status': 'skipped'}
        
        cluster_sentiments = sentiment_results.get('cluster_sentiments', [])
        
        if not cluster_sentiments:
            logger.warning("No cluster sentiment data available - using defaults")
            return {'avg_sentiment': 0.0, 'urgency_topics': []}
        
        avg_sentiment = np.mean([cs['avg_sentiment'] for cs in cluster_sentiments])
        urgent_topics = [cs for cs in cluster_sentiments if cs['urgency_score'] > 0.7]
        
        return {
            'avg_sentiment': float(avg_sentiment),
            'urgent_topics_count': len(urgent_topics),
            'urgency_topics': urgent_topics[:5]  # Top 5 urgent topics
        }
    
    def _summarize_trends(self, trend_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize trend analysis results"""
        if 'error' in trend_results:
            logger.warning(f"Trend analysis degraded: {trend_results['error']}")
            return {'error': trend_results['error']}
        
        # Extract key trend metrics
        temporal_trends = trend_results.get('temporal_trends', {})
        summary = trend_results.get('summary', {})
        
        if not temporal_trends and not summary:
            logger.warning("No trend data available - using empty defaults")
        
        return {
            'trending_up': temporal_trends.get('trending_up', [])[:5],
            'trending_down': temporal_trends.get('trending_down', [])[:5],
            'seasonal_patterns': temporal_trends.get('seasonal_patterns', {}),
            'summary': summary
        }
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """Get current analysis status"""
        return {
            'is_running': self.is_running,
            'current_run_id': self.current_run_id,
            'database_status': self.data_service.get_analysis_status()
        }


# Global analysis engine instance
_analysis_engine = None

def get_analysis_engine() -> DatabaseAnalysisEngine:
    """Get global analysis engine instance"""
    global _analysis_engine
    if _analysis_engine is None:
        _analysis_engine = DatabaseAnalysisEngine()
    return _analysis_engine