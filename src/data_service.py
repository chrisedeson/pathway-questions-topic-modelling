"""
Database service layer for managing questions, analysis runs, and cached results
"""
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_

from database import (
    get_db_manager, Question, AnalysisRun, QuestionEmbedding, 
    TopicCluster, QuestionClusterAssignment, AnalysisCache, SyncLog
)
from data_cleaning import QuestionCleaner

logger = logging.getLogger(__name__)

class DataService:
    """Service layer for database operations"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        self.cleaner = QuestionCleaner()
    
    def store_questions(self, df: pd.DataFrame, source: str = "manual_upload") -> Tuple[int, int, int]:
        """
        Store questions in database, handling duplicates
        
        Args:
            df: DataFrame with cleaned questions
            source: Source description for logging
            
        Returns:
            Tuple of (added_count, updated_count, skipped_count)
        """
        added_count = 0
        updated_count = 0
        skipped_count = 0
        
        # Start sync log
        sync_log = SyncLog(
            sync_type='questions',
            source=source,
            started_at=datetime.now(timezone.utc),
            status='running'
        )
        
        try:
            with self.db_manager.get_session() as session:
                session.add(sync_log)
                session.flush()  # Get the ID
                
                # Optimized bulk approach - much faster
                logger.info(f"Processing {len(df)} questions in bulk...")
                
                # Get all existing questions in one query for comparison
                existing_questions = {}
                if len(df) > 0:
                    # Create a set of (cleaned_question, timestamp) pairs to check
                    question_pairs = [(row['cleaned_question'], row['timestamp']) for _, row in df.iterrows()]
                    
                    # Query existing questions in batches to avoid overwhelming the database
                    batch_size = 1000
                    for i in range(0, len(question_pairs), batch_size):
                        batch = question_pairs[i:i + batch_size]
                        
                        # Create OR conditions for this batch
                        conditions = []
                        for cleaned_q, ts in batch:
                            conditions.append(
                                and_(
                                    Question.cleaned_question == cleaned_q,
                                    Question.timestamp == ts
                                )
                            )
                        
                        if conditions:
                            existing_batch = session.query(Question).filter(or_(*conditions)).all()
                            for q in existing_batch:
                                existing_questions[(q.cleaned_question, q.timestamp)] = q
                
                # Process questions in batches
                batch_size = 500
                for i in range(0, len(df), batch_size):
                    batch_df = df.iloc[i:i + batch_size]
                    new_questions = []
                    
                    for _, row in batch_df.iterrows():
                        try:
                            key = (row['cleaned_question'], row['timestamp'])
                            existing = existing_questions.get(key)
                            
                            if existing:
                                # Update if any fields are different
                                updated = False
                                for field in ['user_language', 'country', 'state', 'user_role']:
                                    if field in row and pd.notna(row[field]):
                                        new_value = row[field]
                                        if getattr(existing, field) != new_value:
                                            setattr(existing, field, new_value)
                                            updated = True
                                
                                if updated:
                                    existing.updated_at = datetime.now(timezone.utc)
                                    updated_count += 1
                                else:
                                    skipped_count += 1
                            else:
                                # Create new question
                                question = Question(
                                    original_text=row.get('original_text', ''),
                                    cleaned_question=row['cleaned_question'],
                                    timestamp=row['timestamp'],
                                    user_language=row.get('user_language', 'en'),
                                    country=row.get('country'),
                                    state=row.get('state'),
                                    user_role=row.get('user_role')
                                )
                                new_questions.append(question)
                                added_count += 1
                        
                        except Exception as e:
                            logger.error(f"Error processing question row: {e}")
                            skipped_count += 1
                            continue
                    
                    # Add new questions in bulk
                    if new_questions:
                        session.add_all(new_questions)
                    
                    # Commit batch
                    session.commit()
                    logger.info(f"Processed batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
                
                # Update sync log
                sync_log.completed_at = datetime.now(timezone.utc)
                sync_log.status = 'completed'
                sync_log.records_processed = len(df)
                sync_log.records_added = added_count
                sync_log.records_updated = updated_count
                sync_log.records_skipped = skipped_count
                
                session.commit()
                
                logger.info(f"Stored questions: {added_count} added, {updated_count} updated, {skipped_count} skipped")
                
        except Exception as e:
            logger.error(f"Failed to store questions: {e}")
            sync_log.status = 'failed'
            sync_log.error_message = str(e)
            with self.db_manager.get_session() as session:
                session.merge(sync_log)
                session.commit()
            raise
        
        return added_count, updated_count, skipped_count
    
    def create_analysis_run(self, config: Dict[str, Any]) -> str:
        """
        Create a new analysis run record
        
        Args:
            config: Configuration snapshot for this run
            
        Returns:
            run_id for the new analysis run
        """
        run_id = str(uuid.uuid4())
        
        with self.db_manager.get_session() as session:
            analysis_run = AnalysisRun(
                run_id=run_id,
                started_at=datetime.now(timezone.utc),
                status='running',
                config_snapshot=config
            )
            session.add(analysis_run)
            session.commit()
        
        logger.info(f"Created analysis run: {run_id}")
        return run_id
    
    def complete_analysis_run(self, run_id: str, total_questions: int, total_topics: int):
        """Mark analysis run as completed"""
        with self.db_manager.get_session() as session:
            run = session.query(AnalysisRun).filter(AnalysisRun.run_id == run_id).first()
            if run:
                run.completed_at = datetime.now(timezone.utc)
                run.status = 'completed'
                run.total_questions = total_questions
                run.total_topics = total_topics
                session.commit()
                logger.info(f"Analysis run {run_id} completed")
    
    def fail_analysis_run(self, run_id: str, error_message: str):
        """Mark analysis run as failed"""
        with self.db_manager.get_session() as session:
            run = session.query(AnalysisRun).filter(AnalysisRun.run_id == run_id).first()
            if run:
                run.status = 'failed'
                run.error_message = error_message
                session.commit()
                logger.error(f"Analysis run {run_id} failed: {error_message}")
    
    def store_embeddings(self, run_id: str, question_embeddings: List[Tuple[int, np.ndarray]]):
        """
        Store question embeddings for an analysis run
        
        Args:
            run_id: Analysis run ID
            question_embeddings: List of (question_id, embedding_vector) tuples
        """
        with self.db_manager.get_session() as session:
            embeddings = []
            for question_id, embedding in question_embeddings:
                emb = QuestionEmbedding(
                    question_id=question_id,
                    run_id=run_id,
                    embedding=embedding.tolist(),  # Convert numpy array to list
                    model_version='text-embedding-3-small'
                )
                embeddings.append(emb)
            
            session.add_all(embeddings)
            session.commit()
            
            logger.info(f"Stored {len(embeddings)} embeddings for run {run_id}")
    
    def store_topic_clusters(self, run_id: str, clusters_data: List[Dict[str, Any]]) -> Dict[int, int]:
        """
        Store topic clusters and return mapping of cluster_id to database id
        
        Args:
            run_id: Analysis run ID
            clusters_data: List of cluster dictionaries
            
        Returns:
            Mapping of cluster_id to database record id
        """
        cluster_id_map = {}
        
        with self.db_manager.get_session() as session:
            for cluster_data in clusters_data:
                topic = TopicCluster(
                    run_id=run_id,
                    cluster_id=cluster_data['cluster_id'],
                    topic_name=cluster_data['topic_name'],
                    topic_description=cluster_data.get('topic_description'),
                    representative_question=cluster_data['representative_question'],
                    question_count=cluster_data.get('question_count', 0),
                    avg_sentiment=cluster_data.get('avg_sentiment'),
                    urgency_score=cluster_data.get('urgency_score', 0.0),
                    keywords=cluster_data.get('keywords', [])
                )
                session.add(topic)
                session.flush()  # Get the database ID
                cluster_id_map[cluster_data['cluster_id']] = topic.id
            
            session.commit()
            
            logger.info(f"Stored {len(clusters_data)} topic clusters for run {run_id}")
        
        return cluster_id_map
    
    def store_question_assignments(self, run_id: str, assignments: List[Dict[str, Any]]):
        """Store question-to-cluster assignments"""
        with self.db_manager.get_session() as session:
            assignment_records = []
            for assignment in assignments:
                record = QuestionClusterAssignment(
                    question_id=assignment['question_id'],
                    cluster_id=assignment['cluster_db_id'],  # Database ID, not cluster_id
                    run_id=run_id,
                    similarity_score=assignment.get('similarity_score'),
                    is_representative=assignment.get('is_representative', False)
                )
                assignment_records.append(record)
            
            session.add_all(assignment_records)
            session.commit()
            
            logger.info(f"Stored {len(assignment_records)} question assignments for run {run_id}")
    
    def cache_analysis_result(self, run_id: str, cache_type: str, cache_key: str, 
                            data: Dict[str, Any], expires_hours: Optional[int] = None):
        """Cache analysis results for fast retrieval"""
        expires_at = None
        if expires_hours:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_hours)
        
        with self.db_manager.get_session() as session:
            # Remove existing cache entry if it exists
            existing = session.query(AnalysisCache).filter(
                AnalysisCache.cache_key == cache_key
            ).first()
            
            if existing:
                session.delete(existing)
            
            # Create new cache entry
            cache_entry = AnalysisCache(
                cache_key=cache_key,
                run_id=run_id,
                cache_type=cache_type,
                data=data,
                expires_at=expires_at
            )
            session.add(cache_entry)
            session.commit()
    
    def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached analysis result"""
        with self.db_manager.get_session() as session:
            cache_entry = session.query(AnalysisCache).filter(
                and_(
                    AnalysisCache.cache_key == cache_key,
                    or_(
                        AnalysisCache.expires_at.is_(None),
                        AnalysisCache.expires_at > datetime.now(timezone.utc)
                    )
                )
            ).first()
            
            if cache_entry:
                return cache_entry.data
            return None
    
    def get_questions_for_analysis(self, run_id: Optional[str] = None, 
                                  filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Get questions for analysis with optional filtering
        
        Args:
            run_id: Specific run ID to get questions for (if None, gets all)
            filters: Optional filters (date_range, languages, etc.)
            
        Returns:
            DataFrame with questions
        """
        with self.db_manager.get_session() as session:
            query = session.query(Question)
            
            # Apply filters
            if filters:
                if 'start_date' in filters:
                    query = query.filter(Question.timestamp >= filters['start_date'])
                if 'end_date' in filters:
                    query = query.filter(Question.timestamp <= filters['end_date'])
                if 'languages' in filters:
                    query = query.filter(Question.user_language.in_(filters['languages']))
                if 'countries' in filters:
                    query = query.filter(Question.country.in_(filters['countries']))
            
            # Execute query and convert to DataFrame
            questions = query.order_by(Question.timestamp).all()
            
            data = []
            for q in questions:
                data.append({
                    'id': q.id,
                    'question': q.cleaned_question,
                    'original_text': q.original_text,
                    'timestamp': q.timestamp,
                    'user_language': q.user_language,
                    'country': q.country,
                    'state': q.state,
                    'user_role': q.user_role
                })
            
            return pd.DataFrame(data)
    
    def get_dashboard_data(self, time_period: str = 'all') -> Dict[str, Any]:
        """
        Get pre-computed dashboard data for the latest analysis run
        
        Args:
            time_period: Time period filter ('24h', '7d', '30d', 'all')
            
        Returns:
            Dictionary with dashboard data
        """
        # Get latest completed run
        latest_run_id = self.db_manager.get_latest_analysis_run()
        if not latest_run_id:
            return {'error': 'No completed analysis runs found'}
        
        # Build cache key
        cache_key = f"dashboard_{time_period}_{latest_run_id}"
        
        # Try to get from cache first
        cached_result = self.get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # If not cached, compute on demand (this should be rare)
        logger.warning(f"Dashboard data not cached for {cache_key}, computing on demand")
        return self._compute_dashboard_data(latest_run_id, time_period, cache_key)
    
    def _compute_dashboard_data(self, run_id: str, time_period: str, cache_key: str) -> Dict[str, Any]:
        """Compute dashboard data and cache it"""
        with self.db_manager.get_session() as session:
            # Get time filter
            time_filter = self._get_time_filter(time_period)
            
            # Base queries
            questions_query = session.query(Question)
            if time_filter:
                questions_query = questions_query.filter(Question.timestamp >= time_filter)
            
            # Get basic metrics
            total_questions = questions_query.count()
            
            # Language distribution
            lang_dist = session.query(
                Question.user_language,
                func.count(Question.id).label('count')
            ).filter(
                Question.timestamp >= time_filter if time_filter else True
            ).group_by(Question.user_language).all()
            
            # Topic distribution from latest run
            topic_dist = session.query(
                TopicCluster.topic_name,
                TopicCluster.question_count
            ).filter(TopicCluster.run_id == run_id).all()
            
            # Compile dashboard data
            dashboard_data = {
                'run_id': run_id,
                'time_period': time_period,
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'metrics': {
                    'total_questions': total_questions,
                    'total_topics': len(topic_dist),
                },
                'language_distribution': {lang: count for lang, count in lang_dist},
                'topic_distribution': {topic: count for topic, count in topic_dist},
                'filters_applied': {
                    'time_period': time_period,
                    'date_range': {
                        'start': time_filter.isoformat() if time_filter else None,
                        'end': datetime.now(timezone.utc).isoformat()
                    }
                }
            }
            
            # Cache the result
            self.cache_analysis_result(
                run_id=run_id,
                cache_type='dashboard',
                cache_key=cache_key,
                data=dashboard_data,
                expires_hours=1  # Cache for 1 hour
            )
            
            return dashboard_data
    
    def _get_time_filter(self, time_period: str) -> Optional[datetime]:
        """Get datetime filter based on time period"""
        now = datetime.now(timezone.utc)
        
        if time_period == '24h':
            return now - timedelta(hours=24)
        elif time_period == '7d':
            return now - timedelta(days=7)
        elif time_period == '30d':
            return now - timedelta(days=30)
        elif time_period == '90d':
            return now - timedelta(days=90)
        else:  # 'all'
            return None
    
    def get_trend_data(self, time_period: str = '30d', granularity: str = 'daily') -> Dict[str, Any]:
        """Get trend analysis data"""
        latest_run_id = self.db_manager.get_latest_analysis_run()
        if not latest_run_id:
            return {'error': 'No completed analysis runs found'}
        
        cache_key = f"trends_{time_period}_{granularity}_{latest_run_id}"
        cached_result = self.get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Compute trends (implementation would go here)
        return {'message': 'Trend computation not yet implemented'}
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """Get current analysis status"""
        with self.db_manager.get_session() as session:
            # Get latest run
            latest_run = session.query(AnalysisRun).order_by(
                desc(AnalysisRun.started_at)
            ).first()
            
            if not latest_run:
                return {'status': 'no_runs'}
            
            # Get sync status
            sync_status = self.db_manager.get_sync_status()
            
            return {
                'latest_run': {
                    'run_id': latest_run.run_id,
                    'status': latest_run.status,
                    'started_at': latest_run.started_at.isoformat(),
                    'completed_at': latest_run.completed_at.isoformat() if latest_run.completed_at else None,
                    'total_questions': latest_run.total_questions,
                    'total_topics': latest_run.total_topics,
                    'error_message': latest_run.error_message
                },
                'sync_status': sync_status,
                'database_stats': {
                    'total_questions': self.db_manager.get_questions_count(),
                    'questions_in_latest_run': self.db_manager.get_questions_count(latest_run.run_id)
                }
            }


# Global service instance
_data_service = None

def get_data_service() -> DataService:
    """Get global data service instance"""
    global _data_service
    if _data_service is None:
        _data_service = DataService()
    return _data_service