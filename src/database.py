"""
Database models and connection management for BYU Pathway Questions Analysis
"""
import os
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, 
    Float, Boolean, JSON, Index, func, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from pgvector.sqlalchemy import Vector
import pandas as pd

logger = logging.getLogger(__name__)

Base = declarative_base()

class Question(Base):
    """Store raw and processed questions"""
    __tablename__ = 'questions'
    
    id = Column(Integer, primary_key=True)
    original_text = Column(Text, nullable=False)  # Raw kwargs text if applicable
    cleaned_question = Column(Text, nullable=False)  # Extracted clean question
    timestamp = Column(DateTime(timezone=True), nullable=False)
    user_language = Column(String(10), default='en')
    country = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    user_role = Column(String(50), nullable=True)  # missionary, student, etc.
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_questions_timestamp', 'timestamp'),
        Index('idx_questions_language', 'user_language'),
        Index('idx_questions_created', 'created_at'),
        Index('idx_questions_country_state', 'country', 'state'),
    )

class AnalysisRun(Base):
    """Track analysis runs and their metadata"""
    __tablename__ = 'analysis_runs'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(String(100), unique=True, nullable=False)  # UUID for this run
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(50), default='running')  # running, completed, failed
    total_questions = Column(Integer, nullable=True)
    total_topics = Column(Integer, nullable=True)
    config_snapshot = Column(JSON, nullable=True)  # Settings used for this run
    error_message = Column(Text, nullable=True)
    created_by = Column(String(100), default='system')
    
    # Index for finding latest runs
    __table_args__ = (
        Index('idx_analysis_runs_status_started', 'status', 'started_at'),
        Index('idx_analysis_runs_completed', 'completed_at'),
    )

class QuestionEmbedding(Base):
    """Store question embeddings for similarity analysis"""
    __tablename__ = 'question_embeddings'
    
    id = Column(Integer, primary_key=True)
    question_id = Column(Integer, nullable=False)  # FK to questions.id
    run_id = Column(String(100), nullable=False)  # FK to analysis_runs.run_id
    embedding = Column(Vector(1536), nullable=False)  # OpenAI text-embedding-3-small
    model_version = Column(String(100), default='text-embedding-3-small')
    created_at = Column(DateTime(timezone=True), default=func.now())
    
    __table_args__ = (
        Index('idx_embeddings_question_run', 'question_id', 'run_id'),
        Index('idx_embeddings_run', 'run_id'),
    )

class TopicCluster(Base):
    """Store discovered topics and clusters"""
    __tablename__ = 'topic_clusters'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(String(100), nullable=False)
    cluster_id = Column(Integer, nullable=False)  # Cluster number from algorithm
    topic_name = Column(String(200), nullable=False)
    topic_description = Column(Text, nullable=True)
    representative_question = Column(Text, nullable=False)
    question_count = Column(Integer, default=0)
    avg_sentiment = Column(Float, nullable=True)
    urgency_score = Column(Float, default=0.0)
    keywords = Column(JSON, nullable=True)  # List of keywords
    created_at = Column(DateTime(timezone=True), default=func.now())
    
    __table_args__ = (
        Index('idx_topics_run_cluster', 'run_id', 'cluster_id'),
        Index('idx_topics_run', 'run_id'),
    )

class QuestionClusterAssignment(Base):
    """Many-to-many mapping of questions to clusters"""
    __tablename__ = 'question_cluster_assignments'
    
    id = Column(Integer, primary_key=True)
    question_id = Column(Integer, nullable=False)
    cluster_id = Column(Integer, nullable=False)  # FK to topic_clusters.id
    run_id = Column(String(100), nullable=False)
    similarity_score = Column(Float, nullable=True)
    is_representative = Column(Boolean, default=False)
    
    __table_args__ = (
        Index('idx_assignments_question_run', 'question_id', 'run_id'),
        Index('idx_assignments_cluster_run', 'cluster_id', 'run_id'),
    )

class AnalysisCache(Base):
    """Cache computed analysis results for fast retrieval"""
    __tablename__ = 'analysis_cache'
    
    id = Column(Integer, primary_key=True)
    cache_key = Column(String(200), unique=True, nullable=False)
    run_id = Column(String(100), nullable=False)
    cache_type = Column(String(100), nullable=False)  # trends, sentiments, frequencies, etc.
    data = Column(JSON, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    
    __table_args__ = (
        Index('idx_cache_key_type', 'cache_key', 'cache_type'),
        Index('idx_cache_run', 'run_id'),
        Index('idx_cache_expires', 'expires_at'),
    )

class SyncLog(Base):
    """Track Google Sheets synchronization"""
    __tablename__ = 'sync_logs'
    
    id = Column(Integer, primary_key=True)
    sync_type = Column(String(50), nullable=False)  # questions, topics
    source = Column(String(200), nullable=False)  # Google Sheets ID or file path
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(50), default='running')  # running, completed, failed
    records_processed = Column(Integer, default=0)
    records_added = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_skipped = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    
    __table_args__ = (
        Index('idx_sync_type_status', 'sync_type', 'status'),
        Index('idx_sync_completed', 'completed_at'),
    )


class DatabaseManager:
    """Manage database connections and operations"""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database manager"""
        if database_url is None:
            # Use DATABASE_URL from config which has proper URL encoding
            from config import DATABASE_URL
            database_url = DATABASE_URL
            
            # Fallback: Build from individual environment variables if DATABASE_URL not available
            if not database_url:
                host = os.getenv('DB_HOST', '34.56.196.144')
                port = os.getenv('DB_PORT', '5432')
                database = os.getenv('DB_NAME', 'hybrid_topic_modelling_db')
                user = os.getenv('DB_USER', 'postgres')
                password = os.getenv('DB_PASSWORD', 'OPiK@nfA7mJ#oy=^')
                
                database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize database engine with connection pooling"""
        try:
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False  # Set to True for SQL debugging
            )
            
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("Database engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise
    
    def create_tables(self):
        """Create all tables if they don't exist"""
        try:
            # Enable pgvector extension
            with self.engine.connect() as conn:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a database session"""
        if self.SessionLocal is None:
            raise RuntimeError("Database not initialized")
        return self.SessionLocal()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            from sqlalchemy import text
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_latest_analysis_run(self) -> Optional[str]:
        """Get the run_id of the most recent completed analysis"""
        try:
            with self.get_session() as session:
                result = session.query(AnalysisRun.run_id).filter(
                    AnalysisRun.status == 'completed'
                ).order_by(AnalysisRun.completed_at.desc()).first()
                
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"Failed to get latest analysis run: {e}")
            return None
    
    def get_questions_count(self, run_id: Optional[str] = None) -> int:
        """Get total count of questions, optionally for a specific run"""
        try:
            with self.get_session() as session:
                if run_id:
                    # Count questions that were part of a specific analysis run
                    query = session.query(func.count(Question.id)).select_from(Question).join(
                        QuestionClusterAssignment, Question.id == QuestionClusterAssignment.question_id
                    ).filter(QuestionClusterAssignment.run_id == run_id)
                else:
                    # Count all questions
                    query = session.query(func.count(Question.id))
                
                result = query.scalar()
                return result or 0
                
        except Exception as e:
            logger.error(f"Failed to get questions count: {e}")
            return 0
    
    def clear_all_data(self):
        """Clear all data from all tables while preserving structure"""
        try:
            with self.get_session() as session:
                # Clear data in order to respect foreign key constraints
                session.execute(text("TRUNCATE TABLE question_cluster_assignments CASCADE"))
                session.execute(text("TRUNCATE TABLE question_embeddings CASCADE"))
                session.execute(text("TRUNCATE TABLE topic_clusters CASCADE"))
                session.execute(text("TRUNCATE TABLE analysis_cache CASCADE"))
                session.execute(text("TRUNCATE TABLE questions CASCADE"))
                session.execute(text("TRUNCATE TABLE analysis_runs CASCADE"))
                session.execute(text("TRUNCATE TABLE sync_logs CASCADE"))
                session.commit()
                
            logger.info("✅ All database tables cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False

    def get_sync_status(self) -> Dict[str, Any]:
        """Get latest sync status for each sync type"""
        try:
            with self.get_session() as session:
                # Get latest sync for each type
                results = {}
                
                for sync_type in ['questions', 'topics']:
                    latest = session.query(SyncLog).filter(
                        SyncLog.sync_type == sync_type
                    ).order_by(SyncLog.started_at.desc()).first()
                    
                    if latest:
                        results[sync_type] = {
                            'status': latest.status,
                            'started_at': latest.started_at,
                            'completed_at': latest.completed_at,
                            'records_processed': latest.records_processed,
                            'records_added': latest.records_added,
                            'error_message': latest.error_message
                        }
                    else:
                        results[sync_type] = {'status': 'never_run'}
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get sync status: {e}")
            return {}


# Global database manager instance
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def init_database():
    """Initialize database with required extensions and tables"""
    from sqlalchemy import text
    from config import DATABASE_URL
    
    engine = create_engine(DATABASE_URL)
    
    try:
        # Create extensions first
        with engine.connect() as conn:
            # Enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""))
            conn.commit()
            print("✅ PostgreSQL extensions enabled!")
        
        # Create all tables
        Base.metadata.create_all(engine)
        print("✅ Database tables created successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False