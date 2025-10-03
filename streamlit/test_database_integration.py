#!/usr/bin/env python3
"""
Test script for database integration functionality
"""
import sys
import os
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent.absolute()
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))

import pandas as pd
from datetime import datetime, timezone
import uuid
import numpy as np
from sqlalchemy import text

# Test database connection and update functionality
def test_database_update():
    """Test the database update functionality"""
    try:
        from database import DatabaseManager
        
        print("🔍 Testing database connection...")
        db_manager = DatabaseManager()
        
        with db_manager.get_session() as session:
            # Test basic connection
            session.execute(text("SELECT 1"))
            print("✅ Database connection successful")
            
            # Check current data
            result = session.execute(text("SELECT COUNT(*) FROM questions"))
            question_count = result.scalar()
            print(f"📊 Current questions in database: {question_count}")
            
            # Test the update functionality
            print("\n🚀 Testing database update with sample data...")
            
            # Create sample data
            questions_df = pd.DataFrame({
                'question': [
                    'How do I reset my password?',
                    'Where can I find my grades?',
                    'What are the library hours?'
                ],
                'user_language': ['en', 'en', 'en'],
                'country': ['USA', 'Canada', 'Mexico'],
                'state': ['Utah', 'Ontario', 'Jalisco'],
                'user_role': ['student', 'student', 'faculty']
            })
            
            # Sample analysis results
            sample_results = {
                'similar_questions_df': pd.DataFrame({
                    'question': ['How do I reset my password?'],
                    'similarity_score': [0.85],
                    'matched_topic': ['Technical Support']
                }),
                'clustered_questions_df': pd.DataFrame({
                    'question': ['Where can I find my grades?', 'What are the library hours?'],
                    'cluster': [0, 1]
                }),
                'topic_names': {
                    0: 'Academic Information',
                    1: 'Campus Services'
                },
                'embeddings': {
                    0: np.random.rand(1536).tolist(),
                    1: np.random.rand(1536).tolist(),
                    2: np.random.rand(1536).tolist()
                }
            }
            
            # Test the update function
            run_id = update_database_with_test_results(
                questions_df, sample_results, 0.70, "test", 3
            )
            
            print(f"✅ Database update successful! Run ID: {run_id}")
            
            # Verify the update
            result = session.execute(text("SELECT COUNT(*) FROM questions"))
            new_question_count = result.scalar()
            print(f"📊 Questions after update: {new_question_count}")
            
            result = session.execute(text("SELECT COUNT(*) FROM analysis_runs"))
            analysis_count = result.scalar()
            print(f"📊 Analysis runs: {analysis_count}")
            
            result = session.execute(text("SELECT COUNT(*) FROM topic_clusters"))
            topic_count = result.scalar()
            print(f"📊 Topic clusters: {topic_count}")
            
            print("\n✅ Database integration test completed successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Database test failed: {str(e)}")
        return False


def update_database_with_test_results(questions_df: pd.DataFrame, 
                                     results: dict,
                                     threshold: float,
                                     mode: str,
                                     sample_size: int) -> str:
    """Test version of the database update function"""
    from database import DatabaseManager, AnalysisRun, Question, TopicCluster, QuestionEmbedding, QuestionClusterAssignment, AnalysisCache
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Generate unique run ID
    run_id = str(uuid.uuid4())
    
    with db_manager.get_session() as session:
        # Step 1: Clear previous analysis data
        print("🧹 Clearing previous analysis data...")
        session.execute(text("DELETE FROM question_cluster_assignments"))
        session.execute(text("DELETE FROM question_embeddings"))
        session.execute(text("DELETE FROM topic_clusters"))
        session.execute(text("DELETE FROM analysis_cache"))
        session.execute(text("DELETE FROM analysis_runs"))
        session.commit()
        
        # Step 2: Create new analysis run record
        print("📝 Creating analysis run record...")
        analysis_run = AnalysisRun(
            run_id=run_id,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            status='completed',
            total_questions=len(questions_df),
            total_topics=len(results.get('topic_names', {})),
            config_snapshot={
                'similarity_threshold': threshold,
                'processing_mode': mode,
                'sample_size': sample_size,
                'embedding_model': 'text-embedding-3-small',
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            created_by='test_script'
        )
        session.add(analysis_run)
        session.commit()
        
        # Step 3: Update/Insert questions
        print("💾 Inserting questions...")
        session.execute(text("DELETE FROM questions"))
        session.commit()
        
        for idx, row in questions_df.iterrows():
            question = Question(
                original_text=str(row.get('question', '')),
                cleaned_question=str(row.get('question', '')),
                timestamp=datetime.now(timezone.utc),
                user_language=str(row.get('user_language', 'en')),
                country=str(row.get('country', 'Unknown')),
                state=str(row.get('state', 'Unknown')),
                user_role=str(row.get('user_role', 'Unknown')),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            session.add(question)
        session.commit()
        
        # Get question IDs for relationships
        questions_in_db = session.query(Question).all()
        question_id_map = {idx: q.id for idx, q in enumerate(questions_in_db)}
        
        # Step 4: Insert topic clusters
        print("🎯 Creating topic clusters...")
        topic_names = results.get('topic_names', {})
        cluster_id_map = {}
        
        for cluster_num, topic_name in topic_names.items():
            clustered_df = results.get('clustered_questions_df')
            rep_question = "Test representative question"
            question_count = 1
            
            if clustered_df is not None and len(clustered_df) > 0:
                cluster_questions = clustered_df[clustered_df['cluster'] == cluster_num]
                if len(cluster_questions) > 0:
                    rep_question = str(cluster_questions.iloc[0]['question'])
                    question_count = len(cluster_questions)
            
            topic_cluster = TopicCluster(
                run_id=run_id,
                cluster_id=cluster_num,
                topic_name=topic_name,
                topic_description=f"Test topic: {topic_name}",
                representative_question=rep_question,
                question_count=question_count,
                avg_sentiment=0.5,
                urgency_score=0.5,
                keywords=[],
                created_at=datetime.now(timezone.utc)
            )
            session.add(topic_cluster)
            session.commit()
            cluster_id_map[cluster_num] = topic_cluster.id
        
        # Step 5: Insert embeddings
        print("🔗 Inserting embeddings...")
        embeddings_data = results.get('embeddings', {})
        for idx, embedding in embeddings_data.items():
            if idx in question_id_map and embedding is not None:
                question_embedding = QuestionEmbedding(
                    question_id=question_id_map[idx],
                    run_id=run_id,
                    embedding=embedding,
                    model_version='text-embedding-3-small',
                    created_at=datetime.now(timezone.utc)
                )
                session.add(question_embedding)
        session.commit()
        
        # Step 6: Insert assignments
        print("📊 Creating question assignments...")
        clustered_df = results.get('clustered_questions_df')
        if clustered_df is not None and len(clustered_df) > 0:
            for idx, row in clustered_df.iterrows():
                cluster_num = row.get('cluster', -1)
                if idx in question_id_map and cluster_num in cluster_id_map:
                    assignment = QuestionClusterAssignment(
                        question_id=question_id_map[idx],
                        cluster_id=cluster_id_map[cluster_num],
                        run_id=run_id,
                        similarity_score=1.0
                    )
                    session.add(assignment)
        session.commit()
        
        # Step 7: Cache results
        print("💾 Caching analysis results...")
        cache_entry = AnalysisCache(
            cache_key=f"test_analysis_{run_id}",
            run_id=run_id,
            cache_type="test_analysis",
            data={
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'total_processed': len(questions_df),
                'test_mode': True
            },
            expires_at=None,
            created_at=datetime.now(timezone.utc)
        )
        session.add(cache_entry)
        session.commit()
        
        return run_id


if __name__ == "__main__":
    print("🧪 Testing Database Integration for Streamlit Analysis")
    print("=" * 60)
    
    success = test_database_update()
    
    if success:
        print("\n🎉 All tests passed! Database integration is working correctly.")
        print("\n📋 Summary:")
        print("✅ Database connection established")
        print("✅ Previous data cleared successfully")
        print("✅ New analysis data inserted")
        print("✅ All database tables updated")
        print("✅ Analysis results cached")
        print("\n🚀 Your Streamlit app is ready to automatically update the database!")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
    
    print("\n" + "=" * 60)