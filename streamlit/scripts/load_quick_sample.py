#!/usr/bin/env python3
"""
Quick Sample Data Loader - Load minimal data to test the application
"""

import sys
import os
from datetime import datetime, timezone, timedelta
import uuid

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from database import DatabaseManager, Question, AnalysisRun, TopicCluster
import random

def load_quick_sample():
    """Load minimal sample data for quick testing"""
    print("🚀 Loading quick sample data...")
    
    db_manager = DatabaseManager()
    session = db_manager.get_session()
    
    try:
        # Clear existing data
        session.query(TopicCluster).delete()
        session.query(AnalysisRun).delete() 
        session.query(Question).delete()
        
        # Create analysis run
        run_id = str(uuid.uuid4())
        analysis_run = AnalysisRun(
            run_id=run_id,
            started_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            completed_at=datetime.now(timezone.utc),
            status='completed',
            total_questions=10,
            total_topics=3,
            created_by='quick_loader'
        )
        session.add(analysis_run)
        
        # Add sample questions
        sample_questions = [
            "When does enrollment start?",
            "How do I access my grades?",
            "What are the graduation requirements?",
            "How can I get financial aid?",
            "Where can I find my class schedule?",
            "How do I contact my advisor?",
            "What are the library hours?",
            "How do I withdraw from a course?",
            "Where can I get tutoring help?",
            "How do I apply for graduation?"
        ]
        
        questions = []
        for i, q_text in enumerate(sample_questions):
            question = Question(
                original_text=q_text,
                cleaned_question=q_text,
                timestamp=datetime.now(timezone.utc) - timedelta(days=random.randint(1, 30)),
                user_language='en',
                country='United States',
                state='Utah',
                user_role='student'
            )
            questions.append(question)
        
        session.add_all(questions)
        
        # Add sample topics
        topics = [
            TopicCluster(
                run_id=run_id,
                cluster_id=0,
                topic_name="Academic Requirements",
                topic_description="Questions about graduation and course requirements",
                representative_question="What are the graduation requirements?",
                question_count=4,
                avg_sentiment=0.6,
                urgency_score=0.7,
                keywords=["graduation", "requirements", "courses"]
            ),
            TopicCluster(
                run_id=run_id,
                cluster_id=1, 
                topic_name="Student Services",
                topic_description="Questions about support services and resources",
                representative_question="How do I contact my advisor?",
                question_count=3,
                avg_sentiment=0.5,
                urgency_score=0.5,
                keywords=["advisor", "tutoring", "support"]
            ),
            TopicCluster(
                run_id=run_id,
                cluster_id=2,
                topic_name="Registration and Enrollment",
                topic_description="Questions about enrollment and class registration",
                representative_question="When does enrollment start?",
                question_count=3,
                avg_sentiment=0.7,
                urgency_score=0.8,
                keywords=["enrollment", "registration", "classes"]
            )
        ]
        
        session.add_all(topics)
        session.commit()
        
        print("✅ Quick sample data loaded successfully!")
        print(f"   📊 {len(questions)} questions")
        print(f"   🎯 {len(topics)} topics")
        print(f"   🔧 Run ID: {run_id}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == "__main__":
    load_quick_sample()