#!/usr/bin/env python3
"""
Database Data Viewer - Check what data is currently in the database
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from database import (
    DatabaseManager, Question, AnalysisRun, QuestionEmbedding, 
    TopicCluster, QuestionClusterAssignment, AnalysisCache, SyncLog
)

def show_database_contents():
    """Display contents of all database tables"""
    print("🔍 BYU Pathway Questions Database Contents")
    print("=" * 60)
    
    db_manager = DatabaseManager()
    session = db_manager.get_session()
    
    try:
        # Table counts
        print("\n📊 TABLE COUNTS:")
        tables = [
            (Question, "Questions"),
            (AnalysisRun, "Analysis Runs"),
            (QuestionEmbedding, "Question Embeddings"),
            (TopicCluster, "Topic Clusters"),
            (QuestionClusterAssignment, "Question Assignments"),
            (AnalysisCache, "Cache Entries"),
            (SyncLog, "Sync Logs")
        ]
        
        for model, name in tables:
            count = session.query(model).count()
            print(f"   {name}: {count} records")
        
        # Latest analysis run
        print("\n🔧 LATEST ANALYSIS RUN:")
        latest_run = session.query(AnalysisRun).order_by(AnalysisRun.started_at.desc()).first()
        if latest_run:
            print(f"   Run ID: {latest_run.run_id}")
            print(f"   Status: {latest_run.status}")
            print(f"   Started: {latest_run.started_at}")
            print(f"   Completed: {latest_run.completed_at}")
            print(f"   Questions: {latest_run.total_questions}")
            print(f"   Topics: {latest_run.total_topics}")
        else:
            print("   No analysis runs found")
        
        # Sample questions
        print("\n❓ SAMPLE QUESTIONS:")
        questions = session.query(Question).limit(5).all()
        for i, q in enumerate(questions, 1):
            print(f"   {i}. \"{q.cleaned_question}\" ({q.user_role}, {q.country})")
        
        # Topic clusters
        print("\n🎯 TOPIC CLUSTERS:")
        topics = session.query(TopicCluster).all()
        for topic in topics:
            print(f"   • {topic.topic_name}")
            print(f"     Description: {topic.topic_description}")
            print(f"     Questions: {topic.question_count}")
            print(f"     Representative: \"{topic.representative_question}\"")
            print()
        
        # Recent sync logs
        print("🔄 RECENT SYNC LOGS:")
        logs = session.query(SyncLog).order_by(SyncLog.started_at.desc()).limit(3).all()
        for log in logs:
            print(f"   {log.sync_type}: {log.status} ({log.records_processed} records)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        session.close()


def check_database_health():
    """Check database connectivity and basic health"""
    print("\n🏥 DATABASE HEALTH CHECK:")
    
    try:
        db_manager = DatabaseManager()
        session = db_manager.get_session()
        
        # Test basic query
        count = session.query(Question).count()
        print(f"   ✅ Database connection: OK")
        print(f"   ✅ Basic queries: OK ({count} questions)")
        
        # Check for analysis runs
        runs = session.query(AnalysisRun).count()
        if runs > 0:
            print(f"   ✅ Analysis runs: {runs} found")
        else:
            print(f"   ⚠️  No analysis runs found")
        
        # Check for embeddings
        embeddings = session.query(QuestionEmbedding).count()
        if embeddings > 0:
            print(f"   ✅ Embeddings: {embeddings} found")
        else:
            print(f"   ⚠️  No embeddings found")
        
        session.close()
        
    except Exception as e:
        print(f"   ❌ Database health check failed: {e}")


if __name__ == "__main__":
    show_database_contents()
    check_database_health()