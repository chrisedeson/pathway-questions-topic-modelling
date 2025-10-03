#!/usr/bin/env python3
"""
Sample Data Loading Script for BYU Pathway Questions Analysis Database

This script populates all database tables with realistic sample data for testing and development.
"""

import sys
import os
from datetime import datetime, timezone, timedelta
import random
import uuid
import json
import logging
from typing import List, Dict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from database import (
    DatabaseManager, Question, AnalysisRun, QuestionEmbedding, 
    TopicCluster, QuestionClusterAssignment, AnalysisCache, SyncLog
)
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SampleDataGenerator:
    """Generate realistic sample data for all database tables"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.session = self.db_manager.get_session()
        
        # Sample data templates
        self.sample_questions = [
            "When does the next enrollment period start?",
            "How do I access my student portal?", 
            "What are the graduation requirements?",
            "How can I change my major?",
            "What financial aid options are available?",
            "When are final exams scheduled?",
            "How do I register for classes?",
            "What is the refund policy for tuition?",
            "How do I contact my academic advisor?",
            "Where can I find my transcript?",
            "What are the library hours?",
            "How do I apply for student housing?",
            "What technology requirements do I need?",
            "How do I access online classes?",
            "What support services are available for students?",
            "How do I withdraw from a course?",
            "What is the attendance policy?",
            "How do I appeal a grade?",
            "Where can I get tutoring help?",
            "How do I apply for graduation?",
            "What career services are offered?",
            "How do I access the student email system?",
            "What dining options are available on campus?",
            "How do I report a technical issue?",
            "What are the campus safety procedures?",
            "How do I join student organizations?",
            "What are the parking policies?",
            "How do I get a student ID card?",
            "What mental health resources are available?",
            "How do I access disability services?"
        ]
        
        self.countries = ["United States", "Canada", "Mexico", "Brazil", "United Kingdom", 
                         "Germany", "France", "Australia", "Philippines", "Ghana"]
        
        self.states = ["Utah", "Idaho", "Arizona", "California", "Texas", "Florida", 
                      "New York", "Colorado", "Nevada", "Washington"]
        
        self.user_roles = ["student", "missionary", "mentor", "instructor", "administrator"]
        
        self.topic_templates = [
            {
                "name": "Enrollment and Registration",
                "description": "Questions about course enrollment, registration deadlines, and class selection",
                "keywords": ["enrollment", "registration", "classes", "deadlines", "schedule"]
            },
            {
                "name": "Academic Requirements",
                "description": "Questions about graduation requirements, major requirements, and academic policies",
                "keywords": ["graduation", "requirements", "major", "credits", "policies"]
            },
            {
                "name": "Financial Aid and Tuition",
                "description": "Questions about financial assistance, scholarships, and tuition payments",
                "keywords": ["financial aid", "tuition", "scholarships", "payment", "refunds"]
            },
            {
                "name": "Student Support Services",
                "description": "Questions about academic advising, tutoring, and student assistance",
                "keywords": ["advisor", "tutoring", "support", "services", "help"]
            },
            {
                "name": "Technology and Online Learning",
                "description": "Questions about online classes, technology requirements, and digital tools",
                "keywords": ["online", "technology", "portal", "email", "technical"]
            },
            {
                "name": "Campus Life and Services",
                "description": "Questions about campus facilities, dining, housing, and student activities",
                "keywords": ["campus", "housing", "dining", "activities", "organizations"]
            }
        ]

    def generate_questions(self, count: int = 50) -> List[Dict]:
        """Generate sample questions with realistic metadata"""
        questions = []
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        
        for i in range(count):
            # Add some variation to the base questions
            question_text = random.choice(self.sample_questions)
            if random.random() < 0.3:  # 30% chance to add some variation
                prefixes = ["(ACM Question): ", "Quick question: ", "Help needed: ", ""]
                question_text = random.choice(prefixes) + question_text
            
            # Generate realistic timestamp within last 30 days
            timestamp = base_time + timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            questions.append({
                'original_text': question_text,
                'cleaned_question': question_text.replace("(ACM Question): ", "").replace("Quick question: ", "").replace("Help needed: ", ""),
                'timestamp': timestamp,
                'user_language': random.choices(['en', 'es', 'pt', 'fr'], weights=[80, 10, 5, 5])[0],
                'country': random.choice(self.countries),
                'state': random.choice(self.states),
                'user_role': random.choice(self.user_roles)
            })
        
        return questions

    def generate_embeddings(self, question_ids: List[int], run_id: str) -> List[Dict]:
        """Generate sample embeddings (random vectors for demo purposes)"""
        embeddings = []
        
        for question_id in question_ids:
            # Generate random embedding vector (1536 dimensions for text-embedding-3-small)
            embedding_vector = np.random.normal(0, 1, 1536).tolist()
            
            embeddings.append({
                'question_id': question_id,
                'run_id': run_id,
                'embedding': embedding_vector,
                'model_version': 'text-embedding-3-small'
            })
        
        return embeddings

    def generate_topic_clusters(self, run_id: str, num_clusters: int = 6) -> List[Dict]:
        """Generate sample topic clusters"""
        clusters = []
        
        for i, template in enumerate(self.topic_templates[:num_clusters]):
            clusters.append({
                'run_id': run_id,
                'cluster_id': i,
                'topic_name': template['name'],
                'topic_description': template['description'],
                'representative_question': random.choice(self.sample_questions),
                'question_count': random.randint(5, 15),
                'avg_sentiment': random.uniform(0.3, 0.8),  # Generally positive
                'urgency_score': random.uniform(0.1, 0.9),
                'keywords': template['keywords']
            })
        
        return clusters

    def generate_assignments(self, question_ids: List[int], cluster_ids: List[int], run_id: str) -> List[Dict]:
        """Generate question-to-cluster assignments"""
        assignments = []
        
        for question_id in question_ids:
            # Most questions get assigned to a cluster
            if random.random() < 0.85:  # 85% assignment rate
                cluster_id = random.choice(cluster_ids)
                assignments.append({
                    'question_id': question_id,
                    'cluster_id': cluster_id,
                    'run_id': run_id,
                    'similarity_score': random.uniform(0.6, 0.95),
                    'is_representative': random.random() < 0.1  # 10% are representative
                })
        
        return assignments

    def generate_analysis_cache(self, run_id: str) -> List[Dict]:
        """Generate sample cached analysis results"""
        cache_entries = []
        
        # Dashboard cache
        dashboard_data = {
            'total_questions': 50,
            'total_topics': 6,
            'clusters': [
                {'name': topic['name'], 'count': random.randint(5, 15)} 
                for topic in self.topic_templates[:6]
            ],
            'language_distribution': {'en': 40, 'es': 5, 'pt': 3, 'fr': 2},
            'country_distribution': {country: random.randint(1, 10) for country in self.countries[:5]}
        }
        
        cache_entries.append({
            'cache_key': f'dashboard_all_{run_id}',
            'run_id': run_id,
            'cache_type': 'dashboard',
            'data': dashboard_data,
            'expires_at': datetime.now(timezone.utc) + timedelta(hours=24)
        })
        
        # Trends cache
        trends_data = {
            'daily_counts': [random.randint(1, 5) for _ in range(30)],
            'topic_trends': {
                topic['name']: [random.randint(0, 3) for _ in range(30)]
                for topic in self.topic_templates[:6]
            }
        }
        
        cache_entries.append({
            'cache_key': f'trends_30d_{run_id}',
            'run_id': run_id,
            'cache_type': 'trends',
            'data': trends_data,
            'expires_at': datetime.now(timezone.utc) + timedelta(hours=12)
        })
        
        return cache_entries

    def generate_sync_logs(self) -> List[Dict]:
        """Generate sample sync log entries"""
        logs = []
        base_time = datetime.now(timezone.utc) - timedelta(days=7)
        
        for i in range(10):  # 10 sync entries over past week
            start_time = base_time + timedelta(days=i * 0.7)
            
            logs.append({
                'sync_type': random.choice(['questions', 'topics']),
                'source': '1KIu4W9-BYRpZKxrpoWy6qpCBXjSDeRRmKek6q71wTRE',
                'started_at': start_time,
                'completed_at': start_time + timedelta(minutes=random.randint(1, 5)),
                'status': random.choices(['completed', 'failed'], weights=[90, 10])[0],
                'records_processed': random.randint(20, 100),
                'records_added': random.randint(0, 20),
                'records_updated': random.randint(0, 10),
                'records_skipped': random.randint(0, 5),
                'error_message': None
            })
        
        return logs

    def clear_all_data(self):
        """Clear all existing data from tables"""
        logger.info("🧹 Clearing existing data...")
        
        try:
            # Delete in reverse order of dependencies
            self.session.query(AnalysisCache).delete()
            self.session.query(QuestionClusterAssignment).delete()
            self.session.query(TopicCluster).delete()
            self.session.query(QuestionEmbedding).delete()
            self.session.query(SyncLog).delete()
            self.session.query(AnalysisRun).delete()
            self.session.query(Question).delete()
            
            self.session.commit()
            logger.info("✅ All existing data cleared")
            
        except Exception as e:
            logger.error(f"❌ Error clearing data: {e}")
            self.session.rollback()
            raise

    def load_sample_data(self, clear_first: bool = True):
        """Load complete sample dataset"""
        if clear_first:
            self.clear_all_data()
        
        logger.info("🚀 Loading sample data...")
        
        try:
            # 1. Create analysis run
            run_id = str(uuid.uuid4())
            analysis_run = AnalysisRun(
                run_id=run_id,
                started_at=datetime.now(timezone.utc) - timedelta(minutes=30),
                completed_at=datetime.now(timezone.utc),
                status='completed',
                total_questions=50,
                total_topics=6,
                config_snapshot={
                    'similarity_threshold': 0.7,
                    'min_cluster_size': 3,
                    'embedding_model': 'text-embedding-3-small'
                },
                created_by='sample_data_script'
            )
            self.session.add(analysis_run)
            self.session.flush()
            logger.info(f"✅ Created analysis run: {run_id}")
            
            # 2. Load questions
            question_data = self.generate_questions(50)
            questions = [Question(**q) for q in question_data]
            self.session.add_all(questions)
            self.session.flush()
            question_ids = [q.id for q in questions]
            logger.info(f"✅ Loaded {len(questions)} sample questions")
            
            # 3. Load embeddings
            embedding_data = self.generate_embeddings(question_ids, run_id)
            embeddings = [QuestionEmbedding(**e) for e in embedding_data]
            self.session.add_all(embeddings)
            logger.info(f"✅ Loaded {len(embeddings)} embeddings")
            
            # 4. Load topic clusters
            cluster_data = self.generate_topic_clusters(run_id, 6)
            clusters = [TopicCluster(**c) for c in cluster_data]
            self.session.add_all(clusters)
            self.session.flush()
            cluster_ids = [c.id for c in clusters]
            logger.info(f"✅ Loaded {len(clusters)} topic clusters")
            
            # 5. Load question assignments
            assignment_data = self.generate_assignments(question_ids, cluster_ids, run_id)
            assignments = [QuestionClusterAssignment(**a) for a in assignment_data]
            self.session.add_all(assignments)
            logger.info(f"✅ Loaded {len(assignments)} question assignments")
            
            # 6. Load analysis cache
            cache_data = self.generate_analysis_cache(run_id)
            cache_entries = [AnalysisCache(**c) for c in cache_data]
            self.session.add_all(cache_entries)
            logger.info(f"✅ Loaded {len(cache_entries)} cache entries")
            
            # 7. Load sync logs
            sync_data = self.generate_sync_logs()
            sync_logs = [SyncLog(**s) for s in sync_data]
            self.session.add_all(sync_logs)
            logger.info(f"✅ Loaded {len(sync_logs)} sync log entries")
            
            # Commit everything
            self.session.commit()
            
            logger.info("🎉 Sample data loading completed successfully!")
            
            # Print summary
            self.print_data_summary()
            
        except Exception as e:
            logger.error(f"❌ Error loading sample data: {e}")
            self.session.rollback()
            raise
        finally:
            self.session.close()

    def print_data_summary(self):
        """Print summary of loaded data"""
        logger.info("\n📊 DATA SUMMARY:")
        
        try:
            session = self.db_manager.get_session()
            
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
                logger.info(f"   {name}: {count} records")
            
            # Show sample topics
            topics = session.query(TopicCluster).limit(3).all()
            if topics:
                logger.info("\n🔍 SAMPLE TOPICS:")
                for topic in topics:
                    logger.info(f"   • {topic.topic_name} ({topic.question_count} questions)")
            
            session.close()
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")


def main():
    """Main function to run sample data loading"""
    print("🏗️  BYU Pathway Questions Analysis - Sample Data Loader")
    print("=" * 60)
    
    try:
        generator = SampleDataGenerator()
        generator.load_sample_data(clear_first=True)
        
        print("\n✅ Sample data loading completed!")
        print("📌 You can now test the application with realistic data.")
        print("🔗 Run: streamlit run streamlit_app.py")
        
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()