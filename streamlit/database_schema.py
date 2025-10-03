"""
Database Schema Documentation - BYU Pathway Questions Analysis

This file documents the complete database schema with all tables,
primary keys, foreign keys, and relationships.
"""

# Database Schema Overview
# ========================

DATABASE_SCHEMA = {
    "tables": {
        "questions": {
            "description": "Store raw and processed questions",
            "primary_key": "id (Integer, Auto-increment)",
            "columns": {
                "id": "Integer (Primary Key)",
                "original_text": "Text (Raw question text)",
                "cleaned_question": "Text (Processed question)",
                "timestamp": "DateTime (Question timestamp)",
                "user_language": "String(10) (Language code)",
                "country": "String(100) (User country)",
                "state": "String(100) (User state)",
                "user_role": "String(50) (missionary, student, etc.)",
                "created_at": "DateTime (Record creation)",
                "updated_at": "DateTime (Last update)"
            },
            "indexes": [
                "idx_questions_timestamp (timestamp)",
                "idx_questions_language (user_language)",
                "idx_questions_created (created_at)",
                "idx_questions_country_state (country, state)"
            ],
            "relationships": {
                "outgoing": [
                    "question_embeddings.question_id → questions.id",
                    "question_cluster_assignments.question_id → questions.id"
                ]
            }
        },

        "analysis_runs": {
            "description": "Track analysis runs and their metadata",
            "primary_key": "id (Integer, Auto-increment)",
            "columns": {
                "id": "Integer (Primary Key)",
                "run_id": "String(100) (UUID, Unique)",
                "started_at": "DateTime (Analysis start time)",
                "completed_at": "DateTime (Analysis completion time)",
                "status": "String(50) (running, completed, failed)",
                "total_questions": "Integer (Questions processed)",
                "total_topics": "Integer (Topics discovered)",
                "config_snapshot": "JSON (Analysis configuration)",
                "error_message": "Text (Error details if failed)",
                "created_by": "String(100) (User/system that ran analysis)"
            },
            "indexes": [
                "idx_analysis_runs_status_started (status, started_at)",
                "idx_analysis_runs_completed (completed_at)"
            ],
            "relationships": {
                "outgoing": [
                    "question_embeddings.run_id → analysis_runs.run_id",
                    "topic_clusters.run_id → analysis_runs.run_id",
                    "question_cluster_assignments.run_id → analysis_runs.run_id",
                    "analysis_cache.run_id → analysis_runs.run_id"
                ]
            }
        },

        "question_embeddings": {
            "description": "Store question embeddings for similarity analysis",
            "primary_key": "id (Integer, Auto-increment)",
            "columns": {
                "id": "Integer (Primary Key)",
                "question_id": "Integer (Foreign Key → questions.id)",
                "run_id": "String(100) (Foreign Key → analysis_runs.run_id)",
                "embedding": "Vector(1536) (OpenAI embedding vector)",
                "model_version": "String(100) (Embedding model used)",
                "created_at": "DateTime (Record creation)"
            },
            "indexes": [
                "idx_embeddings_question_run (question_id, run_id)",
                "idx_embeddings_run (run_id)"
            ],
            "relationships": {
                "incoming": [
                    "questions.id → question_embeddings.question_id",
                    "analysis_runs.run_id → question_embeddings.run_id"
                ]
            }
        },

        "topic_clusters": {
            "description": "Store discovered topics and clusters",
            "primary_key": "id (Integer, Auto-increment)",
            "columns": {
                "id": "Integer (Primary Key)",
                "run_id": "String(100) (Foreign Key → analysis_runs.run_id)",
                "cluster_id": "Integer (Algorithm cluster number)",
                "topic_name": "String(200) (Generated topic name)",
                "topic_description": "Text (Topic description)",
                "representative_question": "Text (Example question)",
                "question_count": "Integer (Questions in cluster)",
                "avg_sentiment": "Float (Average sentiment score)",
                "urgency_score": "Float (Urgency level)",
                "keywords": "JSON (Topic keywords)",
                "created_at": "DateTime (Record creation)"
            },
            "indexes": [
                "idx_topics_run_cluster (run_id, cluster_id)",
                "idx_topics_run (run_id)"
            ],
            "relationships": {
                "incoming": [
                    "analysis_runs.run_id → topic_clusters.run_id"
                ],
                "outgoing": [
                    "question_cluster_assignments.cluster_id → topic_clusters.id"
                ]
            }
        },

        "question_cluster_assignments": {
            "description": "Many-to-many mapping of questions to clusters",
            "primary_key": "id (Integer, Auto-increment)",
            "columns": {
                "id": "Integer (Primary Key)",
                "question_id": "Integer (Foreign Key → questions.id)",
                "cluster_id": "Integer (Foreign Key → topic_clusters.id)",
                "run_id": "String(100) (Foreign Key → analysis_runs.run_id)",
                "similarity_score": "Float (Similarity/confidence score)",
                "is_representative": "Boolean (Is this question representative)"
            },
            "indexes": [
                "idx_assignments_question_run (question_id, run_id)",
                "idx_assignments_cluster_run (cluster_id, run_id)"
            ],
            "relationships": {
                "incoming": [
                    "questions.id → question_cluster_assignments.question_id",
                    "topic_clusters.id → question_cluster_assignments.cluster_id",
                    "analysis_runs.run_id → question_cluster_assignments.run_id"
                ]
            }
        },

        "analysis_cache": {
            "description": "Cache computed analysis results for fast retrieval",
            "primary_key": "id (Integer, Auto-increment)",
            "columns": {
                "id": "Integer (Primary Key)",
                "cache_key": "String(200) (Unique cache identifier)",
                "run_id": "String(100) (Foreign Key → analysis_runs.run_id)",
                "cache_type": "String(100) (trends, sentiments, etc.)",
                "data": "JSON (Cached analysis data)",
                "expires_at": "DateTime (Cache expiration)",
                "created_at": "DateTime (Cache creation)"
            },
            "indexes": [
                "idx_cache_key_type (cache_key, cache_type)",
                "idx_cache_run (run_id)",
                "idx_cache_expires (expires_at)"
            ],
            "relationships": {
                "incoming": [
                    "analysis_runs.run_id → analysis_cache.run_id"
                ]
            }
        },

        "sync_logs": {
            "description": "Track Google Sheets synchronization",
            "primary_key": "id (Integer, Auto-increment)",
            "columns": {
                "id": "Integer (Primary Key)",
                "sync_type": "String(50) (questions, topics)",
                "source": "String(200) (Google Sheets ID or file path)",
                "started_at": "DateTime (Sync start time)",
                "completed_at": "DateTime (Sync completion time)",
                "status": "String(50) (running, completed, failed)",
                "records_processed": "Integer (Total records processed)",
                "records_added": "Integer (New records added)",
                "records_updated": "Integer (Records updated)",
                "records_skipped": "Integer (Records skipped)",
                "error_message": "Text (Error details if failed)"
            },
            "indexes": [
                "idx_sync_type_status (sync_type, status)",
                "idx_sync_completed (completed_at)"
            ],
            "relationships": {
                "outgoing": []  # No foreign key relationships
            }
        }
    },

    "relationships": {
        "one_to_many": [
            "analysis_runs → question_embeddings (run_id)",
            "analysis_runs → topic_clusters (run_id)",
            "analysis_runs → question_cluster_assignments (run_id)",
            "analysis_runs → analysis_cache (run_id)",
            "questions → question_embeddings (question_id)",
            "questions → question_cluster_assignments (question_id)",
            "topic_clusters → question_cluster_assignments (cluster_id)"
        ],
        "many_to_many": [
            "questions ↔ topic_clusters (via question_cluster_assignments)"
        ]
    },

    "data_flow": {
        "analysis_pipeline": [
            "1. Questions loaded into 'questions' table",
            "2. Analysis run created in 'analysis_runs' table",
            "3. Embeddings generated and stored in 'question_embeddings'",
            "4. Topics discovered and stored in 'topic_clusters'",
            "5. Question-topic assignments stored in 'question_cluster_assignments'",
            "6. Analysis results cached in 'analysis_cache'",
            "7. Sync operations logged in 'sync_logs'"
        ]
    }
}

def print_schema_summary():
    """Print a summary of the database schema"""
    print("🗄️ BYU Pathway Questions Database Schema")
    print("=" * 50)

    for table_name, table_info in DATABASE_SCHEMA["tables"].items():
        print(f"\n📋 {table_name.upper()}")
        print(f"   {table_info['description']}")
        print(f"   Primary Key: {table_info['primary_key']}")

        if table_info['relationships']['outgoing']:
            print("   Foreign Key Relationships:")
            for rel in table_info['relationships']['outgoing']:
                print(f"     → {rel}")

    print("\n🔗 Key Relationships:")
    for rel in DATABASE_SCHEMA["relationships"]["one_to_many"]:
        print(f"   • {rel}")

    print("\n🔄 Many-to-Many:")
    for rel in DATABASE_SCHEMA["relationships"]["many_to_many"]:
        print(f"   • {rel}")

if __name__ == "__main__":
    print_schema_summary()