"""
Streamlit Data Manager - Handles caching and database integration
"""
import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import threading
import json

logger = logging.getLogger(__name__)


class StreamlitDataManager:
    """
    Manages data flow between Streamlit cache, database, and analysis results
    Ensures fast startup and non-blocking database updates
    """

    def __init__(self):
        self.cache_expiry_hours = 24  # Cache expires after 24 hours
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize session state variables"""
        if "data_manager_initialized" not in st.session_state:
            st.session_state.data_manager_initialized = True
            st.session_state.last_database_sync = None
            st.session_state.database_status = "unknown"
            st.session_state.background_update_running = False

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_latest_analysis_from_database(_self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent analysis from database (cached)
        """
        try:
            import sys
            from pathlib import Path

            src_path = Path(__file__).parent
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

            from database import DatabaseManager, AnalysisRun, AnalysisCache
            from sqlalchemy import desc

            db_manager = DatabaseManager()

            with db_manager.get_session() as session:
                latest_run = (
                    session.query(AnalysisRun)
                    .filter(AnalysisRun.status == "completed")
                    .order_by(desc(AnalysisRun.completed_at))
                    .first()
                )

                if not latest_run:
                    logger.info("No completed analysis runs found in database")
                    return None

                cache_entry = (
                    session.query(AnalysisCache)
                    .filter(
                        AnalysisCache.run_id == latest_run.run_id,
                        AnalysisCache.cache_type == "complete_analysis",
                    )
                    .first()
                )

                if cache_entry:
                    return {
                        "run_id": latest_run.run_id,
                        "completed_at": latest_run.completed_at.isoformat(),
                        "total_questions": latest_run.total_questions,
                        "total_topics": latest_run.total_topics,
                        "config": latest_run.config_snapshot,
                        "cached_results": cache_entry.data,
                        "from_database": True,
                    }

                return None

        except Exception as e:
            logger.error(f"Failed to load analysis from database: {e}")
            return None

    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def get_database_summary(_self) -> Dict[str, Any]:
        """
        Get database summary (cached)
        """
        try:
            import sys
            from pathlib import Path

            src_path = Path(__file__).parent
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

            from database import DatabaseManager
            from sqlalchemy import text

            db_manager = DatabaseManager()

            with db_manager.get_session() as session:
                questions_count = session.execute(
                    text("SELECT COUNT(*) FROM questions")
                ).scalar()
                runs_count = session.execute(
                    text("SELECT COUNT(*) FROM analysis_runs")
                ).scalar()
                topics_count = session.execute(
                    text("SELECT COUNT(*) FROM topic_clusters")
                ).scalar()
                embeddings_count = session.execute(
                    text("SELECT COUNT(*) FROM question_embeddings")
                ).scalar()

                return {
                    "questions": questions_count,
                    "analysis_runs": runs_count,
                    "topics": topics_count,
                    "embeddings": embeddings_count,
                    "status": "connected",
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }

        except Exception as e:
            logger.error(f"Failed to get database summary: {e}")
            return {
                "status": "error",
                "error": str(e),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }

    def update_database_background(
        self,
        questions_df: pd.DataFrame,
        analysis_results: Dict[str, Any],
        config: Dict[str, Any],
    ) -> str:
        """
        Update database in background (non-blocking)
        """
        import uuid

        run_id = str(uuid.uuid4())

        st.session_state.background_update_running = True
        st.session_state.current_background_run_id = run_id

        thread = threading.Thread(
            target=self._update_database_sync,
            args=(run_id, questions_df.copy(), analysis_results.copy(), config.copy()),
            daemon=True,
        )
        thread.start()

        return run_id

    def _update_database_sync(
        self,
        run_id: str,
        questions_df: pd.DataFrame,
        analysis_results: Dict[str, Any],
        config: Dict[str, Any],
    ):
        """
        Synchronous database update (runs in background thread)
        """
        try:
            logger.info(f"Starting background database update: {run_id}")

            import sys
            from pathlib import Path

            src_path = Path(__file__).parent
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

            from database import (
                DatabaseManager,
                AnalysisRun,
                Question,
                TopicCluster,
                QuestionEmbedding,
                QuestionClusterAssignment,
                AnalysisCache,
            )
            import numpy as np

            db_manager = DatabaseManager()

            with db_manager.get_session() as session:
                # Step 1: Create new analysis run (do NOT delete all history)
                analysis_run = AnalysisRun(
                    run_id=run_id,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    status="completed",
                    total_questions=len(questions_df),
                    total_topics=len(analysis_results.get("topic_names", {})),
                    config_snapshot=config,
                    created_by="streamlit_app",
                )
                session.add(analysis_run)
                session.flush()

                # Step 2: Insert questions
                question_id_map = {}
                for idx, row in questions_df.iterrows():
                    question = Question(
                        run_id=run_id,
                        original_text=str(row.get("question", "")),
                        cleaned_question=str(row.get("question", "")),
                        timestamp=datetime.now(timezone.utc),
                        user_language=str(row.get("user_language", "en")),
                        country=str(row.get("country", "Unknown")),
                        state=str(row.get("state", "Unknown")),
                        user_role=str(row.get("user_role", "unknown")),
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    )
                    session.add(question)
                    session.flush()
                    question_id_map[idx] = question.id

                # Step 3: Insert topic clusters
                topic_names = analysis_results.get("topic_names", {})
                cluster_id_map = {}

                clustered_df = analysis_results.get("clustered_questions_df")

                for cluster_num, topic_name in topic_names.items():
                    rep_question = "No representative question"
                    question_count = 0

                    if clustered_df is not None and not clustered_df.empty:
                        cluster_questions = clustered_df[
                            clustered_df["cluster"] == cluster_num
                        ]
                        if not cluster_questions.empty:
                            rep_question = str(cluster_questions.iloc[0]["question"])
                            question_count = len(cluster_questions)

                    topic_cluster = TopicCluster(
                        run_id=run_id,
                        cluster_id=cluster_num,
                        topic_name=topic_name,
                        topic_description=f"Auto-generated topic: {topic_name}",
                        representative_question=rep_question,
                        question_count=question_count,
                        avg_sentiment=0.5,
                        urgency_score=0.5,
                        keywords=[],
                        created_at=datetime.now(timezone.utc),
                    )
                    session.add(topic_cluster)
                    session.flush()
                    cluster_id_map[cluster_num] = topic_cluster.id

                # Step 4: Insert embeddings (support multiple dims)
                embeddings_data = analysis_results.get("embeddings", {})
                if embeddings_data:
                    for idx, embedding in embeddings_data.items():
                        if idx in question_id_map and embedding is not None:
                            try:
                                if isinstance(embedding, (list, np.ndarray)):
                                    embedding_vector = list(embedding)
                                    if len(embedding_vector) >= 512:
                                        question_embedding = QuestionEmbedding(
                                            question_id=question_id_map[idx],
                                            run_id=run_id,
                                            embedding=embedding_vector,
                                            model_version="text-embedding",
                                            created_at=datetime.now(timezone.utc),
                                        )
                                        session.add(question_embedding)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to save embedding for question {idx}: {e}"
                                )

                # Step 5: Insert assignments
                similar_df = analysis_results.get("similar_questions_df")
                if similar_df is not None and not similar_df.empty:
                    for idx, row in similar_df.iterrows():
                        if idx in question_id_map:
                            assignment = QuestionClusterAssignment(
                                question_id=question_id_map[idx],
                                cluster_id=-1,
                                run_id=run_id,
                                similarity_score=float(
                                    row.get("similarity_score", 0.0)
                                ),
                            )
                            session.add(assignment)

                if clustered_df is not None and not clustered_df.empty:
                    for idx, row in clustered_df.iterrows():
                        cluster_num = row.get("cluster", -1)
                        if idx in question_id_map and cluster_num in cluster_id_map:
                            assignment = QuestionClusterAssignment(
                                question_id=question_id_map[idx],
                                cluster_id=cluster_id_map[cluster_num],
                                run_id=run_id,
                                similarity_score=1.0,
                            )
                            session.add(assignment)

                # Step 6: Cache results
                cache_entry = AnalysisCache(
                    cache_key=f"streamlit_analysis_{run_id}",
                    run_id=run_id,
                    cache_type="complete_analysis",
                    data={
                        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                        "total_processed": len(questions_df),
                        "similar_questions": len(similar_df)
                        if similar_df is not None
                        else 0,
                        "new_topics": len(topic_names),
                        "config": config,
                    },
                    expires_at=None,
                    created_at=datetime.now(timezone.utc),
                )
                session.add(cache_entry)

                session.commit()

                logger.info(f"Database update completed successfully: {run_id}")

                # Thread-safe session state update
                st.session_state.update(
                    {
                        "background_update_running": False,
                        "last_database_sync": datetime.now(timezone.utc).isoformat(),
                        "database_status": "updated",
                        "last_run_id": run_id,
                    }
                )

                # Clear Streamlit caches
                try:
                    self.load_latest_analysis_from_database.clear()
                    self.get_database_summary.clear()
                except Exception as e:
                    logger.warning(f"Failed to clear Streamlit caches: {e}")

        except Exception as e:
            logger.error(f"Background database update failed: {e}")
            st.session_state.update(
                {
                    "background_update_running": False,
                    "database_status": "error",
                    "last_database_error": str(e),
                }
            )

    def get_background_update_status(self) -> Dict[str, Any]:
        """Get status of background DB update"""
        return {
            "running": getattr(st.session_state, "background_update_running", False),
            "last_sync": getattr(st.session_state, "last_database_sync", None),
            "status": getattr(st.session_state, "database_status", "unknown"),
            "last_run_id": getattr(st.session_state, "last_run_id", None),
            "error": getattr(st.session_state, "last_database_error", None),
        }

    def display_startup_data_loading(self):
        """UI for startup data loading"""
        st.subheader("📊 Loading Previous Analysis")

        with st.spinner("Checking database for previous analysis..."):
            latest_analysis = self.load_latest_analysis_from_database()
            db_summary = self.get_database_summary()

        if latest_analysis:
            st.success(
                f"✅ Found previous analysis from {latest_analysis['completed_at']}"
            )
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Questions Analyzed", latest_analysis["total_questions"])
                st.metric("Topics Discovered", latest_analysis["total_topics"])
            with col2:
                st.info(
                    f"**Run ID:** `{latest_analysis['run_id']}`\n"
                    f"**Completed:** {latest_analysis['completed_at']}\n"
                    f"**Status:** Ready for use"
                )

            if st.button("📥 **Load Previous Analysis**", type="secondary"):
                st.session_state["hybrid_results"] = {
                    "database_run_id": latest_analysis["run_id"],
                    "from_database": True,
                    "analysis_timestamp": latest_analysis["completed_at"],
                    "total_questions": latest_analysis["total_questions"],
                    "total_topics": latest_analysis["total_topics"],
                    "cached_data": latest_analysis["cached_results"],
                }
                st.success("✅ Previous analysis loaded successfully!")
                st.rerun()
        else:
            st.info("ℹ️ No previous analysis found. Start with a new analysis.")

        if db_summary["status"] == "connected":
            st.success(
                f"🗄️ Database connected: {db_summary['questions']} questions, {db_summary['topics']} topics"
            )
        else:
            st.warning(f"⚠️ Database issue: {db_summary.get('error', 'Unknown error')}")

    def display_background_update_status(self):
        """UI for background update status"""
        status = self.get_background_update_status()
        if status["running"]:
            st.info("🔄 **Database Update**: Running in background...")
            st.progress(0.5)
        elif status["last_sync"]:
            st.success(f"✅ **Database**: Last updated {status['last_sync'][:19]}")
        elif status["status"] == "error":
            st.error(f"❌ **Database Error**: {status['error']}")
        else:
            st.info("💾 **Database**: No recent updates")


# Global instance
data_manager = StreamlitDataManager()
