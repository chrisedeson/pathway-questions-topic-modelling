#!/usr/bin/env python3
"""
Test script for Streamlit Analysis Cache Manager
Demonstrates the local caching workflow before database storage
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
current_dir = Path(__file__).parent.absolute()
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))

from streamlit_cache_manager import StreamlitAnalysisCacheManager
import uuid


def create_sample_data():
    """Create sample analysis data for testing"""
    
    # Sample questions dataframe
    questions_data = [
        {"id": 1, "question": "How do I register for classes?", "language": "en", "timestamp": datetime.now()},
        {"id": 2, "question": "What are the library hours?", "language": "en", "timestamp": datetime.now()},
        {"id": 3, "question": "How can I get financial aid?", "language": "en", "timestamp": datetime.now()},
        {"id": 4, "question": "Where do I find my transcript?", "language": "en", "timestamp": datetime.now()},
        {"id": 5, "question": "How do I contact my advisor?", "language": "en", "timestamp": datetime.now()},
    ]
    
    questions_df = pd.DataFrame(questions_data)
    
    # Sample hybrid results
    hybrid_results = {
        'clusters': [
            {
                'id': 1,
                'name': 'Academic Services',
                'representative_question': 'How do I register for classes?',
                'questions_count': 2
            },
            {
                'id': 2,
                'name': 'Student Support',
                'representative_question': 'How can I get financial aid?',
                'questions_count': 3
            }
        ],
        'similar_questions_df': pd.DataFrame([
            {"question": "How do I register for classes?", "similar_to": "Class registration help", "similarity": 0.85}
        ]),
        'clustered_questions_df': pd.DataFrame([
            {"question": "What are the library hours?", "cluster_id": 1, "cluster_name": "Academic Services"},
            {"question": "How can I get financial aid?", "cluster_id": 2, "cluster_name": "Student Support"},
        ]),
        'embeddings': [[0.1, 0.2, 0.3] for _ in range(5)],  # Mock embeddings
        'processing_stats': {
            'total_processed': 5,
            'similar_found': 1,
            'new_clusters': 2
        }
    }
    
    # Sample configuration
    config = {
        'similarity_threshold': 0.7,
        'processing_mode': 'sample',
        'sample_size': 5,
        'embedding_model': 'text-embedding-3-small',
        'clustering_algorithm': 'HDBSCAN'
    }
    
    return questions_df, hybrid_results, config


def test_caching_workflow():
    """Test the complete caching workflow"""
    
    print("🧪 Testing Streamlit Analysis Cache Manager")
    print("=" * 50)
    
    # Initialize cache manager
    cache_manager = StreamlitAnalysisCacheManager()
    
    # Create sample data
    questions_df, hybrid_results, config = create_sample_data()
    
    # Test 1: Cache analysis results
    print("\n📦 Test 1: Caching analysis results...")
    analysis_id = str(uuid.uuid4())[:8]
    
    metadata = {
        'test_run': True,
        'total_questions': len(questions_df),
        'sample_size': 5
    }
    
    cache_key = cache_manager.cache_analysis_results(
        analysis_id=analysis_id,
        questions_df=questions_df,
        hybrid_results=hybrid_results,
        config=config,
        metadata=metadata
    )
    
    print(f"✅ Analysis cached with key: {cache_key}")
    
    # Test 2: Retrieve pending analyses
    print("\n🔍 Test 2: Retrieving pending analyses...")
    pending = cache_manager.get_pending_analyses()
    print(f"✅ Found {len(pending)} pending analysis(es)")
    
    for key, analysis in pending.items():
        print(f"   - {key}: {analysis['analysis_id']} ({analysis['status']})")
    
    # Test 3: Get specific analysis
    print(f"\n📋 Test 3: Getting specific analysis ({cache_key})...")
    analysis = cache_manager.get_analysis(cache_key)
    if analysis:
        print(f"✅ Retrieved analysis: {analysis['analysis_id']}")
        print(f"   - Questions: {analysis['results']['questions_count']}")
        print(f"   - Topics: {analysis['results']['topics_discovered']}")
        print(f"   - Status: {analysis['status']}")
    else:
        print("❌ Analysis not found")
    
    # Test 4: Approve analysis
    print(f"\n✅ Test 4: Approving analysis...")
    reviewer = "test_user"
    notes = "Test approval - analysis looks good"
    
    approved = cache_manager.approve_analysis(cache_key, reviewer, notes)
    if approved:
        print(f"✅ Analysis approved by {reviewer}")
    else:
        print("❌ Failed to approve analysis")
    
    # Test 5: Check approval status
    print(f"\n🔍 Test 5: Checking approval status...")
    analysis = cache_manager.get_analysis(cache_key)
    if analysis:
        approval = analysis['approval_status']
        print(f"✅ Approval status: {approval['approved']}")
        print(f"   - Reviewed by: {approval['reviewed_by']}")
        print(f"   - Notes: {approval['notes']}")
    
    # Test 6: Load from disk
    print(f"\n💾 Test 6: Testing disk persistence...")
    loaded_count = cache_manager.load_from_disk()
    print(f"✅ Loaded {loaded_count} analyses from disk")
    
    # Test 7: Analysis history
    print(f"\n📚 Test 7: Checking analysis history...")
    history = cache_manager.get_analysis_history()
    print(f"✅ Found {len(history)} items in history")
    
    # Test 8: Clear analysis (cleanup)
    print(f"\n🗑️ Test 8: Cleaning up...")
    cleared = cache_manager.clear_pending_analysis(cache_key)
    if cleared:
        print(f"✅ Cleared analysis {cache_key}")
    else:
        print(f"❌ Failed to clear analysis {cache_key}")
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")


def test_multiple_analyses():
    """Test handling multiple analyses"""
    
    print("\n🔄 Testing multiple analyses workflow...")
    
    cache_manager = StreamlitAnalysisCacheManager()
    
    # Create multiple sample analyses
    for i in range(3):
        questions_df, hybrid_results, config = create_sample_data()
        
        # Modify data to make each analysis unique
        config['sample_size'] = 5 + i
        hybrid_results['processing_stats']['total_processed'] = 5 + i
        
        analysis_id = f"test_analysis_{i+1}"
        
        cache_key = cache_manager.cache_analysis_results(
            analysis_id=analysis_id,
            questions_df=questions_df,
            hybrid_results=hybrid_results,
            config=config,
            metadata={'batch_test': True, 'sequence': i+1}
        )
        
        print(f"   ✅ Created analysis {i+1}: {cache_key}")
    
    # Check pending count
    pending = cache_manager.get_pending_analyses()
    print(f"   📊 Total pending analyses: {len(pending)}")
    
    # Approve some, reject others
    analyses_list = list(pending.items())
    
    if len(analyses_list) >= 2:
        # Approve first
        cache_manager.approve_analysis(analyses_list[0][0], "batch_reviewer", "Approved in batch test")
        print(f"   ✅ Approved: {analyses_list[0][0]}")
        
        # Reject second
        cache_manager.reject_analysis(analyses_list[1][0], "batch_reviewer", "Rejected for testing")
        print(f"   ❌ Rejected: {analyses_list[1][0]}")
    
    # Final status
    pending = cache_manager.get_pending_analyses()
    print(f"   📊 Remaining pending: {len(pending)}")
    
    # Clear all for cleanup
    cleared = cache_manager.clear_all_pending()
    print(f"   🗑️ Cleared {cleared} pending analyses")


if __name__ == "__main__":
    # Run basic workflow test
    test_caching_workflow()
    
    # Run multiple analyses test
    test_multiple_analyses()
    
    print("\n✨ Cache manager testing complete!")
    print("\n💡 To use in Streamlit:")
    print("   1. Run analysis in the app")
    print("   2. Review cached results in 'Cache Management' tab")
    print("   3. Approve to push to database")
    print("   4. Monitor analysis history")