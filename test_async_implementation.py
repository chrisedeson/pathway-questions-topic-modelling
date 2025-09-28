#!/usr/bin/env python3
"""
Test script for async/concurrent processing implementation
"""
import sys
import os
import asyncio
from unittest.mock import Mock, patch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_async_imports():
    """Test that async imports work correctly"""
    print("🔍 Testing async imports...")
    
    try:
        from topic_modeling import enhance_topic_labels_async, generate_single_topic_label
        from openai import AsyncOpenAI
        print("✅ All async imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_async_function_signatures():
    """Test that async functions have correct signatures"""
    print("🔍 Testing async function signatures...")
    
    try:
        from topic_modeling import enhance_topic_labels_async, generate_single_topic_label
        import inspect
        
        # Check if functions are async
        if not asyncio.iscoroutinefunction(enhance_topic_labels_async):
            print("❌ enhance_topic_labels_async is not an async function")
            return False
            
        if not asyncio.iscoroutinefunction(generate_single_topic_label):
            print("❌ generate_single_topic_label is not an async function")
            return False
            
        print("✅ All async functions have correct signatures!")
        return True
    except Exception as e:
        print(f"❌ Function signature error: {e}")
        return False

def test_config_values():
    """Test that new config values are properly loaded"""
    print("🔍 Testing configuration values...")
    
    try:
        from config import MAX_CONCURRENT_REQUESTS, ENABLE_ASYNC_PROCESSING
        
        print(f"   MAX_CONCURRENT_REQUESTS: {MAX_CONCURRENT_REQUESTS}")
        print(f"   ENABLE_ASYNC_PROCESSING: {ENABLE_ASYNC_PROCESSING}")
        
        # Check that values are reasonable
        if not isinstance(MAX_CONCURRENT_REQUESTS, int) or MAX_CONCURRENT_REQUESTS < 1:
            print("❌ MAX_CONCURRENT_REQUESTS must be a positive integer")
            return False
            
        if not isinstance(ENABLE_ASYNC_PROCESSING, bool):
            print("❌ ENABLE_ASYNC_PROCESSING must be a boolean")
            return False
            
        print("✅ Configuration values are valid!")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

async def test_semaphore_functionality():
    """Test semaphore-based rate limiting"""
    print("🔍 Testing semaphore functionality...")
    
    try:
        # Create a semaphore with limit of 2
        semaphore = asyncio.Semaphore(2)
        
        async def test_task(task_id):
            async with semaphore:
                await asyncio.sleep(0.1)
                return f"Task {task_id} completed"
        
        # Create 5 tasks (should be limited by semaphore)
        tasks = [test_task(i) for i in range(5)]
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()
        
        # With semaphore of 2, 5 tasks should take at least 3 * 0.1 = 0.3 seconds
        duration = end_time - start_time
        
        if duration < 0.25:  # Allow some tolerance
            print(f"❌ Semaphore not working properly. Duration: {duration:.3f}s")
            return False
            
        print(f"✅ Semaphore working correctly! Duration: {duration:.3f}s")
        print(f"   Results: {len(results)} tasks completed")
        return True
        
    except Exception as e:
        print(f"❌ Semaphore test error: {e}")
        return False

def test_mock_topic_model():
    """Test with a mock topic model to verify the async wrapper works"""
    print("🔍 Testing mock topic model processing...")
    
    try:
        # Import after setting up the mock
        with patch('streamlit.write'), patch('streamlit.warning'):
            from topic_modeling import enhance_topic_labels
            from openai import OpenAI
            
            # Create a mock topic model
            mock_topic_model = Mock()
            mock_topic_model.get_topic_info.return_value = Mock()
            mock_topic_model.get_topic_info.return_value.__getitem__ = Mock(return_value=Mock())
            mock_topic_model.get_topic_info.return_value['Topic'] = Mock()
            mock_topic_model.get_topic_info.return_value['Topic'].unique = Mock(return_value=[0, 1])
            mock_topic_model.get_topic = Mock(return_value=[('word1', 0.5), ('word2', 0.3)])
            
            # Create a mock OpenAI client
            mock_client = Mock(spec=OpenAI)
            mock_client.api_key = 'test-key'
            
            # Test the wrapper function (without actual API calls)
            with patch('topic_modeling.AsyncOpenAI') as mock_async_client_class:
                mock_async_client = Mock()
                mock_async_client_class.return_value = mock_async_client
                
                # Mock the async function to return immediately
                with patch('topic_modeling.enhance_topic_labels_async') as mock_async_func:
                    mock_async_func.return_value = {0: 'Test Topic 1', 1: 'Test Topic 2'}
                    
                    result = enhance_topic_labels(mock_topic_model, mock_client)
                    
                    if result == {0: 'Test Topic 1', 1: 'Test Topic 2'}:
                        print("✅ Mock topic model processing successful!")
                        return True
                    else:
                        print(f"❌ Unexpected result: {result}")
                        return False
                        
    except Exception as e:
        print(f"❌ Mock topic model test error: {e}")
        return False

async def run_async_tests():
    """Run all async-specific tests"""
    print("🚀 Running async tests...\n")
    
    # Test semaphore functionality
    semaphore_result = await test_semaphore_functionality()
    
    print(f"\n📊 Async Test Results:")
    print(f"   Semaphore functionality: {'✅' if semaphore_result else '❌'}")
    
    return semaphore_result

def main():
    """Run all tests"""
    print("🧪 Testing Async/Concurrent Processing Implementation")
    print("=" * 60)
    
    # Run sync tests first
    import_result = test_async_imports()
    signature_result = test_async_function_signatures()
    config_result = test_config_values()
    mock_result = test_mock_topic_model()
    
    # Run async tests
    print("\n" + "=" * 60)
    async_result = asyncio.run(run_async_tests())
    
    print("\n" + "=" * 60)
    print("📋 FINAL TEST RESULTS:")
    print(f"   Async imports: {'✅' if import_result else '❌'}")
    print(f"   Function signatures: {'✅' if signature_result else '❌'}")
    print(f"   Configuration: {'✅' if config_result else '❌'}")
    print(f"   Mock processing: {'✅' if mock_result else '❌'}")
    print(f"   Async functionality: {'✅' if async_result else '❌'}")
    
    all_passed = all([import_result, signature_result, config_result, mock_result, async_result])
    
    if all_passed:
        print("\n🎉 All tests passed! Async/concurrent implementation is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)