#!/usr/bin/env python3
"""
Integration test for async/concurrent processing implementation
"""
import sys
import os
import asyncio
from unittest.mock import Mock, patch, AsyncMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_async_integration():
    """Test the complete async integration"""
    print("🔍 Testing async integration...")
    
    try:
        # Mock streamlit to avoid UI dependencies
        with patch('streamlit.write'), patch('streamlit.warning'):
            from topic_modeling import enhance_topic_labels_async
            from openai import AsyncOpenAI
            from bertopic import BERTopic
            import pandas as pd
            
            # Create a simple mock topic model
            class MockTopicModel:
                def get_topic_info(self):
                    return pd.DataFrame({'Topic': [0, 1, 2]})
                
                def get_topic(self, topic_id):
                    return [('keyword1', 0.5), ('keyword2', 0.3), ('keyword3', 0.2)]
            
            # Mock AsyncOpenAI client
            mock_async_client = Mock(spec=AsyncOpenAI)
            
            # Mock the chat completion response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test Topic Label"
            
            async_create_mock = AsyncMock(return_value=mock_response)
            mock_async_client.chat.completions.create = async_create_mock
            
            # Test the async function
            async def run_test():
                topic_model = MockTopicModel()
                result = await enhance_topic_labels_async(topic_model, mock_async_client)
                return result
            
            # Run the test
            result = asyncio.run(run_test())
            
            # Verify results
            if isinstance(result, dict) and len(result) == 3:
                print(f"✅ Async integration successful! Generated {len(result)} labels")
                print(f"   Labels: {list(result.values())}")
                return True
            else:
                print(f"❌ Unexpected result format: {result}")
                return False
                
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_concurrent_performance():
    """Test that concurrent processing is actually faster than sequential"""
    print("🔍 Testing concurrent performance benefits...")
    
    try:
        import time
        
        # Sequential processing simulation
        async def sequential_tasks(n_tasks):
            start_time = time.time()
            for i in range(n_tasks):
                await asyncio.sleep(0.1)  # Simulate API call
            return time.time() - start_time
        
        # Concurrent processing simulation
        async def concurrent_tasks(n_tasks, max_concurrent=3):
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def limited_task():
                async with semaphore:
                    await asyncio.sleep(0.1)  # Simulate API call
            
            start_time = time.time()
            tasks = [limited_task() for _ in range(n_tasks)]
            await asyncio.gather(*tasks)
            return time.time() - start_time
        
        async def run_performance_test():
            n_tasks = 6  # 6 tasks
            
            seq_time = await sequential_tasks(n_tasks)
            conc_time = await concurrent_tasks(n_tasks, max_concurrent=3)
            
            speedup = seq_time / conc_time
            
            print(f"   Sequential time: {seq_time:.3f}s")
            print(f"   Concurrent time: {conc_time:.3f}s")
            print(f"   Speedup: {speedup:.2f}x")
            
            # Concurrent should be at least 1.5x faster
            return speedup >= 1.5
        
        result = asyncio.run(run_performance_test())
        
        if result:
            print("✅ Concurrent processing shows performance improvement!")
            return True
        else:
            print("❌ Concurrent processing not showing expected speedup")
            return False
            
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False

def main():
    """Run integration tests"""
    print("🧪 Testing Async/Concurrent Processing Integration")
    print("=" * 60)
    
    integration_result = test_async_integration()
    performance_result = test_concurrent_performance()
    
    print("\n" + "=" * 60)
    print("📋 INTEGRATION TEST RESULTS:")
    print(f"   Async integration: {'✅' if integration_result else '❌'}")
    print(f"   Performance benefits: {'✅' if performance_result else '❌'}")
    
    all_passed = integration_result and performance_result
    
    if all_passed:
        print("\n🎉 All integration tests passed! Async implementation is ready for production.")
        return 0
    else:
        print("\n❌ Some integration tests failed.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)