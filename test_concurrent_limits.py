#!/usr/bin/env python3
"""
Test script to determine the maximum concurrent requests limit for OpenAI API
"""
import sys
import os
import asyncio
import time
from unittest.mock import Mock, AsyncMock
import aiohttp

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_concurrent_limit_simulation():
    """Test concurrent request limits with simulated API calls"""
    print("🔍 Testing concurrent request limits (simulated)...")
    
    async def simulate_api_call(request_id, delay=0.2):
        """Simulate an OpenAI API call with typical latency"""
        await asyncio.sleep(delay)  # Typical API response time
        return f"Response {request_id}"
    
    # Test different concurrency levels
    concurrency_levels = [1, 3, 5, 8, 10, 15, 20, 25, 30]
    results = {}
    
    for max_concurrent in concurrency_levels:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_call(request_id):
            async with semaphore:
                return await simulate_api_call(request_id)
        
        # Test with 20 total requests
        n_requests = 20
        start_time = time.time()
        
        tasks = [limited_call(i) for i in range(n_requests)]
        responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = n_requests / duration
        
        results[max_concurrent] = {
            'duration': duration,
            'throughput': throughput,
            'responses': len(responses)
        }
        
        print(f"   Concurrency {max_concurrent:2d}: {duration:.2f}s, {throughput:.1f} req/s")
    
    # Find optimal concurrency (point of diminishing returns)
    max_throughput = 0
    optimal_concurrency = 1
    
    for concurrency, data in results.items():
        if data['throughput'] > max_throughput:
            max_throughput = data['throughput']
            optimal_concurrency = concurrency
        
        # Check for diminishing returns (less than 10% improvement)
        if concurrency > 5:
            prev_throughput = results[concurrency - 2]['throughput'] if concurrency - 2 in results else 0
            improvement = (data['throughput'] - prev_throughput) / prev_throughput if prev_throughput > 0 else 0
            
            if improvement < 0.1:  # Less than 10% improvement
                print(f"   ⚠️  Diminishing returns detected at concurrency {concurrency}")
                break
    
    print(f"\n📊 Simulation Results:")
    print(f"   Optimal concurrency: {optimal_concurrency}")
    print(f"   Max throughput: {max_throughput:.1f} requests/second")
    
    return optimal_concurrency

async def test_openai_rate_limits():
    """Test actual OpenAI rate limits (requires API key)"""
    print("🔍 Testing actual OpenAI rate limits...")
    
    try:
        from config import OPENAI_API_KEY
        from openai import AsyncOpenAI
        
        if not OPENAI_API_KEY or "your-openai-api-key" in OPENAI_API_KEY.lower():
            print("   ⚠️  No valid OpenAI API key found - skipping real API test")
            return None
        
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Test with a small, fast API call
        async def test_api_call(request_id):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",  # Fast, cheap model
                    messages=[{"role": "user", "content": f"Say 'Test {request_id}'"}],
                    max_completion_tokens=10
                )
                return f"Success: {response.choices[0].message.content}"
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    return f"Rate limited: {request_id}"
                else:
                    return f"Error: {e}"
        
        # Test increasing concurrency until we hit rate limits
        for concurrency in [1, 3, 5, 8, 10, 15]:
            print(f"   Testing concurrency level: {concurrency}")
            
            semaphore = asyncio.Semaphore(concurrency)
            
            async def limited_call(request_id):
                async with semaphore:
                    return await test_api_call(request_id)
            
            # Small batch to avoid excessive API usage
            n_requests = min(concurrency * 2, 10)
            
            start_time = time.time()
            tasks = [limited_call(i) for i in range(n_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Analyze results
            successes = sum(1 for r in results if isinstance(r, str) and "Success" in r)
            rate_limits = sum(1 for r in results if isinstance(r, str) and "rate_limit" in r.lower())
            errors = sum(1 for r in results if isinstance(r, Exception) or ("Error" in str(r) and "Success" not in str(r)))
            
            duration = end_time - start_time
            
            print(f"     ✅ Success: {successes}, ❌ Rate limited: {rate_limits}, 🔥 Errors: {errors}")
            print(f"     Duration: {duration:.2f}s")
            
            if rate_limits > 0:
                print(f"   ⚠️  Rate limit hit at concurrency {concurrency}")
                safe_concurrency = max(1, concurrency - 1)
                print(f"   💡 Recommended safe concurrency: {safe_concurrency}")
                return safe_concurrency
            
            # Small delay between tests to avoid rate limiting
            await asyncio.sleep(2)
        
        print("   ✅ No rate limits detected up to concurrency 15")
        return 15
        
    except ImportError:
        print("   ⚠️  OpenAI library not available - skipping real API test")
        return None
    except Exception as e:
        print(f"   ❌ API test error: {e}")
        return None

def test_openai_documentation_limits():
    """Reference OpenAI's documented rate limits"""
    print("🔍 OpenAI Documentation Rate Limits:")
    
    limits = {
        "Free Tier": {
            "RPM": 3,  # Requests per minute
            "TPM": 40000,  # Tokens per minute
            "Safe Concurrency": 1
        },
        "Pay-as-you-go (Tier 1)": {
            "RPM": 500,
            "TPM": 40000,
            "Safe Concurrency": 5
        },
        "Pay-as-you-go (Tier 2)": {
            "RPM": 5000,
            "TPM": 450000,
            "Safe Concurrency": 15
        },
        "Pay-as-you-go (Tier 3)": {
            "RPM": 5000,
            "TPM": 1000000,
            "Safe Concurrency": 15
        },
        "Pay-as-you-go (Tier 4)": {
            "RPM": 10000,
            "TPM": 2000000,
            "Safe Concurrency": 25
        }
    }
    
    print("\n📋 OpenAI Rate Limits by Tier:")
    for tier, limit in limits.items():
        rpm = limit["RPM"]
        safe_concurrent = limit["Safe Concurrency"]
        print(f"   {tier}:")
        print(f"     • {rpm:,} requests/minute")
        print(f"     • Safe concurrency: {safe_concurrent}")
    
    print(f"\n💡 Recommendations:")
    print(f"   • Start with 3-5 concurrent requests (safe for most tiers)")
    print(f"   • Monitor for 429 rate limit errors")
    print(f"   • Increase gradually based on your tier limits")
    print(f"   • Use exponential backoff for rate limit retries")

async def run_all_tests():
    """Run all concurrent limit tests"""
    print("🧪 Testing Maximum Concurrent Requests Limits")
    print("=" * 60)
    
    # Test 1: Simulation
    optimal_simulated = await test_concurrent_limit_simulation()
    
    print("\n" + "=" * 60)
    # Test 2: Documentation
    test_openai_documentation_limits()
    
    print("\n" + "=" * 60)
    # Test 3: Real API (if available)
    real_limit = await test_openai_rate_limits()
    
    print("\n" + "=" * 60)
    print("📊 FINAL RECOMMENDATIONS:")
    print(f"   Simulated optimal: {optimal_simulated}")
    if real_limit:
        print(f"   Real API tested limit: {real_limit}")
        recommended = min(optimal_simulated, real_limit)
    else:
        recommended = min(optimal_simulated, 5)  # Conservative default
    
    print(f"   🎯 RECOMMENDED: {recommended} concurrent requests")
    print(f"   📝 Current setting: 5 (good balance)")
    
    if recommended != 5:
        print(f"\n💡 Consider updating your configuration:")
        print(f"   MAX_CONCURRENT_REQUESTS={recommended}")
    else:
        print(f"\n✅ Current setting of 5 is optimal!")

def main():
    """Run the test"""
    return asyncio.run(run_all_tests())

if __name__ == "__main__":
    main()