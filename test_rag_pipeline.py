#!/usr/bin/env python3
"""
Test script for the improved RAG pipeline with database-first approach
"""

import requests
import json

def test_rag_pipeline():
    """Test the RAG pipeline with various types of questions"""
    
    base_url = "http://localhost:8000"
    
    # Test questions covering different scenarios
    test_cases = [
        {
            "question": "How to treat leaf curl disease in tomatoes?",
            "expected": "database_first",
            "description": "Agricultural disease query - should find database match"
        },
        {
            "question": "What fertilizer is best for rice cultivation?",
            "expected": "database_first", 
            "description": "Agricultural fertilizer query - should find database match"
        },
        {
            "question": "How to grow dragon fruit in space?",
            "expected": "fallback",
            "description": "Unusual agricultural query - should fallback to LLM"
        },
        {
            "question": "Hello, how are you?",
            "expected": "greeting",
            "description": "Greeting - should be handled directly"
        },
        {
            "question": "Who is the president of USA?",
            "expected": "non_agri",
            "description": "Non-agricultural query - should be politely declined"
        }
    ]
    
    print("🌱 Testing Improved RAG Pipeline")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Question: {test_case['question']}")
        print("-" * 30)
        
        try:
            # Make API call
            response = requests.post(
                f"{base_url}/api/query",
                data={
                    "question": test_case["question"],
                    "device_id": "test-device-123",
                    "state": "Haryana",
                    "language": "English"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "No answer received")
                print(f"✅ Response received")
                print(f"Answer: {answer[:200]}...")
                
                # Analyze response type
                if "fallback" in answer.lower() or "limited information" in answer.lower():
                    print("🔄 Type: Fallback/LLM response")
                elif "hello" in test_case["question"].lower() and ("welcome" in answer.lower() or "farming" in answer.lower()):
                    print("👋 Type: Greeting response")
                elif "specialize" in answer.lower() and "agricultural" in answer.lower():
                    print("🚫 Type: Non-agricultural decline")
                else:
                    print("📚 Type: Database/RAG response")
                    
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()

if __name__ == "__main__":
    test_rag_pipeline()
