#!/usr/bin/env python3
"""
Test script to validate the crop-specific validation fix
This tests the sugarcane red insect query that was incorrectly matching cotton content
"""

import sys
import os
sys.path.append('/home/ubuntu/agrichat-annam/agrichat-backend/Agentic_RAG')

def test_crop_validation_fix():
    """Test that the crop validation prevents false positives"""
    print("="*60)
    print("TESTING CROP VALIDATION FIX")
    print("="*60)
    
    # Test the problematic query
    test_query = "There's a red insect on my sugarcane which is harming the quality of it. What to do?"
    
    print(f"\n🧪 Testing Query: {test_query}")
    print("\n📋 EXPECTED BEHAVIOR:")
    print("  ❌ Should NOT match cotton/bollworm content")
    print("  ✅ Should fallback to LLM when no sugarcane content available")
    print("  ✅ Cross-crop false positive should be detected and rejected")
    
    try:
        from chroma_query_handler import ChromaQueryHandler
        
        # Initialize the handler
        if os.path.exists("/app"):
            chroma_db_path = "/app/chromaDb"
        else:
            chroma_db_path = "/home/ubuntu/agrichat-annam/agrichat-backend/chromaDb"
            
        handler = ChromaQueryHandler(chroma_path=chroma_db_path)
        
        print(f"\n🔍 Testing with ChromaDB path: {chroma_db_path}")
        
        # Test the query
        result = handler.get_answer_with_source(test_query, [], "", None)
        
        print(f"\n📊 RESULTS:")
        print(f"  Answer: {result['answer'][:150]}...")
        print(f"  Source: {result['source']}")
        print(f"  Cosine Similarity: {result['cosine_similarity']:.4f}")
        print(f"  Metadata: {result.get('document_metadata', {})}")
        
        # Analyze the result
        is_fallback = result['answer'] == "__FALLBACK__" or "I don't have enough information" in result['answer']
        similarity_score = result['cosine_similarity']
        
        print(f"\n🧠 ANALYSIS:")
        if is_fallback:
            print("  ✅ GOOD: System correctly identified insufficient information")
            print("  ✅ GOOD: Proper fallback behavior activated")
        else:
            if similarity_score >= 0.6:
                print(f"  ⚠️  WARNING: High similarity score ({similarity_score:.4f}) detected")
                print("  🔍 Checking if crop validation worked...")
                
                # Check if it's cotton content being returned
                answer_lower = result['answer'].lower()
                if 'cotton' in answer_lower and 'bollworm' in answer_lower:
                    print("  ❌ FAIL: Still returning cotton content for sugarcane query!")
                    print("  ❌ FAIL: Crop validation did not prevent false positive!")
                elif 'sugarcane' in answer_lower:
                    print("  ✅ GOOD: Response contains sugarcane-specific information")
                else:
                    print("  ⚠️  UNCLEAR: Response doesn't clearly indicate crop type")
            else:
                print(f"  ✅ GOOD: Low similarity score ({similarity_score:.4f}) properly handled")
        
        print(f"\n🎯 CONCLUSION:")
        if is_fallback:
            print("  🎉 SUCCESS: Fix is working! Query properly falls back to LLM")
            print("  📈 IMPROVEMENT: Cross-crop false positives are now prevented")
        else:
            print("  🔬 NEEDS REVIEW: Manual verification required for this result")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Could not complete test - check ChromaDB setup and dependencies")

if __name__ == "__main__":
    test_crop_validation_fix()