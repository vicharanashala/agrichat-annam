import os
from typing import List, Dict, Optional

from fast_response_handler import FastResponseHandler
from tools import FallbackAgriTool

USE_FAST_MODE = os.getenv("USE_FAST_MODE", "true").lower() == "true"

if os.path.exists("/app"):
    chroma_path = "/app/chromaDb"
    environment = "Docker"
else:
    chroma_path = "/home/ubuntu/agrichat-annam/agrichat-backend/chromaDb" 
    environment = "Local"

print(f"[CONFIG] Environment: {environment}")
print(f"[CONFIG] ChromaDB path: {chroma_path}")
print(f"[CONFIG] ChromaDB exists: {os.path.exists(chroma_path)}")

conversation_history = []

fast_handler = None
if USE_FAST_MODE:
    try:
        fast_handler = FastResponseHandler()
        print(f"[CONFIG] Fast Mode ENABLED - Using FastResponseHandler for optimal performance")
    except Exception as e:
        print(f"[CONFIG] Fast Mode initialization failed: {e}")
        print(f"[CONFIG] Falling back to CrewAI Mode")
        USE_FAST_MODE = False
    else:
        print(f"[CONFIG] Fallback Tool Mode ENABLED - Using single-tool fallback pipeline")

def get_answer(question: str, conversation_history_param: Optional[List[Dict]] = None, user_state: str = None, db_config=None) -> str:
    """
    Get answer using either Fast Mode or CrewAI Mode
    
    Args:
        question: User's agricultural question
        conversation_history_param: Previous conversation for context
        user_state: User's state/region (default: India)
        db_config: Database configuration for toggles
    
    Returns:
        Generated response
    """
    global conversation_history
    
    if conversation_history_param is None:
        conversation_history_param = conversation_history
    
    if USE_FAST_MODE and fast_handler:
        result = fast_handler.get_answer(question, conversation_history_param, user_state or "India", db_config)
    else:
        try:
            fallback_tool = FallbackAgriTool()
            tool_result = fallback_tool.run({'question': question})
            result = tool_result.get('answer', '__FALLBACK__')
        except Exception as e:
            print(f"[MAIN] Fallback tool failed: {e}")
            result = "__FALLBACK__"
    
    conversation_history.append({
        "question": question,
        "answer": str(result)
    })
    
    if len(conversation_history) > 5:
        conversation_history.pop(0)
    
    return result

if __name__ == "__main__":
    mode_name = "Fast Mode (50% Faster)" if USE_FAST_MODE else "CrewAI Mode (Multi-Agent)"
    print(f"AgriChat Local Test - {mode_name}")
    print("=" * 60)
    print(f"Environment: {environment}")
    print(f"ChromaDB Path: {chroma_path}")
    print(f"ChromaDB Available: {os.path.exists(chroma_path)}")
    print(f"Configuration: USE_FAST_MODE = {USE_FAST_MODE}")
    print(f"Handler: {'FastResponseHandler' if USE_FAST_MODE else 'CrewAI Workflow'}")
    print("Chain of Thought: Enabled")
    print("=" * 60)
    
    if not os.path.exists(chroma_path):
        print(f"WARNING: ChromaDB not found at {chroma_path}")
        print("   Make sure the database is built and accessible")
        print("=" * 60)
    
    while True:
        question = input("\nAsk your agricultural question or type 'exit' to quit: ")
        if question.strip().lower() == "exit":
            print("Thank you for using AgriChat!")
            break
        
        print(f"\n[Processing with {mode_name}...]")
        result = get_answer(question)
        if isinstance(result, dict):
            answer_text = result.get('answer', '')
            print("\n--- Full Response Preview ---")
            print(answer_text)

            research = result.get('research_data') or []
            if research:
                print("\n--- Research Data ---")
                for i, doc in enumerate(research, start=1):
                    content = doc.get('content') or doc.get('content_preview') or ''
                    if not content:
                        content = (doc.get('full_content') or doc.get('page_content') or '')
                    print(f"[{i}] Source: {doc.get('source', doc.get('collection_type', 'unknown'))}")
                    print(f"Content: {content}\n")
                    if doc.get('metadata'):
                        meta_preview = ', '.join([f"{k}={v}" for k, v in list(doc.get('metadata').items())[:10]])
                        print(f"Metadata: {meta_preview}")
                    print('-' * 40)

            reasoning = result.get('reasoning_steps') or []
            if reasoning:
                print("\n--- Reasoning Steps ---")
                for step in reasoning:
                    print(f"- {step}")

            src = result.get('source')
            sim = result.get('similarity') or result.get('confidence') or 0.0
            meta = result.get('metadata') or {}
            footer_lines = []
            if src:
                footer_lines.append(f"Source: {src}")
            footer_lines.append(f"Confidence/Similarity: {sim:.2f}")
            if meta:
                meta_preview = ', '.join([f"{k}={v}" for k, v in list(meta.items())[:5]])
                footer_lines.append(f"Metadata: {meta_preview}")

            print("\n--- Sources ---")
            for line in footer_lines:
                print(line)
        else:
            print(f"\n{result}")
        print(f"\n[Conversation history length: {len(conversation_history)}]")