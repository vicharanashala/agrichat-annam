import os
from typing import List, Dict, Optional

from fast_response_handler import FastResponseHandler

from crewai import Crew
from crew_agents import (
    Retriever_Agent, Grader_agent,
    hallucination_grader, answer_grader
)
from crew_tasks import (
    retriever_task, grader_task,
    hallucination_task, answer_task
)

USE_FAST_MODE = os.getenv("USE_FAST_MODE", "true").lower() == "true"

if os.path.exists("/app"):
    chroma_path = "/app/Agentic_RAG/Knowledge_base/chromaDb"
    environment = "Docker"
else:
    chroma_path = "/home/ubuntu/agrichat-annam/agrichat-backend/Agentic_RAG/Knowledge_base/chromaDb" 
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
    print(f"[CONFIG] CrewAI Mode ENABLED - Using multi-agent workflow")

def get_answer(question: str, conversation_history_param: Optional[List[Dict]] = None, user_state: str = None) -> str:
    """
    Get answer using either Fast Mode or CrewAI Mode
    
    Args:
        question: User's agricultural question
        conversation_history_param: Previous conversation for context
        user_state: User's state/region (default: India)
    
    Returns:
        Generated response
    """
    global conversation_history
    
    if conversation_history_param is None:
        conversation_history_param = conversation_history
    
    if USE_FAST_MODE and fast_handler:
        result = fast_handler.get_answer(question, conversation_history_param, user_state or "India")
    else:
        rag_crew = Crew(
            agents=[Retriever_Agent],
            tasks=[retriever_task],
            verbose=True,
        )
        
        inputs = {
            "question": question,
            "conversation_history": conversation_history_param
        }
        result = rag_crew.kickoff(inputs=inputs)
    
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
        answer = get_answer(question)
        print(f"\n{answer}")
        print(f"\n[Conversation history length: {len(conversation_history)}]")
