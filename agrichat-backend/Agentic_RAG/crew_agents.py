from tools import RAGTool, FallbackAgriTool
import os
from dotenv import load_dotenv
from crewai import LLM, Agent
from typing import List, Dict, Optional
load_dotenv()


import os
current_dir = os.path.dirname(os.path.abspath(__file__))
chroma_path = os.path.join(current_dir, "chromaDb")
print(f"[DEBUG] Using ChromaDB path: {chroma_path}")
print(f"[DEBUG] ChromaDB path exists: {os.path.exists(chroma_path)}")

rag_tool = RAGTool(chroma_path=chroma_path)

from crewai import LLM, Agent
llm = LLM(
    model=f"ollama/{os.getenv('OLLAMA_MODEL', 'llama3.1-optimized')}",
    base_url=f"http://{os.getenv('OLLAMA_HOST', 'localhost:11434')}",
    api_key="not-needed",
    temperature=0.0
)

fallback_tool = FallbackAgriTool()

Retriever_Agent = Agent(
    role="Retriever Agent",
    goal=(
        "Route the user's question to the appropriate tool."
        "Use the RAG tool first to answer agricultural queries. "
        "If the RAG tool returns '__FALLBACK__' or cannot answer confidently, "
        "then invoke the fallback tool (LLM plus web search). "
        "For non-agricultural queries, the tools will respond politely declining."
        " For normal greetings or salutations (e.g., “hello,” “how are you?”, “good morning”), respond gently and politely with a soft, appropriate answer."
    ),
    backstory=(
        "You do not answer questions yourself. Instead, you decide which tool to call based on the user's query and the tool's responses. "
        "You prioritize using trusted internal knowledge (RAG tool) before falling back to external web search aided responses."
    ),
    verbose=True,
    llm=llm,
    tools=[rag_tool, fallback_tool],
)

def retriever_response(question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = None) -> str:
    """
    Process question with improved database-first approach and fallback handling
    
    Args:
        question: Current user question
        conversation_history: List of previous Q&A pairs for context-aware responses
        user_state: User's state/region detected from frontend
        
    Returns:
        Generated response using database-first approach with intelligent fallback
    """
    try:
        question_lower = question.lower().strip()
        simple_greetings = [
            'hi', 'hello', 'hey', 'namaste', 'namaskaram', 'vanakkam', 
            'good morning', 'good afternoon', 'good evening', 'good day',
            'howdy', 'greetings', 'salaam', 'adaab'
        ]
        
        if len(question_lower) < 20 and any(greeting in question_lower for greeting in simple_greetings):
            print(f"[FAST GREETING] Detected simple greeting: {question}")
            print(f"[SOURCE] Fast pattern matching used for greeting: {question}")
            if 'namaste' in question_lower:
                fast_response = "Namaste! Welcome to AgriChat. I'm here to help you with all your farming and agriculture questions. What would you like to know about?"
            elif 'namaskaram' in question_lower:
                fast_response = "Namaskaram! I'm your agricultural assistant. Feel free to ask me anything about crops, farming techniques, or agricultural practices."
            elif 'vanakkam' in question_lower:
                fast_response = "Vanakkam! I'm here to assist you with farming and agriculture. What agricultural topic would you like to discuss today?"
            elif any(time in question_lower for time in ['morning', 'afternoon', 'evening']):
                fast_response = f"Good {question_lower.split()[-1]}! I'm your agricultural assistant. How can I help you with your farming questions today?"
            else:
                fast_response = "Hello! I'm your agricultural assistant. I'm here to help with farming, crops, and agricultural practices. What would you like to know?"
            
            return fast_response
        
        rag_response = rag_tool._run(question, conversation_history, user_state)
        
        if rag_response == "__FALLBACK__":
            print(f"[DEBUG] RAG tool requested fallback for question: {question}")
            fallback_response = fallback_tool._run(question, conversation_history)
            return fallback_response
        
        return rag_response
        
    except Exception as e:
        print(f"[ERROR] Error in retriever_response: {e}")
        try:
            return fallback_tool._run(question, conversation_history)
        except Exception as fallback_error:
            print(f"[ERROR] Fallback tool also failed: {fallback_error}")
            return "I'm having trouble processing your question right now. Please try again or rephrase your question."

Grader_agent = Agent(
    role='Answer Grader',
    goal='Filter out erroneous retrievals',
    backstory=(
        "You are a grader assessing relevance of a retrieved document to a user question."
        "If the document contains keywords related to the user question, grade it as relevant."
        "It does not need to be a stringent test. You have to make sure that the answer is relevant to the question."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

hallucination_grader = Agent(
    role="Hallucination Grader",
    goal="Filter out hallucination",
    backstory=(
        "You are a hallucination grader assessing whether an answer is grounded in / supported by a set of facts."
        "Make sure you meticulously review the answer and check if the response provided is in alignment with the question asked"
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

answer_grader = Agent(
    role="Answer Grader",
    goal="Filter out hallucination from the answer.",
    backstory=(
        "You are a grader assessing whether an answer is useful to resolve a question."
        "Make sure you meticulously review the answer and check if it makes sense for the question asked."
        "If the answer is relevant generate a clear and concise response."
        "If the answer generated is not relevant then perform a websearch using 'fallback_tool'."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[fallback_tool],
)
