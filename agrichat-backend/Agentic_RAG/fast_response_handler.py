"""
Fast Response Handler - Rule-based approach bypassing CrewAI
This provides 10-15x faster responses by using direct tool calls instead of multi-agent workflows
"""

import os
import sys
import logging
from typing import List, Dict, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from tools import RAGTool, FallbackAgriTool

logger = logging.getLogger("uvicorn.error")

class FastResponseHandler:
    """
    Rule-based response handler that bypasses CrewAI for faster responses
    Uses direct tool calls with simple decision logic
    """
    
    def __init__(self):
        """Initialize the fast response handler with direct tool access"""
        chroma_db_path = os.path.join(current_dir, "chromaDb")
        logger.info(f"[FAST] Initializing with ChromaDB path: {chroma_db_path}")
        
        self.rag_tool = RAGTool(chroma_path=chroma_db_path)
        self.fallback_tool = FallbackAgriTool()
        
        self.simple_greetings = [
            'hi', 'hello', 'hey', 'namaste', 'namaskaram', 'vanakkam', 
            'good morning', 'good afternoon', 'good evening', 'good day',
            'howdy', 'greetings', 'salaam', 'adaab', 'hi there', 'hello there'
        ]
        
    def is_simple_greeting(self, question: str) -> bool:
        """
        Fast greeting detection without LLM processing
        """
        question_lower = question.lower().strip()
        return (len(question_lower) < 20 and 
                any(greeting in question_lower for greeting in self.simple_greetings))
    
    def get_greeting_response(self, question: str, user_state: str = None) -> str:
        """
        Generate fast greeting responses focused on Indian agriculture
        """
        question_lower = question.lower().strip()
        state_context = f" in {user_state}" if user_state and user_state.lower() != "unknown" else " in India"
        
        if 'namaste' in question_lower:
            return f"Namaste! Welcome to AgriChat, your trusted agricultural assistant for Indian farming{state_context}. I specialize in crop management, soil health, weather patterns, and farming practices specific to Indian conditions. What agricultural challenge can I help you with today?"
        elif 'namaskaram' in question_lower:
            return f"Namaskaram! I'm your specialized agricultural assistant for Indian farmers{state_context}. Feel free to ask me about Indian crop varieties, monsoon farming, soil management, or any farming techniques suited to Indian climate and conditions."
        elif 'vanakkam' in question_lower:
            return f"Vanakkam! I'm here to assist you with Indian farming and agriculture{state_context}. What regional agricultural topic would you like to discuss - from rice cultivation to spice farming, I'm here to help with India-specific guidance!"
        elif any(time in question_lower for time in ['morning', 'afternoon', 'evening']):
            time_word = next(time for time in ['morning', 'afternoon', 'evening'] if time in question_lower)
            return f"Good {time_word}! I'm your agricultural assistant specializing in Indian farming practices{state_context}. How can I help you with your crop management, seasonal farming, or any agriculture-related questions specific to Indian conditions today?"
        else:
            return f"Hello! I'm your agricultural assistant specializing in Indian farming and crop management{state_context}. I'm here to help with crops, farming techniques, and agricultural practices tailored to Indian soil, climate, and regional conditions. What would you like to know?"
    
    def get_answer(self, question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = None) -> str:
        """
        Fast rule-based response generation
        
        Decision Logic:
        1. Check for simple greetings (0.1s) - return immediately
        2. Try RAG tool first (5-8s) - our knowledge base
        3. If RAG fails, use fallback tool (5-8s) - LLM knowledge
        
        Args:
            question: Current user question
            conversation_history: List of previous Q&A pairs for context
            user_state: User's state/region detected from frontend
            
        Returns:
            Generated response using fast rule-based approach
        """
        if conversation_history:
            logger.info(f"[CONTEXT] Conversation history received with {len(conversation_history)} entries:")
            for i, entry in enumerate(conversation_history):
                if isinstance(entry, dict) and 'question' in entry and 'answer' in entry:
                    logger.info(f"[CONTEXT] Entry {i}: Q='{entry['question'][:50]}...', A='{entry['answer'][:50]}...'")
                else:
                    logger.info(f"[CONTEXT] Entry {i}: {type(entry)} - {str(entry)[:100]}...")
        else:
            logger.info(f"[CONTEXT] No conversation history provided")
        
        logger.info(f"[FAST] Processing question: {question[:50]}...")
        
        if self.is_simple_greeting(question):
            logger.info(f"[FAST] Detected greeting, returning fast response")
            return self.get_greeting_response(question, user_state)
        
        try:
            logger.info(f"[FAST] Trying RAG tool first...")
            rag_response = self.rag_tool._run(question, conversation_history, user_state)
            
            if rag_response != "__FALLBACK__":
                logger.info(f"[FAST] RAG tool succeeded")
                return rag_response
            
            logger.info(f"[FAST] RAG tool failed, using fallback...")
            fallback_response = self.fallback_tool._run(question, conversation_history)
            return fallback_response
            
        except Exception as e:
            logger.error(f"[FAST] Error in fast processing: {e}")
            try:
                return self.fallback_tool._run(question, conversation_history)
            except Exception as fallback_error:
                logger.error(f"[FAST] Fallback also failed: {fallback_error}")
                return "I'm having trouble processing your question right now. Please try again or rephrase your question."
    
    def get_performance_stats(self) -> Dict:
        """
        Return performance statistics for monitoring
        """
        return {
            "handler_type": "fast_rule_based",
            "uses_crewai": False,
            "expected_response_time": "5-10 seconds",
            "greeting_response_time": "0.1 seconds"
        }


fast_handler = FastResponseHandler()

def get_fast_answer(question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = None) -> str:
    """
    Convenience function for fast answer generation
    """
    return fast_handler.get_answer(question, conversation_history, user_state)
