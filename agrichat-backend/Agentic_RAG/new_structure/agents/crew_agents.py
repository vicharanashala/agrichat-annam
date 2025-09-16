"""
Agricultural AI Agents for CrewAI-based RAG System.

This module contains specialized agents for Indian agricultural knowledge processing,
designed to work with agricultural RAG tools and provide India-specific farming guidance.
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from crewai import LLM, Agent

from core.config import ConfigManager
from tools.rag_tool import AgriRAGTool
from tools.fallback_tool import AgriFallbackTool


class AgriAgentsManager:
    """
    Manager class for agricultural AI agents in the CrewAI system.
    
    This class handles initialization and configuration of specialized agricultural
    agents designed for Indian farming contexts and regional variations.
    
    Features:
    - Specialized Indian agricultural agents
    - Configurable LLM integration with Ollama
    - Tool routing and response coordination
    - Quality assurance through grading agents
    - Context-aware regional adaptation
    """
    
    def __init__(self):
        """Initialize the agricultural agents manager."""
        load_dotenv()
        
        # Initialize configuration
        self.config = ConfigManager()
        
        # Initialize tools
        self._initialize_tools()
        
        # Initialize LLM
        self._initialize_llm()
        
        # Initialize agents
        self._initialize_agents()
    
    def _initialize_tools(self) -> None:
        """Initialize agricultural RAG and fallback tools."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Determine ChromaDB path based on environment
        if os.path.exists("/app"):
            chroma_path = "/app/chromaDb"
        else:
            chroma_path = "/home/ubuntu/agrichat-annam/agrichat-backend/chromaDb"
        
        print(f"[DEBUG] Using ChromaDB path: {chroma_path}")
        print(f"[DEBUG] ChromaDB path exists: {os.path.exists(chroma_path)}")
        
        # Initialize tools
        self.rag_tool = AgriRAGTool(chroma_path=chroma_path)
        self.fallback_tool = AgriFallbackTool()
    
    def _initialize_llm(self) -> None:
        """Initialize the language model for agent operations."""
        self.llm = LLM(
            model=f"ollama/{os.getenv('OLLAMA_MODEL', 'llama3.1:latest')}",
            base_url=f"http://{os.getenv('OLLAMA_HOST', 'localhost:11434')}",
            api_key="not-needed",
            temperature=0.0
        )
    
    def _initialize_agents(self) -> None:
        """Initialize all agricultural agents."""
        self.retriever_agent = self._create_retriever_agent()
        self.grader_agent = self._create_grader_agent()
        self.hallucination_grader = self._create_hallucination_grader()
        self.answer_grader = self._create_answer_grader()
    
    def _create_retriever_agent(self) -> Agent:
        """
        Create the main retriever agent for agricultural queries.
        
        Returns:
            Agent: Configured retriever agent for Indian agriculture
        """
        return Agent(
            role="Indian Agricultural Retriever Agent",
            goal=(
                "Route the user's agricultural question to appropriate tools, focusing exclusively on Indian agricultural context and regional conditions. "
                "Use the RAG tool first to answer agricultural queries specific to Indian farming, crop varieties, soil conditions, and climate patterns. "
                "If the RAG tool returns '__FALLBACK__' or cannot answer confidently, "
                "then invoke the fallback tool (LLM with Indian agricultural knowledge). "
                "All responses must be specific to Indian agriculture, Indian regions, and Indian farming practices. "
                "For non-agricultural queries, respond politely declining while maintaining Indian agricultural focus. "
                "For normal greetings or salutations (e.g., 'hello,' 'how are you?', 'good morning'), respond gently and politely with a soft, appropriate answer that includes Indian agricultural context."
            ),
            backstory=(
                "You are a specialized Indian agricultural assistant with deep knowledge of Indian farming systems, regional variations, monsoon patterns, and crop varieties suited to Indian conditions. "
                "You do not answer questions yourself but route them to tools that provide India-specific agricultural guidance. "
                "You prioritize using trusted internal knowledge (RAG tool with Indian agricultural data) before falling back to Indian agricultural LLM responses. "
                "All your tool selections should result in advice tailored to Indian farmers, Indian soil types, Indian climate conditions, and Indian agricultural markets."
            ),
            verbose=True,
            llm=self.llm,
            tools=[self.rag_tool, self.fallback_tool],
        )
    
    def _create_grader_agent(self) -> Agent:
        """
        Create the grader agent for document relevance assessment.
        
        Returns:
            Agent: Configured grader agent for Indian agricultural context
        """
        return Agent(
            role='Indian Agricultural Answer Grader',
            goal='Filter out erroneous retrievals and ensure India-specific agricultural relevance',
            backstory=(
                "You are a grader specialized in Indian agriculture, assessing relevance of retrieved documents to user questions about Indian farming. "
                "You evaluate whether documents contain information relevant to Indian agricultural practices, Indian crop varieties, Indian soil conditions, and Indian climate patterns. "
                "If the document contains keywords and context related to Indian agriculture and the user question, grade it as relevant. "
                "Prioritize information that is applicable to Indian farming conditions, regional variations within India, and Indian agricultural markets. "
                "It does not need to be a stringent test. You have to make sure that the answer is relevant to the question."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )
    
    def _create_hallucination_grader(self) -> Agent:
        """
        Create the hallucination grader agent for fact verification.
        
        Returns:
            Agent: Configured hallucination grader for Indian agricultural facts
        """
        return Agent(
            role="Indian Agricultural Hallucination Grader",
            goal="Filter out hallucination and ensure responses are grounded in Indian agricultural facts",
            backstory=(
                "You are a hallucination grader specializing in Indian agriculture, assessing whether an answer about Indian farming is grounded in and supported by factual Indian agricultural information. "
                "You ensure that responses about Indian crops, farming practices, soil conditions, and agricultural techniques are accurate and applicable to Indian conditions. "
                "Make sure you meticulously review answers for Indian agricultural accuracy and check if the response provided is aligned with the question asked and applicable to Indian farming contexts."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )
    
    def _create_answer_grader(self) -> Agent:
        """
        Create the answer grader agent for response quality assessment.
        
        Returns:
            Agent: Configured answer grader with fallback capabilities
        """
        return Agent(
            role="Indian Agricultural Answer Grader",
            goal="Filter out hallucination and ensure answers are useful for Indian agricultural questions.",
            backstory=(
                "You are a grader specialized in Indian agriculture, assessing whether an answer is useful to resolve Indian farming questions and applicable to Indian agricultural conditions. "
                "You evaluate responses for their relevance to Indian farming practices, crop varieties, soil management, and regional agricultural conditions across different Indian states. "
                "Make sure you meticulously review answers for Indian agricultural accuracy and check if they make sense for the Indian farming context of the question asked. "
                "If the answer is relevant to Indian agriculture, generate a clear and concise response that emphasizes Indian-specific applications. "
                "If the answer generated is not relevant to Indian farming conditions, then perform a fallback search using 'fallback_tool' to find India-specific agricultural information."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.fallback_tool],
        )
    
    def retriever_response(self, 
                          question: str, 
                          conversation_history: Optional[List[Dict]] = None, 
                          user_state: str = None) -> str:
        """
        Process agricultural questions with database-first approach and intelligent fallback.
        
        This method implements a sophisticated response generation strategy:
        1. Fast greeting detection for common salutations
        2. Primary RAG tool processing for agricultural queries
        3. Intelligent fallback to LLM when RAG cannot provide answers
        4. Regional adaptation based on user state
        
        Args:
            question: Current user question about agriculture
            conversation_history: List of previous Q&A pairs for context
            user_state: User's state/region for regional adaptation
            
        Returns:
            str: Generated response optimized for Indian agricultural context
        """
        try:
            question_lower = question.lower().strip()
            
            # Fast greeting detection for common Indian greetings
            simple_greetings = [
                'hi', 'hello', 'hey', 'namaste', 'namaskaram', 'vanakkam', 
                'good morning', 'good afternoon', 'good evening', 'good day',
                'howdy', 'greetings', 'salaam', 'adaab'
            ]
            
            # Handle simple greetings with fast response
            if len(question_lower) < 20 and any(greeting in question_lower for greeting in simple_greetings):
                print(f"[FAST GREETING] Detected simple greeting: {question}")
                print(f"[SOURCE] Fast pattern matching used for greeting: {question}")
                
                return self._generate_greeting_response(question_lower)
            
            # Primary processing with RAG tool
            print(f"[DEBUG] Processing agricultural query with RAG tool: {question}")
            rag_response = self.rag_tool.process_query(question, conversation_history, user_state)
            
            # Handle fallback if RAG tool cannot provide answer
            if rag_response == "__FALLBACK__":
                print(f"[DEBUG] RAG tool requested fallback for question: {question}")
                fallback_response = self.fallback_tool.process_query(question, conversation_history)
                print(f"[SOURCE] Fallback LLM used for: {question}")
                return fallback_response
            
            print(f"[SOURCE] RAG database used for: {question}")
            return rag_response
            
        except Exception as e:
            print(f"[ERROR] Error in retriever_response: {e}")
            try:
                return self.fallback_tool.process_query(question, conversation_history)
            except Exception as fallback_error:
                print(f"[ERROR] Fallback tool also failed: {fallback_error}")
                return "I'm having trouble processing your question right now. Please try again or rephrase your question."
    
    def _generate_greeting_response(self, question_lower: str) -> str:
        """
        Generate appropriate greeting responses for different Indian greetings.
        
        Args:
            question_lower: Lowercase version of the greeting question
            
        Returns:
            str: Culturally appropriate greeting response
        """
        if 'namaste' in question_lower:
            return "Namaste! Welcome to AgriChat. I'm here to help you with all your farming and agriculture questions. What would you like to know about?"
        elif 'namaskaram' in question_lower:
            return "Namaskaram! I'm your agricultural assistant. Feel free to ask me anything about crops, farming techniques, or agricultural practices."
        elif 'vanakkam' in question_lower:
            return "Vanakkam! I'm here to assist you with farming and agriculture. What agricultural topic would you like to discuss today?"
        elif any(time in question_lower for time in ['morning', 'afternoon', 'evening']):
            time_period = next(time for time in ['morning', 'afternoon', 'evening'] if time in question_lower)
            return f"Good {time_period}! I'm your agricultural assistant. How can I help you with your farming questions today?"
        else:
            return "Hello! I'm your agricultural assistant. I'm here to help with farming, crops, and agricultural practices. What would you like to know?"
    
    def get_agents(self) -> Dict[str, Agent]:
        """
        Get all initialized agents.
        
        Returns:
            Dict[str, Agent]: Dictionary of agent names and instances
        """
        return {
            'retriever': self.retriever_agent,
            'grader': self.grader_agent,
            'hallucination_grader': self.hallucination_grader,
            'answer_grader': self.answer_grader
        }
    
    def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on all agents and tools.
        
        Returns:
            Dict[str, bool]: Health status of components
        """
        health_status = {}
        
        try:
            # Check tools
            health_status['rag_tool'] = self.rag_tool.health_check()
            health_status['fallback_tool'] = self.fallback_tool.health_check()
            
            # Check agents (basic validation)
            health_status['retriever_agent'] = hasattr(self.retriever_agent, 'role')
            health_status['grader_agent'] = hasattr(self.grader_agent, 'role')
            health_status['hallucination_grader'] = hasattr(self.hallucination_grader, 'role')
            health_status['answer_grader'] = hasattr(self.answer_grader, 'role')
            
            # Check LLM configuration
            health_status['llm'] = hasattr(self.llm, 'model')
            
        except Exception as e:
            print(f"[ERROR] Health check failed: {e}")
            health_status['error'] = str(e)
        
        return health_status


# Global agents manager instance for backward compatibility
agents_manager = AgriAgentsManager()

# Legacy exports for backward compatibility
Retriever_Agent = agents_manager.retriever_agent
Grader_agent = agents_manager.grader_agent
hallucination_grader = agents_manager.hallucination_grader
answer_grader = agents_manager.answer_grader
retriever_response = agents_manager.retriever_response


def get_agents_manager() -> AgriAgentsManager:
    """
    Get the global agents manager instance.
    
    Returns:
        AgriAgentsManager: Global agents manager instance
    """
    return agents_manager


if __name__ == "__main__":
    # Test the agents manager
    test_question = "What are the best practices for rice cultivation in Tamil Nadu during the monsoon season?"
    response = agents_manager.retriever_response(
        test_question, 
        conversation_history=[], 
        user_state="Tamil Nadu"
    )
    print(f"Response: {response}")
    
    # Health check
    health = agents_manager.health_check()
    print(f"Health Status: {health}")
