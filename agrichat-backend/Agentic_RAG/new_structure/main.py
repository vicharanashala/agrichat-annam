"""
AgriChat Agentic RAG System - Professional Implementation

This is the main entry point for the restructured AgriChat system with proper
object-oriented design, clean architecture, and comprehensive functionality.
"""

import os
import logging
from typing import List, Dict, Optional, Any

from core import (
    AgriSystemInterface, 
    QueryContext, 
    AgriResponse,
    ResponseMode,
    get_config,
    validate_setup,
    get_config_summary
)
from handlers.fast_response_handler import FastResponseHandler

# Import legacy components for CrewAI compatibility
from crewai import Crew
from crew_agents import (
    Retriever_Agent, Grader_agent,
    hallucination_grader, answer_grader
)
from crew_tasks import (
    retriever_task, grader_task,
    hallucination_task, answer_task
)


class AgriChatSystem(AgriSystemInterface):
    """
    Main AgriChat system implementation with dual processing modes.
    
    This class provides the primary interface for agricultural question processing
    with intelligent mode selection between Fast Response and CrewAI workflows.
    
    Features:
    - Automatic mode selection based on query complexity
    - Fallback mechanisms for robust operation
    - Performance monitoring and optimization
    - Comprehensive error handling
    - Agricultural domain specialization
    """
    
    def __init__(self):
        """Initialize the AgriChat system with all components."""
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        
        # Response handlers
        self.fast_handler = None
        
        # System state
        self._initialized = False
        self._conversation_history = []
        
        self.logger.info("[SYSTEM] AgriChat system created")
    
    def initialize_system(self) -> bool:
        """
        Initialize all system components and validate setup.
        
        Returns:
            bool: True if system initialized successfully, False otherwise
        """
        try:
            self.logger.info("[SYSTEM] Starting system initialization...")
            
            # Validate basic setup
            if not validate_setup():
                self.logger.error("[SYSTEM] Basic setup validation failed")
                return False
            
            # Initialize Fast Response Handler if enabled
            if self.config.use_fast_mode:
                try:
                    self.fast_handler = FastResponseHandler()
                    if self.fast_handler.initialize():
                        self.logger.info("[SYSTEM] Fast Response Handler initialized successfully")
                    else:
                        self.logger.warning("[SYSTEM] Fast Response Handler initialization failed")
                        self.fast_handler = None
                except Exception as e:
                    self.logger.warning(f"[SYSTEM] Fast Response Handler error: {e}")
                    self.fast_handler = None
            
            self._initialized = True
            
            # Log system configuration
            config_summary = get_config_summary()
            self.logger.info(f"[SYSTEM] Initialization complete. Config: {config_summary}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[SYSTEM] Initialization failed: {e}")
            return False
    
    def process_question(self, 
                        question: str, 
                        conversation_history: Optional[List[Dict]] = None,
                        user_state: Optional[str] = None) -> str:
        """
        Process an agricultural question and return response.
        
        Args:
            question: User's agricultural question
            conversation_history: Previous conversation context
            user_state: User's geographical state/region
            
        Returns:
            str: Generated response to the agricultural question
        """
        if not self._initialized:
            if not self.initialize_system():
                return "System initialization failed. Please try again later."
        
        try:
            # Create query context
            context = QueryContext(
                question=question,
                conversation_history=conversation_history or [],
                user_state=user_state or "India"
            )
            
            # Select appropriate handler and process
            response = self._process_with_best_handler(context)
            
            # Update conversation history
            self._update_conversation_history(question, response.content)
            
            # Log processing summary
            self.logger.info(
                f"[SYSTEM] Processed query via {response.mode} "
                f"(confidence: {response.confidence:.2f}, "
                f"time: {response.processing_time:.3f}s)"
            )
            
            return response.content
            
        except Exception as e:
            self.logger.error(f"[SYSTEM] Question processing failed: {e}")
            return "I apologize, but I encountered an error processing your question. Please try again."
    
    def _process_with_best_handler(self, context: QueryContext) -> AgriResponse:
        """
        Select the best handler for processing the query.
        
        Args:
            context: Query context with question and metadata
            
        Returns:
            AgriResponse: Response from the selected handler
        """
        # Try Fast Handler first if available
        if self.fast_handler and self.fast_handler.can_handle(context):
            try:
                response = self.fast_handler.get_response(context)
                if response.content != "__FALLBACK__":
                    response.mode = ResponseMode.FAST
                    return response
                else:
                    self.logger.info("[SYSTEM] Fast handler requested fallback")
            except Exception as e:
                self.logger.warning(f"[SYSTEM] Fast handler failed: {e}")
        
        # Use CrewAI as fallback
        try:
            crew_response = self._process_with_crew(context)
            return AgriResponse(
                content=crew_response,
                source="crew_ai_workflow",
                confidence=0.8,
                mode=ResponseMode.CREW_AI
            )
        except Exception as e:
            self.logger.error(f"[SYSTEM] CrewAI processing failed: {e}")
        
        # Final fallback
        return AgriResponse(
            content="I apologize, but I'm unable to process your question right now. Please try again later.",
            source="system_fallback",
            confidence=0.0,
            mode=ResponseMode.FALLBACK
        )
    
    def _process_with_crew(self, context: QueryContext) -> str:
        """
        Process query using CrewAI workflow.
        
        Args:
            context: Query context
            
        Returns:
            str: Generated response from CrewAI
        """
        rag_crew = Crew(
            agents=[Retriever_Agent],
            tasks=[retriever_task],
            verbose=True,
        )
        
        inputs = {
            "question": context.question,
            "conversation_history": context.conversation_history
        }
        
        result = rag_crew.kickoff(inputs=inputs)
        return str(result)
    
    def _update_conversation_history(self, question: str, answer: str) -> None:
        """
        Update internal conversation history for context awareness.
        
        Args:
            question: User's question
            answer: Generated answer
        """
        self._conversation_history.append({
            "question": question,
            "answer": answer
        })
        
        # Keep only last 5 exchanges for memory efficiency
        if len(self._conversation_history) > 5:
            self._conversation_history.pop(0)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status and health information.
        
        Returns:
            Dict containing system status and component health
        """
        status = {
            "system_initialized": self._initialized,
            "configuration": get_config_summary(),
            "handlers": {},
            "conversation_history_length": len(self._conversation_history)
        }
        
        # Fast Handler status
        if self.fast_handler:
            status["handlers"]["fast"] = {
                "available": True,
                "healthy": self.fast_handler.health_check()
            }
        else:
            status["handlers"]["fast"] = {"available": False}
        
        # CrewAI Handler status (always available)
        status["handlers"]["crew"] = {"available": True, "healthy": True}
        
        return status
    
    def reset_conversation(self) -> None:
        """Reset conversation history."""
        self._conversation_history.clear()
        self.logger.info("[SYSTEM] Conversation history reset")


# Legacy function interface for backward compatibility
def get_answer(question: str, 
               conversation_history_param: Optional[List[Dict]] = None, 
               user_state: Optional[str] = None) -> str:
    """
    Legacy function interface for processing agricultural questions.
    
    Args:
        question: User's agricultural question
        conversation_history_param: Previous conversation for context
        user_state: User's state/region
        
    Returns:
        str: Generated response
    """
    global _global_system
    
    # Initialize global system if not already done
    if '_global_system' not in globals() or _global_system is None:
        _global_system = AgriChatSystem()
        _global_system.initialize_system()
    
    return _global_system.process_question(question, conversation_history_param, user_state)


# Initialize global system for legacy compatibility
_global_system = None


# Additional legacy imports for maximum backward compatibility
from agents.crew_agents import retriever_response


if __name__ == "__main__":
    """
    Interactive console interface for testing and development.
    """
    # Initialize system
    system = AgriChatSystem()
    
    if not system.initialize_system():
        print("Failed to initialize AgriChat system!")
        exit(1)
    
    # Display system information
    status = system.get_system_status()
    config = status["configuration"]
    
    print("=" * 60)
    print("🌱 AgriChat Agentic RAG System - Professional Edition")
    print("=" * 60)
    print(f"Environment: {config['environment']}")
    print(f"ChromaDB Path: {config['chroma_path']}")
    print(f"ChromaDB Available: {config['chroma_exists']}")
    print(f"Fast Mode: {'Enabled' if config['fast_mode'] else 'Disabled'}")
    print(f"Ollama Model: {config['ollama_model']}")
    print(f"Ollama Host: {config['ollama_host']}")
    
    print("\nHandler Status:")
    for handler_name, handler_status in status["handlers"].items():
        status_text = "✅ Available & Healthy" if handler_status.get("available") and handler_status.get("healthy") else "❌ Unavailable"
        print(f"  {handler_name.title()} Handler: {status_text}")
    
    print("=" * 60)
    
    # Interactive session
    print("\nWelcome to AgriChat! Ask any agricultural question or type 'exit' to quit.")
    print("Commands: 'status' for system status, 'reset' to clear conversation")
    
    while True:
        try:
            question = input("\n🌾 Your Question: ").strip()
            
            if question.lower() == 'exit':
                print("Thank you for using AgriChat! Happy farming! 🌱")
                break
            elif question.lower() == 'status':
                current_status = system.get_system_status()
                print(f"\nSystem Status: {'✅ Healthy' if current_status['system_initialized'] else '❌ Issues'}")
                print(f"Conversation Length: {current_status['conversation_history_length']} exchanges")
                continue
            elif question.lower() == 'reset':
                system.reset_conversation()
                print("Conversation history cleared.")
                continue
            elif not question:
                continue
            
            # Process question
            print("\n🤖 Processing...")
            answer = system.process_question(question)
            print(f"\n📝 Response:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n\nThank you for using AgriChat! Happy farming! 🌱")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue
