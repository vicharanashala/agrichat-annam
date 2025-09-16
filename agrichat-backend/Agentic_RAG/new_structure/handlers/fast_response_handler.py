"""
Fast Response Handler - Optimized agricultural response processing.

This module provides rapid response generation by bypassing complex multi-agent workflows
and using direct tool invocation with intelligent decision logic.
"""

import os
import time
import logging
from typing import List, Dict, Optional

from ..core import BaseResponseHandler, QueryContext, AgriResponse, ResponseMode, get_chroma_path
from ..tools import AgriRAGTool, AgriFallbackTool


class FastResponseHandler(BaseResponseHandler):
    """
    High-performance response handler for agricultural queries.
    
    This handler provides 10-15x faster responses by using direct tool calls
    instead of multi-agent workflows while maintaining response quality.
    
    Features:
    - Direct tool invocation for speed
    - Intelligent greeting detection
    - Context-aware processing
    - Automatic fallback mechanisms
    - Performance monitoring
    """
    
    def __init__(self):
        """Initialize the fast response handler with direct tool access."""
        super().__init__(ResponseMode.FAST)
        
        self.logger = logging.getLogger(__name__)
        self.chroma_path = get_chroma_path()
        
        # Initialize tools
        self.rag_tool = None
        self.fallback_tool = None
        
        # Greeting patterns for fast detection
        self.simple_greetings = [
            'hi', 'hello', 'hey', 'namaste', 'namaskaram', 'vanakkam', 
            'good morning', 'good afternoon', 'good evening', 'good day',
            'howdy', 'greetings', 'salaam', 'adaab', 'hi there', 'hello there',
            'how are you', 'how do you do', 'nice to meet you'
        ]
        
        self.logger.info(f"[FAST] Initialized with ChromaDB path: {self.chroma_path}")
    
    def initialize(self) -> bool:
        """
        Initialize fast response handler with required tools.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize RAG tool
            self.rag_tool = AgriRAGTool(chroma_path=self.chroma_path)
            if not self.rag_tool.initialize():
                self.logger.error("[FAST] RAG tool initialization failed")
                return False
            
            # Initialize fallback tool
            self.fallback_tool = AgriFallbackTool()
            if not self.fallback_tool.initialize():
                self.logger.warning("[FAST] Fallback tool initialization failed")
                # Continue without fallback - not critical
            
            self._initialized = True
            self.logger.info("[FAST] Fast response handler initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"[FAST] Initialization failed: {e}")
            return False
    
    def can_handle(self, context: QueryContext) -> bool:
        """
        Check if fast handler can process the given query.
        
        Args:
            context: Query context to evaluate
            
        Returns:
            bool: True if handler can process query, False otherwise
        """
        if not self._initialized:
            return False
        
        # Fast handler can handle most queries efficiently
        # Only skip very complex multi-step reasoning queries
        complex_indicators = [
            'step by step', 'detailed analysis', 'comprehensive study',
            'research report', 'detailed comparison', 'in-depth analysis'
        ]
        
        question_lower = context.question.lower()
        is_complex = any(indicator in question_lower for indicator in complex_indicators)
        
        return not is_complex
    
    def get_response(self, context: QueryContext) -> AgriResponse:
        """
        Generate fast response for agricultural query.
        
        Args:
            context: Query context with question and metadata
            
        Returns:
            AgriResponse: Generated response with metadata
        """
        if not self._initialized:
            if not self.initialize():
                return AgriResponse(
                    content="Fast response handler not available",
                    source="error",
                    confidence=0.0
                )
        
        start_time = time.time()
        
        try:
            # Check for simple greetings first
            if self._is_simple_greeting(context.question):
                response_content = self._handle_greeting(context.question, context.user_state)
                processing_time = time.time() - start_time
                
                self.logger.info(f"[FAST] Handled greeting in {processing_time:.3f}s")
                
                return AgriResponse(
                    content=response_content,
                    source="greeting_handler",
                    confidence=1.0,
                    processing_time=processing_time
                )
            
            # Try RAG tool first for agricultural content
            rag_start = time.time()
            rag_response = self.rag_tool.process_query(context)
            rag_time = time.time() - rag_start
            
            self.logger.debug(f"[FAST] RAG processing took {rag_time:.3f}s")
            
            # Check if RAG provided adequate response
            if (rag_response.content != "__FALLBACK__" and 
                rag_response.confidence >= 0.3):
                
                total_time = time.time() - start_time
                rag_response.processing_time = total_time
                
                self.logger.info(
                    f"[FAST] RAG success in {total_time:.3f}s "
                    f"(confidence: {rag_response.confidence:.3f})"
                )
                
                return rag_response
            
            # Fallback to LLM if RAG insufficient
            if self.fallback_tool:
                self.logger.info("[FAST] RAG insufficient, using fallback LLM")
                
                fallback_start = time.time()
                fallback_response = self.fallback_tool.process_query(context)
                fallback_time = time.time() - fallback_start
                
                total_time = time.time() - start_time
                fallback_response.processing_time = total_time
                
                self.logger.info(f"[FAST] Fallback completed in {total_time:.3f}s")
                
                return fallback_response
            else:
                # No fallback available
                return AgriResponse(
                    content="__FALLBACK__",
                    source="fast_insufficient",
                    confidence=0.0,
                    processing_time=time.time() - start_time
                )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"[FAST] Response generation failed: {e}")
            
            return AgriResponse(
                content="__FALLBACK__",
                source="fast_error",
                confidence=0.0,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    def _is_simple_greeting(self, question: str) -> bool:
        """
        Fast greeting detection without LLM processing.
        
        Args:
            question: User's input
            
        Returns:
            bool: True if question is a simple greeting
        """
        question_lower = question.lower().strip()
        
        # Short questions with greeting keywords
        if len(question_lower) < 30:
            return any(greeting in question_lower for greeting in self.simple_greetings)
        
        return False
    
    def _handle_greeting(self, question: str, user_state: Optional[str]) -> str:
        """
        Generate appropriate greeting response.
        
        Args:
            question: Original greeting question
            user_state: User's geographical state
            
        Returns:
            str: Friendly greeting response with agricultural context
        """
        question_lower = question.lower()
        
        if 'namaste' in question_lower:
            return "Namaste! Welcome to AgriChat. I'm here to help you with all your farming and agriculture questions. What would you like to know about?"
        elif 'namaskaram' in question_lower:
            return "Namaskaram! I'm your agricultural assistant. Feel free to ask me anything about crops, farming techniques, or agricultural practices."
        elif 'vanakkam' in question_lower:
            return "Vanakkam! I'm here to assist you with farming and agriculture. What agricultural topic would you like to discuss today?"
        elif any(time in question_lower for time in ['morning', 'afternoon', 'evening']):
            time_word = next(time for time in ['morning', 'afternoon', 'evening'] if time in question_lower)
            return f"Good {time_word}! I'm your agricultural assistant. How can I help you with your farming questions today?"
        elif 'how are you' in question_lower:
            return "I'm doing well, thank you for asking! I'm here and ready to help you with any agricultural questions or farming advice you need. What would you like to know about?"
        else:
            return "Hello! I'm your agricultural assistant specialized in Indian farming practices. I'm here to help with crops, farming techniques, pest management, and all aspects of agriculture. What can I assist you with today?"
    
    def health_check(self) -> bool:
        """
        Check if fast response handler is functioning correctly.
        
        Returns:
            bool: True if handler is healthy, False otherwise
        """
        try:
            if not self._initialized:
                return False
            
            # Check RAG tool health
            if not self.rag_tool or not self.rag_tool.health_check():
                return False
            
            # Fallback tool health is optional
            if self.fallback_tool and not self.fallback_tool.health_check():
                self.logger.warning("[FAST] Fallback tool unhealthy, but continuing")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[FAST] Health check failed: {e}")
            return False
    
    def get_capabilities(self) -> Dict[str, any]:
        """
        Get fast handler capabilities and status.
        
        Returns:
            Dict containing handler capabilities and configuration
        """
        return {
            "mode": self.mode.value,
            "initialized": self._initialized,
            "healthy": self.health_check(),
            "chroma_path": self.chroma_path,
            "features": [
                "Direct tool invocation",
                "Fast greeting detection", 
                "Context-aware processing",
                "Automatic fallback mechanisms",
                "Performance monitoring"
            ],
            "performance": "10-15x faster than CrewAI workflow",
            "tools": {
                "rag_available": self.rag_tool is not None and self.rag_tool.is_initialized(),
                "fallback_available": self.fallback_tool is not None and self.fallback_tool.is_initialized()
            }
        }
    
    # Legacy interface for backward compatibility
    def get_answer(self, 
                   question: str, 
                   conversation_history: Optional[List[Dict]] = None,
                   user_state: Optional[str] = None) -> str:
        """
        Legacy interface for fast response generation.
        
        Args:
            question: User's agricultural question
            conversation_history: Previous conversation context
            user_state: User's geographical state
            
        Returns:
            str: Generated response
        """
        context = QueryContext(
            question=question,
            conversation_history=conversation_history,
            user_state=user_state
        )
        
        response = self.get_response(context)
        return response.content
