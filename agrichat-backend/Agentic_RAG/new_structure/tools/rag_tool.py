"""
RAG (Retrieval-Augmented Generation) tool for agricultural knowledge retrieval.

This module implements the core RAG functionality using ChromaDB for vector search
and retrieval of agricultural information with proper source attribution.
"""

import os
import csv
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, PrivateAttr

from crewai.tools import BaseTool
from ..core import BaseAgriTool, QueryContext, AgriResponse
from ..handlers.chroma_query_handler import ChromaQueryHandler


class RAGToolSchema(BaseModel):
    """Schema for RAG tool input validation."""
    question: str = Field(description="The user's agricultural question")
    conversation_history: Optional[List[Dict]] = Field(
        default=None, 
        description="Previous conversation history for context"
    )
    user_state: str = Field(
        default="India", 
        description="User's geographical state/region"
    )


class AgriRAGTool(BaseAgriTool, BaseTool):
    """
    Agricultural Retrieval-Augmented Generation tool.
    
    This tool provides intelligent retrieval of agricultural knowledge from a
    curated vector database, with fallback detection and source attribution.
    
    Features:
    - Vector similarity search using ChromaDB
    - Context-aware responses using conversation history
    - Regional agricultural knowledge (India-specific)
    - Automatic fallback detection for incomplete information
    - Performance monitoring and logging
    """
    
    _handler: Any = PrivateAttr()
    args_schema = RAGToolSchema
    
    def __init__(self, chroma_path: str, **kwargs):
        """
        Initialize the Agricultural RAG tool.
        
        Args:
            chroma_path: Path to ChromaDB storage directory
            **kwargs: Additional arguments for parent classes
        """
        BaseAgriTool.__init__(
            self, 
            name="agricultural_rag_tool",
            description="Retrieval-Augmented Generation tool for agricultural knowledge using ChromaDB vector search"
        )
        
        BaseTool.__init__(
            self,
            name="rag_tool",
            description="Retrieval-Augmented Generation tool using ChromaDB for agricultural queries",
            **kwargs
        )
        
        self.chroma_path = chroma_path
        self._handler = None
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> bool:
        """
        Initialize the RAG tool with ChromaDB handler.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self._handler = ChromaQueryHandler(self.chroma_path)
            self._initialized = True
            self.logger.info(f"[RAG] Initialized with ChromaDB path: {self.chroma_path}")
            return True
        except Exception as e:
            self.logger.error(f"[RAG] Initialization failed: {e}")
            return False
    
    def health_check(self) -> bool:
        """
        Check if the RAG tool is functioning correctly.
        
        Returns:
            bool: True if tool is healthy, False otherwise
        """
        try:
            if not self._handler:
                return False
            
            # Test query to verify ChromaDB connectivity
            test_result = self._handler.get_answer_with_source(
                "test query", [], "India"
            )
            return test_result is not None
        except Exception as e:
            self.logger.error(f"[RAG] Health check failed: {e}")
            return False
    
    def process_query(self, context: QueryContext) -> AgriResponse:
        """
        Process agricultural query using RAG approach.
        
        Args:
            context: Query context with question and metadata
            
        Returns:
            AgriResponse: Response with content and source attribution
        """
        if not self._initialized:
            if not self.initialize():
                return AgriResponse(
                    content="RAG tool not available",
                    source="error",
                    confidence=0.0
                )
        
        start_time = time.time()
        
        try:
            # Query ChromaDB for relevant agricultural information
            result = self._handler.get_answer_with_source(
                context.question,
                context.conversation_history or [],
                context.user_state or "India"
            )
            
            processing_time = time.time() - start_time
            
            # Check for fallback conditions
            if self._should_fallback(result):
                self._log_fallback_query(context.question, result)
                return AgriResponse(
                    content="__FALLBACK__",
                    source="rag_fallback",
                    confidence=0.0,
                    processing_time=processing_time,
                    metadata=result
                )
            
            # Process and clean the response
            answer = self._process_answer(result['answer'])
            
            return AgriResponse(
                content=answer,
                source=result.get('source', 'rag_database'),
                confidence=result.get('cosine_similarity', 0.0),
                processing_time=processing_time,
                metadata={
                    'document_metadata': result.get('document_metadata', {}),
                    'similarity_score': result.get('cosine_similarity', 0.0)
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"[RAG] Query processing failed: {e}")
            
            return AgriResponse(
                content="__FALLBACK__",
                source="error",
                confidence=0.0,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    def _run(self, question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = "") -> str:
        """
        Legacy interface for CrewAI compatibility.
        
        Args:
            question: User's agricultural question
            conversation_history: Previous conversation context
            user_state: User's geographical state/region
            
        Returns:
            str: Generated response or __FALLBACK__ indicator
        """
        context = QueryContext(
            question=question,
            conversation_history=conversation_history,
            user_state=user_state or "India"
        )
        
        response = self.process_query(context)
        
        # Log performance metrics
        self.logger.info(
            f"[RAG] Processed query in {response.processing_time:.3f}s "
            f"(confidence: {response.confidence:.3f})"
        )
        
        return response.content
    
    def run_with_context(self, question: str, conversation_history: List[Dict]) -> str:
        """
        Helper method to explicitly run with conversation context.
        
        Args:
            question: User's agricultural question
            conversation_history: Previous conversation for context
            
        Returns:
            str: Generated response
        """
        return self._run(question, conversation_history, "India")
    
    def _should_fallback(self, result: Dict[str, Any]) -> bool:
        """
        Determine if the query should fall back to alternative processing.
        
        Args:
            result: Query result from ChromaDB handler
            
        Returns:
            bool: True if fallback needed, False otherwise
        """
        if not result or not result.get('answer'):
            return True
        
        answer = result['answer']
        
        # Check for explicit fallback indicators
        if answer.startswith("__FALLBACK__"):
            return True
        
        # Check for insufficient information responses
        insufficient_responses = [
            "I don't have enough information to answer that.",
            "I cannot find relevant information",
            "No relevant information found"
        ]
        
        if any(phrase in answer for phrase in insufficient_responses):
            return True
        
        # Check confidence/similarity threshold
        similarity = result.get('cosine_similarity', 0.0)
        if similarity < 0.3:  # Low similarity threshold
            return True
        
        return False
    
    def _process_answer(self, answer: str) -> str:
        """
        Process and clean the retrieved answer.
        
        Args:
            answer: Raw answer from ChromaDB
            
        Returns:
            str: Processed and cleaned answer
        """
        if answer.startswith("__NO_SOURCE__"):
            answer = answer.replace("__NO_SOURCE__", "")
        
        return answer.strip()
    
    def _log_fallback_query(self, question: str, result: Dict[str, Any]) -> None:
        """
        Log queries that required fallback for analysis and improvement.
        
        Args:
            question: Original user question
            result: Query result that triggered fallback
        """
        try:
            fallback_file = os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "fallback_queries.csv"
            )
            
            file_exists = os.path.isfile(fallback_file)
            
            with open(fallback_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                if not file_exists:
                    writer.writerow([
                        'timestamp', 'question', 'fallback_reason', 
                        'similarity_score', 'source'
                    ])
                
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                fallback_reason = f"RAG insufficient - {result.get('source', 'unknown')}"
                similarity = result.get('cosine_similarity', 0.0)
                source = result.get('source', 'unknown')
                
                writer.writerow([
                    timestamp, question, fallback_reason, similarity, source
                ])
                
        except Exception as e:
            self.logger.warning(f"[RAG] Failed to log fallback query: {e}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get RAG tool capabilities and current status.
        
        Returns:
            Dict containing tool capabilities and configuration
        """
        base_capabilities = super().get_capabilities()
        
        return {
            **base_capabilities,
            "chroma_path": self.chroma_path,
            "chroma_exists": os.path.exists(self.chroma_path),
            "handler_initialized": self._handler is not None,
            "features": [
                "Vector similarity search",
                "Context-aware responses", 
                "Regional agricultural knowledge",
                "Automatic fallback detection",
                "Performance monitoring"
            ]
        }
