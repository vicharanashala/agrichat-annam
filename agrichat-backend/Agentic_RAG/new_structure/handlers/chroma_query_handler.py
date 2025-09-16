"""
ChromaDB query handler for agricultural knowledge retrieval.

This module provides a sophisticated interface to ChromaDB for retrieving agricultural
information with context awareness, conversation memory, and intelligent fallback detection.
"""

import os
import re
import hashlib
import logging
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from numpy.linalg import norm
import pytz
from langchain_community.vectorstores import Chroma

from ..core import BaseAgriTool
from ..utils.context_manager import ConversationContext, MemoryStrategy
from ..interfaces.local_llm_interface import local_llm, local_embeddings, run_local_llm


class ChromaQueryHandler(BaseAgriTool):
    """
    Advanced ChromaDB query handler for agricultural knowledge retrieval.
    
    This class provides sophisticated querying capabilities including:
    - Vector similarity search with agricultural context
    - Conversation-aware query processing
    - Regional agricultural knowledge adaptation
    - Multi-language response generation
    - Performance optimization through caching
    
    Features:
    - Context-aware query enhancement
    - Conversation memory integration
    - Regional preference handling
    - Similarity threshold optimization
    - Structured response formatting
    """
    
    # Regional instruction template for India-specific responses
    REGION_INSTRUCTION = """
IMPORTANT: If a state or region is mentioned in the query, always give preference to that region. 
If the mentioned region is outside India, politely inform the user that you are only trained on 
Indian agriculture data and cannot answer for other regions. If the query is not related to 
Indian agriculture, politely inform the user that you are only trained on Indian agriculture 
data and can only answer questions related to Indian agriculture.

LOCATION FLEXIBILITY: Provide agricultural information from the available context regardless 
of specific location, as agricultural practices are generally applicable across similar 
climatic conditions in India. Only mention specific regional context if it's directly 
relevant to the user's query.
"""
    
    # Structured prompt template for consistent responses
    STRUCTURED_PROMPT = """
You are an agricultural assistant specialized in Indian agriculture. Answer the user's question 
using ONLY the information provided in the context below. Do not add any external knowledge.

IMPORTANT: Always respond in the same language in which the query has been asked.

{region_instruction}
Current month: {current_month}

INSTRUCTIONS:
- Use ONLY the information from the provided context
- If context has sufficient information, provide a clear and helpful answer
- Provide agricultural information applicable across India with regional adaptations when available
- Do NOT add any information from external sources or your own knowledge
- Do NOT mention metadata unless specifically asked or directly relevant
- Structure your response clearly with bullet points or short paragraphs
- If context is insufficient, respond exactly: "I don't have enough information to answer that."

CRITICAL - For High Similarity Matches (Comprehensive Database Response):
- When context directly answers the question (high similarity match), provide COMPLETE information
- Include ALL practical details: varieties, timing, spacing, irrigation, yield, management practices
- Do NOT truncate, summarize, or omit ANY part of the database content
- Ensure EVERY sentence and detail from the context is included in your response
- Present complete information in well-structured, readable format
- NEVER leave out yield data, management practices, or cautionary information

### Context from Agricultural Database:
{context}

### User Question:
{question}

### Response:
"""
    
    def __init__(self, chroma_path: str):
        """
        Initialize ChromaDB query handler.
        
        Args:
            chroma_path: Path to ChromaDB storage directory
        """
        super().__init__(
            name="chroma_query_handler",
            description="Advanced ChromaDB query handler for agricultural knowledge retrieval"
        )
        
        self.chroma_path = chroma_path
        self.vectorstore = None
        self.conversation_context = ConversationContext(MemoryStrategy.SLIDING_WINDOW)
        self.logger = logging.getLogger(__name__)
        
        # Performance optimization settings
        self._cache_size = 100
        self._similarity_threshold = 0.3
        self._max_documents = 5
        
    def initialize(self) -> bool:
        """
        Initialize ChromaDB connection and validate setup.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            if not os.path.exists(self.chroma_path):
                self.logger.error(f"[CHROMA] Database path does not exist: {self.chroma_path}")
                return False
            
            # Initialize ChromaDB with embeddings
            self.vectorstore = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=local_embeddings
            )
            
            # Test database connectivity
            test_results = self.vectorstore.similarity_search("test query", k=1)
            
            self._initialized = True
            self.logger.info(f"[CHROMA] Initialized successfully with {len(test_results)} test documents")
            return True
            
        except Exception as e:
            self.logger.error(f"[CHROMA] Initialization failed: {e}")
            return False
    
    def health_check(self) -> bool:
        """
        Check ChromaDB health and connectivity.
        
        Returns:
            bool: True if database is healthy, False otherwise
        """
        try:
            if not self.vectorstore:
                return False
            
            # Test query to verify database functionality
            results = self.vectorstore.similarity_search("health check", k=1)
            return len(results) > 0
            
        except Exception as e:
            self.logger.error(f"[CHROMA] Health check failed: {e}")
            return False
    
    def get_answer_with_source(self, 
                             question: str, 
                             conversation_history: Optional[List[Dict]] = None,
                             user_state: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive answer with source attribution and metadata.
        
        Args:
            question: User's agricultural question
            conversation_history: Previous conversation context
            user_state: User's geographical state/region
            
        Returns:
            Dict containing answer, source, similarity score, and metadata
        """
        if not self._initialized:
            if not self.initialize():
                return self._create_error_response("ChromaDB not available")
        
        try:
            # Update conversation context
            if conversation_history:
                self.conversation_context.update_history(conversation_history)
            
            # Enhance query with context
            enhanced_query = self._enhance_query_with_context(question, user_state)
            
            # Perform vector similarity search
            search_results = self._perform_similarity_search(enhanced_query)
            
            if not search_results:
                return self._create_fallback_response("No relevant documents found")
            
            # Get best match with similarity score
            best_match = search_results[0]
            similarity_score = self._calculate_similarity_score(enhanced_query, best_match)
            
            # Build context for LLM processing
            context = self._build_context_from_results(search_results)
            
            # Generate structured response
            response = self._generate_structured_response(
                question, context, user_state, similarity_score
            )
            
            return {
                'answer': response,
                'source': self._extract_source_info(best_match),
                'cosine_similarity': similarity_score,
                'document_metadata': self._extract_metadata(best_match),
                'num_documents_used': len(search_results),
                'enhanced_query': enhanced_query
            }
            
        except Exception as e:
            self.logger.error(f"[CHROMA] Query processing failed: {e}")
            return self._create_error_response(f"Query processing error: {str(e)}")
    
    def _enhance_query_with_context(self, question: str, user_state: Optional[str] = None) -> str:
        """
        Enhance query with conversation context and regional information.
        
        Args:
            question: Original user question
            user_state: User's geographical state
            
        Returns:
            str: Enhanced query with additional context
        """
        enhanced_parts = [question]
        
        # Add regional context
        if user_state and user_state.lower() != "india":
            enhanced_parts.append(f"region: {user_state}")
        
        # Add conversation context
        context_keywords = self.conversation_context.get_context_keywords()
        if context_keywords:
            enhanced_parts.append(f"related topics: {' '.join(context_keywords[:3])}")
        
        # Add seasonal context
        current_month = datetime.now().strftime("%B")
        enhanced_parts.append(f"current season context: {current_month}")
        
        enhanced_query = " ".join(enhanced_parts)
        self.logger.debug(f"[CHROMA] Enhanced query: {enhanced_query}")
        
        return enhanced_query
    
    @lru_cache(maxsize=100)
    def _perform_similarity_search(self, query: str) -> List[Any]:
        """
        Perform cached similarity search with optimization.
        
        Args:
            query: Enhanced query string
            
        Returns:
            List of relevant documents from ChromaDB
        """
        try:
            # Perform similarity search with metadata filtering
            results = self.vectorstore.similarity_search_with_score(
                query, 
                k=self._max_documents
            )
            
            # Filter by similarity threshold
            filtered_results = [
                doc for doc, score in results 
                if (1 - score) >= self._similarity_threshold  # Convert distance to similarity
            ]
            
            self.logger.debug(f"[CHROMA] Found {len(filtered_results)} relevant documents")
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"[CHROMA] Similarity search failed: {e}")
            return []
    
    def _calculate_similarity_score(self, query: str, document: Any) -> float:
        """
        Calculate precise similarity score between query and document.
        
        Args:
            query: Enhanced query string
            document: ChromaDB document
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        try:
            # Get embeddings for query and document
            query_embedding = local_embeddings.embed_query(query)
            doc_embedding = local_embeddings.embed_query(document.page_content)
            
            # Calculate cosine similarity
            query_array = np.array(query_embedding)
            doc_array = np.array(doc_embedding)
            
            similarity = np.dot(query_array, doc_array) / (norm(query_array) * norm(doc_array))
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            self.logger.warning(f"[CHROMA] Similarity calculation failed: {e}")
            return 0.5  # Default moderate similarity
    
    def _build_context_from_results(self, results: List[Any]) -> str:
        """
        Build comprehensive context from search results.
        
        Args:
            results: List of relevant documents
            
        Returns:
            str: Formatted context for LLM processing
        """
        if not results:
            return ""
        
        context_parts = []
        
        for i, doc in enumerate(results[:self._max_documents], 1):
            content = doc.page_content.strip()
            
            # Clean and format document content
            formatted_content = self._clean_document_content(content)
            
            context_parts.append(f"Document {i}:\n{formatted_content}")
        
        return "\n\n".join(context_parts)
    
    def _clean_document_content(self, content: str) -> str:
        """
        Clean and format document content for better LLM processing.
        
        Args:
            content: Raw document content
            
        Returns:
            str: Cleaned and formatted content
        """
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters that might interfere
        content = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]]', ' ', content)
        
        # Ensure proper sentence structure
        content = re.sub(r'\.(\w)', r'. \1', content)
        
        return content.strip()
    
    def _generate_structured_response(self, 
                                    question: str, 
                                    context: str, 
                                    user_state: Optional[str],
                                    similarity_score: float) -> str:
        """
        Generate structured response using LLM with agricultural prompt.
        
        Args:
            question: Original user question
            context: Formatted context from documents
            user_state: User's geographical state
            similarity_score: Similarity score for response quality indication
            
        Returns:
            str: Generated structured response
        """
        try:
            # Determine if high-quality context is available
            is_high_similarity = similarity_score >= 0.7
            
            current_month = datetime.now().strftime("%B")
            
            # Build comprehensive prompt
            prompt = self.STRUCTURED_PROMPT.format(
                region_instruction=self.REGION_INSTRUCTION,
                current_month=current_month,
                context=context,
                question=question
            )
            
            # Add user state context if available
            if user_state and user_state.lower() != "india":
                prompt += f"\n\nNote: User is from {user_state}. Adapt advice accordingly."
            
            # Generate response with appropriate parameters
            temperature = 0.1 if is_high_similarity else 0.3
            max_tokens = 2048 if is_high_similarity else 1024
            
            response = run_local_llm(
                prompt, 
                temperature=temperature, 
                max_tokens=max_tokens
            )
            
            if not response or response.strip() == "":
                return "I don't have enough information to answer that."
            
            # Clean and validate response
            cleaned_response = self._clean_response(response)
            
            # Check for fallback conditions
            if self._should_trigger_fallback(cleaned_response, similarity_score):
                return "__FALLBACK__"
            
            return cleaned_response
            
        except Exception as e:
            self.logger.error(f"[CHROMA] Response generation failed: {e}")
            return "__FALLBACK__"
    
    def _clean_response(self, response: str) -> str:
        """
        Clean and format the generated response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            str: Cleaned and formatted response
        """
        # Remove any prompt artifacts
        response = re.sub(r'^.*?Response:\s*', '', response, flags=re.DOTALL)
        
        # Clean excessive whitespace
        response = re.sub(r'\n\s*\n', '\n\n', response)
        response = re.sub(r' +', ' ', response)
        
        # Ensure proper formatting
        response = response.strip()
        
        return response
    
    def _should_trigger_fallback(self, response: str, similarity_score: float) -> bool:
        """
        Determine if response should trigger fallback to alternative processing.
        
        Args:
            response: Generated response
            similarity_score: Similarity score from retrieval
            
        Returns:
            bool: True if fallback should be triggered
        """
        # Check for explicit insufficient information responses
        insufficient_phrases = [
            "I don't have enough information",
            "I cannot find relevant information",
            "insufficient information",
            "not enough information"
        ]
        
        if any(phrase in response.lower() for phrase in insufficient_phrases):
            return True
        
        # Check for very low similarity scores
        if similarity_score < self._similarity_threshold:
            return True
        
        # Check for very short responses (likely insufficient)
        if len(response.strip()) < 50:
            return True
        
        return False
    
    def _extract_source_info(self, document: Any) -> str:
        """
        Extract source information from document metadata.
        
        Args:
            document: ChromaDB document
            
        Returns:
            str: Source information string
        """
        try:
            metadata = getattr(document, 'metadata', {})
            
            source_parts = []
            
            if 'source' in metadata:
                source_parts.append(f"Source: {metadata['source']}")
            
            if 'category' in metadata:
                source_parts.append(f"Category: {metadata['category']}")
            
            if source_parts:
                return " | ".join(source_parts)
            
            return "Agricultural Knowledge Database"
            
        except Exception as e:
            self.logger.warning(f"[CHROMA] Source extraction failed: {e}")
            return "Database"
    
    def _extract_metadata(self, document: Any) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from document.
        
        Args:
            document: ChromaDB document
            
        Returns:
            Dict containing document metadata
        """
        try:
            metadata = getattr(document, 'metadata', {})
            
            return {
                'content_length': len(document.page_content),
                'metadata': metadata,
                'document_type': metadata.get('type', 'agricultural_content')
            }
            
        except Exception as e:
            self.logger.warning(f"[CHROMA] Metadata extraction failed: {e}")
            return {}
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create standardized error response.
        
        Args:
            error_message: Error description
            
        Returns:
            Dict containing error response structure
        """
        return {
            'answer': '__FALLBACK__',
            'source': 'error',
            'cosine_similarity': 0.0,
            'document_metadata': {'error': error_message},
            'num_documents_used': 0,
            'enhanced_query': ''
        }
    
    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """
        Create standardized fallback response.
        
        Args:
            reason: Fallback reason
            
        Returns:
            Dict containing fallback response structure
        """
        return {
            'answer': '__FALLBACK__',
            'source': 'fallback_required',
            'cosine_similarity': 0.0,
            'document_metadata': {'fallback_reason': reason},
            'num_documents_used': 0,
            'enhanced_query': ''
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get ChromaDB handler capabilities and status.
        
        Returns:
            Dict containing handler capabilities and configuration
        """
        base_capabilities = super().get_capabilities()
        
        return {
            **base_capabilities,
            "chroma_path": self.chroma_path,
            "vectorstore_available": self.vectorstore is not None,
            "similarity_threshold": self._similarity_threshold,
            "max_documents": self._max_documents,
            "cache_size": self._cache_size,
            "features": [
                "Vector similarity search",
                "Conversation context awareness",
                "Regional knowledge adaptation",
                "Multi-language support",
                "Performance caching",
                "Intelligent fallback detection"
            ]
        }
