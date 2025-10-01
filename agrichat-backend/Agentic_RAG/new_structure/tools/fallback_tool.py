"""
Fallback agricultural tool for handling queries when RAG system cannot provide adequate responses.

This module implements an LLM-based fallback system specifically designed for Indian
agricultural queries with comprehensive knowledge base integration.
"""

import os
import csv
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, ClassVar
from pydantic import BaseModel, Field, PrivateAttr

from crewai.tools import BaseTool
from ..core import BaseAgriTool, QueryContext, AgriResponse
from ..interfaces.local_llm_interface import run_local_llm


class FallbackToolSchema(BaseModel):
    """Schema for fallback tool input validation."""
    question: str = Field(description="The user's agricultural question")
    conversation_history: Optional[List[Dict]] = Field(
        default=None, 
        description="Previous conversation history for context"
    )


class AgriFallbackTool(BaseAgriTool, BaseTool):
    """
    Agricultural fallback tool using LLM for comprehensive responses.
    
    This tool provides detailed agricultural guidance when the RAG system
    cannot find sufficient information. It specializes in Indian agricultural
    context with region-specific advice and practices.
    
    Features:
    - India-specific agricultural knowledge
    - Multi-language support (responds in query language)
    - Regional farming practice recommendations
    - Disease and pest management guidance
    - Fertilizer and organic solutions
    - Traditional Indian farming method integration
    """
    
    _classifier: Any = PrivateAttr()
    args_schema = FallbackToolSchema
    
    # Comprehensive agricultural prompt for Indian context
    FALLBACK_PROMPT: ClassVar[str] = """
You are an expert agricultural assistant specializing in Indian agriculture and farming practices. 
Focus exclusively on Indian context, regional conditions, and India-specific agricultural solutions. 
Use your expert knowledge to answer agricultural questions relevant to Indian farmers, soil conditions, 
climate patterns, and crop varieties suited to different Indian states and regions.

IMPORTANT: Always respond in the same language in which the query has been asked.

CRITICAL GUIDELINES:
- All responses must be specific to Indian agricultural context, Indian crop varieties, Indian soil types, 
  Indian climate conditions, and farming practices suitable for Indian farmers.
- Consider Indian monsoon patterns, soil types common in India, and crop varieties developed for Indian conditions.
- Reference Indian agricultural practices, local farming techniques, and solutions available to Indian farmers.
- For inputs, focus on products and brands available in Indian markets and cost-effective for Indian farmers.
- Consider regional variations within India (North Indian plains, South Indian conditions, coastal regions, hill states).

GREETING HANDLING:
If the question is a greeting (hello, how are you, good morning, namaste), respond warmly and professionally.
For non-agricultural queries except greetings, politely redirect to agricultural topics.

RESPONSE STRUCTURE:
- Provide detailed, step-by-step advice specific to Indian farming conditions
- Use bullet points, headings, or tables for clarity
- Include traditional Indian farming methods alongside modern practices
- Reference Indian agricultural research institutes (ICAR, state universities) when relevant
- Consider Indian farming seasons (Kharif, Rabi, Zaid) and monsoon patterns
- Provide both commercial and organic/household solutions

SPECIAL FOCUS AREAS:
- Disease and pest management using Indian-available solutions
- Fertilizer recommendations with organic alternatives
- Traditional Indian farming practices and indigenous knowledge
- Water management for Indian climate conditions
- Market-relevant crop varieties for different Indian regions

Source all information as "Agricultural LLM Knowledge Base (Indian Focus)" and ensure
advice is practical and cost-effective for Indian farmers.
"""
    
    def __init__(self, **kwargs):
        """Initialize the Agricultural Fallback tool."""
        BaseAgriTool.__init__(
            self,
            name="agricultural_fallback_tool", 
            description="LLM-based fallback tool for comprehensive Indian agricultural guidance"
        )
        
        BaseTool.__init__(
            self,
            name="fallback_tool",
            description="Fallback LLM tool for agricultural queries with Indian agricultural expertise",
            **kwargs
        )
        
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> bool:
        """
        Initialize the fallback tool.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Test LLM connectivity
            test_response = run_local_llm(
                "test query",
                temperature=0.3,
                model=os.getenv('OLLAMA_MODEL_FALLBACK', 'gpt-oss:20b')
            )
            self._initialized = test_response is not None
            
            if self._initialized:
                self.logger.info("[FALLBACK] Initialized successfully")
            else:
                self.logger.error("[FALLBACK] LLM test failed")
                
            return self._initialized
        except Exception as e:
            self.logger.error(f"[FALLBACK] Initialization failed: {e}")
            return False
    
    def health_check(self) -> bool:
        """
        Check if the fallback tool is functioning correctly.
        
        Returns:
            bool: True if tool is healthy, False otherwise
        """
        try:
            # Quick LLM health check
            response = run_local_llm(
                "test",
                temperature=0.1,
                max_tokens=10,
                model=os.getenv('OLLAMA_MODEL_FALLBACK', 'gpt-oss:20b')
            )
            return response is not None and len(response.strip()) > 0
        except Exception as e:
            self.logger.error(f"[FALLBACK] Health check failed: {e}")
            return False
    
    def process_query(self, context: QueryContext) -> AgriResponse:
        """
        Process agricultural query using LLM fallback approach.
        
        Args:
            context: Query context with question and metadata
            
        Returns:
            AgriResponse: Comprehensive response with Indian agricultural guidance
        """
        start_time = time.time()
        
        try:
            # Check for greeting detection
            if self._is_greeting(context.question):
                response_content = self._handle_greeting(context.question, context.user_state)
                processing_time = time.time() - start_time
                
                return AgriResponse(
                    content=response_content,
                    source="greeting_handler",
                    confidence=1.0,
                    processing_time=processing_time,
                    metadata={"type": "greeting"}
                )
            
            # Build context-aware prompt
            prompt = self._build_contextual_prompt(context)
            
            # Generate LLM response
            response_content = run_local_llm(
                prompt, 
                temperature=0.3, 
                max_tokens=2048,
                model=os.getenv('OLLAMA_MODEL_FALLBACK', 'gpt-oss:20b')
            )
            
            processing_time = time.time() - start_time
            
            if not response_content or response_content.strip() == "":
                return AgriResponse(
                    content="I apologize, but I'm having difficulty generating a response right now. Please try rephrasing your question.",
                    source="fallback_error",
                    confidence=0.0,
                    processing_time=processing_time
                )
            
            # Log successful fallback usage
            self._log_fallback_usage(context.question, response_content)
            
            return AgriResponse(
                content=response_content,
                source="Agricultural LLM Knowledge Base (Indian Focus)",
                confidence=0.8,  # High confidence for LLM responses
                processing_time=processing_time,
                metadata={
                    "user_state": context.user_state,
                    "has_conversation_history": bool(context.conversation_history)
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"[FALLBACK] Query processing failed: {e}")
            
            return AgriResponse(
                content="I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                source="fallback_error", 
                confidence=0.0,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    def _run(self, question: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Legacy interface for CrewAI compatibility.
        
        Args:
            question: User's agricultural question
            conversation_history: Previous conversation context
            
        Returns:
            str: Generated response
        """
        context = QueryContext(
            question=question,
            conversation_history=conversation_history
        )
        
        response = self.process_query(context)
        
        # Log performance metrics
        self.logger.info(
            f"[FALLBACK] Processed query in {response.processing_time:.3f}s "
            f"(confidence: {response.confidence:.1f})"
        )
        
        return response.content
    
    def _is_greeting(self, question: str) -> bool:
        """
        Detect if the question is a greeting or salutation.
        
        Args:
            question: User's input
            
        Returns:
            bool: True if question is a greeting
        """
        question_lower = question.lower().strip()
        
        greetings = [
            'hi', 'hello', 'hey', 'namaste', 'namaskaram', 'vanakkam',
            'good morning', 'good afternoon', 'good evening', 'good day',
            'howdy', 'greetings', 'salaam', 'adaab', 'hi there', 'hello there',
            'how are you', 'how do you do', 'nice to meet you'
        ]
        
        # Check for simple greetings (short questions)
        if len(question_lower) < 30:
            return any(greeting in question_lower for greeting in greetings)
        
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
    
    def _build_contextual_prompt(self, context: QueryContext) -> str:
        """
        Build context-aware prompt for LLM processing.
        
        Args:
            context: Query context with question and metadata
            
        Returns:
            str: Complete prompt for LLM processing
        """
        prompt_parts = [self.FALLBACK_PROMPT]
        
        # Add conversation context if available
        if context.conversation_history:
            prompt_parts.append("\nCONVERSATION CONTEXT:")
            for i, entry in enumerate(context.conversation_history[-3:]):  # Last 3 entries
                prompt_parts.append(f"Q{i+1}: {entry.get('question', '')}")
                prompt_parts.append(f"A{i+1}: {entry.get('answer', '')}")
        
        # Add user state information
        if context.user_state and context.user_state != "India":
            prompt_parts.append(f"\nUSER LOCATION: {context.user_state}")
            prompt_parts.append(f"Please provide advice specifically relevant to {context.user_state} agricultural conditions.")
        
        # Add the current question
        prompt_parts.append(f"\nCURRENT QUESTION: {context.question}")
        prompt_parts.append("\nPROVIDE DETAILED INDIAN AGRICULTURAL GUIDANCE:")
        
        return "\n".join(prompt_parts)
    
    def _log_fallback_usage(self, question: str, response: str) -> None:
        """
        Log fallback tool usage for analysis and improvement.
        
        Args:
            question: Original user question
            response: Generated response
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
                        'timestamp', 'question', 'response_preview', 
                        'tool_used', 'status'
                    ])
                
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                response_preview = response[:100] + "..." if len(response) > 100 else response
                
                writer.writerow([
                    timestamp, question, response_preview,
                    'AgriFallbackTool', 'success'
                ])
                
        except Exception as e:
            self.logger.warning(f"[FALLBACK] Failed to log usage: {e}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get fallback tool capabilities and current status.
        
        Returns:
            Dict containing tool capabilities and configuration
        """
        base_capabilities = super().get_capabilities()
        
        return {
            **base_capabilities,
            "llm_available": self.health_check(),
            "specializations": [
                "Indian agricultural practices",
                "Regional farming guidance",
                "Multi-language support",
                "Traditional farming methods",
                "Disease and pest management",
                "Organic farming solutions"
            ],
            "coverage": [
                "All Indian states and regions",
                "Kharif, Rabi, Zaid seasons",
                "Multiple crop varieties",
                "Soil and climate adaptations"
            ]
        }
