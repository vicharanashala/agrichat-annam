"""
Context Manager for agricultural conversation tracking and memory management.

This module provides sophisticated conversation context management with multiple
memory strategies optimized for agricultural knowledge processing.
"""

import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum


class MemoryStrategy(Enum):
    """Available memory strategy types for conversation management."""
    BUFFER = "buffer"         # Keep recent conversations in full
    SUMMARY = "summary"       # Summarize old conversations  
    HYBRID = "hybrid"         # Combination of buffer + summary
    SLIDING_WINDOW = "sliding_window"  # Simple sliding window
    AUTO = "auto"            # Automatic strategy selection


class ConversationContext:
    """
    Enhanced conversation context manager for agricultural AI systems.
    
    This class provides sophisticated conversation memory management with
    multiple strategies optimized for agricultural domain knowledge processing.
    
    Features:
    - Multiple memory strategies (buffer, summary, hybrid, auto)
    - Agricultural entity tracking (crops, diseases, pests, etc.)
    - Context-aware keyword extraction
    - Token-aware context management
    - Conversation summarization
    - Entity persistence across conversations
    """
    
    def __init__(self, 
                 memory_strategy: MemoryStrategy = MemoryStrategy.SLIDING_WINDOW,
                 max_context_pairs: int = 5,
                 max_context_tokens: int = 800,
                 hybrid_buffer_pairs: int = 3,
                 summary_threshold: int = 8):
        """
        Initialize conversation context manager.
        
        Args:
            memory_strategy: Memory management strategy to use
            max_context_pairs: Maximum Q&A pairs for buffer mode
            max_context_tokens: Maximum estimated tokens for context
            hybrid_buffer_pairs: Recent pairs to keep in hybrid mode
            summary_threshold: Conversation length for summary mode
        """
        self.memory_strategy = memory_strategy
        self.max_context_pairs = max_context_pairs
        self.max_context_tokens = max_context_tokens
        self.hybrid_buffer_pairs = hybrid_buffer_pairs
        self.summary_threshold = summary_threshold
        
        # Conversation storage
        self.conversation_history = []
        self.conversation_summary = ""
        
        # Agricultural entity tracking
        self.tracked_entities = {
            'crops': set(),
            'diseases': set(),
            'pests': set(),
            'fertilizers': set(),
            'locations': set(),
            'seasons': set(),
            'techniques': set()
        }
        
        # Common agricultural terms for entity recognition
        self._entity_patterns = {
            'crops': [
                'rice', 'wheat', 'maize', 'corn', 'barley', 'millet', 'sorghum',
                'cotton', 'sugarcane', 'tobacco', 'tea', 'coffee', 'cardamom',
                'potato', 'tomato', 'onion', 'garlic', 'cabbage', 'cauliflower',
                'carrot', 'beans', 'peas', 'lentils', 'chickpea', 'soybean',
                'coconut', 'palm', 'banana', 'mango', 'apple', 'orange',
                'groundnut', 'sesame', 'mustard', 'sunflower', 'safflower'
            ],
            'diseases': [
                'blight', 'rust', 'smut', 'wilt', 'rot', 'mosaic', 'leaf spot',
                'bacterial', 'fungal', 'viral', 'yellowing', 'brown spot',
                'blast', 'canker', 'scorch', 'mildew', 'anthracnose'
            ],
            'pests': [
                'aphid', 'bollworm', 'army worm', 'cutworm', 'thrips', 'mite',
                'whitefly', 'leafhopper', 'stem borer', 'fruit fly', 'caterpillar',
                'beetle', 'weevil', 'grasshopper', 'locust', 'termite'
            ],
            'fertilizers': [
                'urea', 'dap', 'mop', 'ssp', 'nitrogen', 'phosphorus', 'potassium',
                'organic', 'compost', 'manure', 'vermicompost', 'biofertilizer',
                'neem cake', 'bone meal', 'rock phosphate'
            ],
            'seasons': [
                'kharif', 'rabi', 'zaid', 'summer', 'winter', 'monsoon',
                'pre-monsoon', 'post-monsoon', 'sowing', 'harvesting'
            ],
            'techniques': [
                'irrigation', 'drip', 'sprinkler', 'flooding', 'transplanting',
                'direct seeding', 'intercropping', 'crop rotation', 'mulching',
                'pruning', 'grafting', 'organic farming', 'precision agriculture'
            ]
        }
    
    def update_history(self, new_history: List[Dict]) -> None:
        """
        Update conversation history with new exchanges.
        
        Args:
            new_history: List of conversation exchanges
        """
        self.conversation_history = new_history.copy()
        self._extract_entities_from_history()
        self._apply_memory_strategy()
    
    def add_exchange(self, question: str, answer: str) -> None:
        """
        Add a new question-answer exchange to the conversation.
        
        Args:
            question: User's question
            answer: System's answer
        """
        exchange = {
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }
        
        self.conversation_history.append(exchange)
        self._extract_entities_from_exchange(exchange)
        self._apply_memory_strategy()
    
    def get_context_keywords(self) -> List[str]:
        """
        Extract relevant keywords from conversation context.
        
        Returns:
            List of important keywords for context awareness
        """
        keywords = []
        
        # Add tracked entities as keywords
        for entity_type, entities in self.tracked_entities.items():
            keywords.extend(list(entities)[:3])  # Limit per type
        
        # Add recent question keywords
        for exchange in self.conversation_history[-2:]:
            question_words = self._extract_keywords(exchange['question'])
            keywords.extend(question_words[:3])
        
        # Remove duplicates and return most relevant
        unique_keywords = list(dict.fromkeys(keywords))
        return unique_keywords[:10]
    
    def get_formatted_context(self) -> str:
        """
        Get formatted conversation context for LLM processing.
        
        Returns:
            str: Formatted context string
        """
        if not self.conversation_history and not self.conversation_summary:
            return ""
        
        context_parts = []
        
        # Add summary if available
        if self.conversation_summary:
            context_parts.append(f"Previous conversation summary: {self.conversation_summary}")
        
        # Add recent exchanges
        recent_exchanges = self.conversation_history[-self.max_context_pairs:]
        
        if recent_exchanges:
            context_parts.append("Recent conversation:")
            for i, exchange in enumerate(recent_exchanges, 1):
                context_parts.append(f"Q{i}: {exchange['question']}")
                context_parts.append(f"A{i}: {exchange['answer'][:200]}...")
        
        # Add entity context
        entity_context = self._get_entity_context()
        if entity_context:
            context_parts.append(f"Relevant agricultural topics: {entity_context}")
        
        return "\n".join(context_parts)
    
    def _apply_memory_strategy(self) -> None:
        """Apply the selected memory strategy to manage conversation length."""
        conversation_length = len(self.conversation_history)
        
        if self.memory_strategy == MemoryStrategy.AUTO:
            # Auto-select strategy based on conversation length
            if conversation_length <= self.summary_threshold:
                self._apply_buffer_strategy()
            else:
                self._apply_hybrid_strategy()
        
        elif self.memory_strategy == MemoryStrategy.BUFFER:
            self._apply_buffer_strategy()
        
        elif self.memory_strategy == MemoryStrategy.SUMMARY:
            self._apply_summary_strategy()
        
        elif self.memory_strategy == MemoryStrategy.HYBRID:
            self._apply_hybrid_strategy()
        
        elif self.memory_strategy == MemoryStrategy.SLIDING_WINDOW:
            self._apply_sliding_window_strategy()
    
    def _apply_buffer_strategy(self) -> None:
        """Keep only recent conversations in full detail."""
        if len(self.conversation_history) > self.max_context_pairs:
            self.conversation_history = self.conversation_history[-self.max_context_pairs:]
    
    def _apply_sliding_window_strategy(self) -> None:
        """Simple sliding window approach."""
        if len(self.conversation_history) > self.max_context_pairs:
            self.conversation_history = self.conversation_history[-self.max_context_pairs:]
    
    def _apply_summary_strategy(self) -> None:
        """Summarize old conversations and keep only recent ones."""
        if len(self.conversation_history) > self.summary_threshold:
            # Summarize older conversations
            old_conversations = self.conversation_history[:-self.hybrid_buffer_pairs]
            self.conversation_summary = self._summarize_conversations(old_conversations)
            
            # Keep only recent conversations
            self.conversation_history = self.conversation_history[-self.hybrid_buffer_pairs:]
    
    def _apply_hybrid_strategy(self) -> None:
        """Combination of summary for old conversations and buffer for recent ones."""
        if len(self.conversation_history) > self.summary_threshold:
            # Summarize older conversations
            old_conversations = self.conversation_history[:-self.hybrid_buffer_pairs]
            if old_conversations:
                self.conversation_summary = self._summarize_conversations(old_conversations)
            
            # Keep recent conversations in buffer
            self.conversation_history = self.conversation_history[-self.hybrid_buffer_pairs:]
    
    def _summarize_conversations(self, conversations: List[Dict]) -> str:
        """
        Create a summary of conversation exchanges.
        
        Args:
            conversations: List of conversation exchanges to summarize
            
        Returns:
            str: Summary of conversations
        """
        if not conversations:
            return ""
        
        # Extract main topics and key information
        topics = set()
        key_info = []
        
        for exchange in conversations:
            # Extract topics from questions
            question_topics = self._extract_agricultural_topics(exchange['question'])
            topics.update(question_topics)
            
            # Extract key information from answers
            answer_summary = self._extract_key_info(exchange['answer'])
            if answer_summary:
                key_info.append(answer_summary)
        
        # Build summary
        summary_parts = []
        
        if topics:
            summary_parts.append(f"Discussed topics: {', '.join(list(topics)[:5])}")
        
        if key_info:
            summary_parts.append(f"Key information: {'; '.join(key_info[:3])}")
        
        return ". ".join(summary_parts)
    
    def _extract_entities_from_history(self) -> None:
        """Extract agricultural entities from conversation history."""
        for exchange in self.conversation_history:
            self._extract_entities_from_exchange(exchange)
    
    def _extract_entities_from_exchange(self, exchange: Dict) -> None:
        """
        Extract agricultural entities from a single exchange.
        
        Args:
            exchange: Single conversation exchange
        """
        text = f"{exchange['question']} {exchange['answer']}".lower()
        
        for entity_type, patterns in self._entity_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text:
                    self.tracked_entities[entity_type].add(pattern)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of extracted keywords
        """
        # Simple keyword extraction based on length and agricultural relevance
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # Filter agricultural-relevant words
        keywords = []
        for word in words:
            if (len(word) >= 4 and 
                any(word in patterns for patterns in self._entity_patterns.values())):
                keywords.append(word)
        
        return keywords[:5]
    
    def _extract_agricultural_topics(self, text: str) -> Set[str]:
        """
        Extract agricultural topics from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Set of agricultural topics
        """
        topics = set()
        text_lower = text.lower()
        
        # Look for crop names
        for crop in self._entity_patterns['crops']:
            if crop in text_lower:
                topics.add(crop)
        
        # Look for agricultural processes
        agricultural_terms = [
            'planting', 'harvesting', 'irrigation', 'fertilizer', 'pest control',
            'disease management', 'soil preparation', 'crop rotation'
        ]
        
        for term in agricultural_terms:
            if term in text_lower:
                topics.add(term)
        
        return topics
    
    def _extract_key_info(self, text: str) -> str:
        """
        Extract key information from answer text.
        
        Args:
            text: Answer text to analyze
            
        Returns:
            str: Key information summary
        """
        # Look for specific recommendations or important facts
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 20 and 
                any(keyword in sentence.lower() for keyword in 
                    ['recommend', 'important', 'should', 'must', 'best'])):
                return sentence[:100] + "..." if len(sentence) > 100 else sentence
        
        # Fallback to first substantial sentence
        for sentence in sentences[:3]:
            if len(sentence.strip()) > 30:
                return sentence.strip()[:100] + "..." if len(sentence) > 100 else sentence.strip()
        
        return ""
    
    def _get_entity_context(self) -> str:
        """
        Get string representation of tracked entities for context.
        
        Returns:
            str: Formatted entity context
        """
        entity_summaries = []
        
        for entity_type, entities in self.tracked_entities.items():
            if entities:
                entity_list = list(entities)[:3]  # Limit to 3 per type
                entity_summaries.append(f"{entity_type}: {', '.join(entity_list)}")
        
        return "; ".join(entity_summaries[:4])  # Limit total context
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get conversation statistics and entity tracking information.
        
        Returns:
            Dict containing conversation statistics
        """
        return {
            "total_exchanges": len(self.conversation_history),
            "memory_strategy": self.memory_strategy.value,
            "has_summary": bool(self.conversation_summary),
            "tracked_entities": {
                entity_type: len(entities) 
                for entity_type, entities in self.tracked_entities.items()
            },
            "context_keywords": len(self.get_context_keywords()),
            "summary_length": len(self.conversation_summary) if self.conversation_summary else 0
        }
    
    def reset(self) -> None:
        """Reset conversation context and entity tracking."""
        self.conversation_history.clear()
        self.conversation_summary = ""
        for entity_set in self.tracked_entities.values():
            entity_set.clear()
