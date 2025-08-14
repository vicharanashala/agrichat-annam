"""
Context Manager for Chain of Thought Implementation
Handles conversation context with token optimization for follow-up queries
"""

import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re

class ConversationContext:
    """Manages conversation context with token optimization"""
    
    def __init__(self, max_context_pairs: int = 5, max_context_tokens: int = 800):
        """
        Initialize context manager
        
        Args:
            max_context_pairs: Maximum number of Q&A pairs to maintain in context (increased to 5)
            max_context_tokens: Maximum estimated tokens for context (increased for more context)
        """
        self.max_context_pairs = max_context_pairs
        self.max_context_tokens = max_context_tokens
        
    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation (approximately 4 characters per token)
        This is a simplified estimation for English text
        """
        return len(text) // 4
    
    def _is_followup_query(self, current_query: str, previous_messages: List[Dict]) -> bool:
        """
        Determine if current query is a follow-up based on linguistic patterns
        """
        if not previous_messages:
            print(f"[Context DEBUG] No previous messages, not a follow-up")
            return False
            
        followup_patterns = [
            r'\b(what about|how about|what if|but what|and what)\b',
            r'\b(also|additionally|furthermore|moreover)\b',
            r'\b(more|further|deeper|detail)\b',
            r'\b(this|that|it|they|them)\b',
            r'\b(above|mentioned|previous|earlier)\b',
            r'\b(explain|elaborate|expand)\b',
            r'^\s*(and|but|however|though|although)\b',
            r'\?.*\?',
            r'\b(more about|tell me more|can you explain|explain more)\b',
            r'\b(what else|anything else|other|another)\b',
            r'^\s*(why|how|when|where|who)\b.*\?',
            r'\b(yes|ok|okay|sure|please)\b.*\b(help|tell|explain|more)\b',
            r'^\s*(yes|ok|okay|sure|please)\s*$',
        ]
        
        current_lower = current_query.lower().strip()
        
        print(f"[Context DEBUG] Analyzing query for follow-up patterns: '{current_lower}'")
        
        for pattern in followup_patterns:
            if re.search(pattern, current_lower):
                print(f"[Context DEBUG] Matched follow-up pattern: {pattern}")
                return True
        
        if len(current_lower.split()) <= 4 and re.search(r'\b(it|this|that|them|they)\b', current_lower):
            print(f"[Context DEBUG] Matched short pronoun query")
            return True
                
        if len(previous_messages) > 0:
            last_qa = previous_messages[-1]
            last_question = last_qa.get('question', '').lower()
            last_answer = last_qa.get('answer', '').lower()
            
            current_words = set(re.findall(r'\b\w{4,}\b', current_lower))
            last_words = set(re.findall(r'\b\w{4,}\b', last_question + ' ' + last_answer))
            
            overlap = current_words.intersection(last_words)
            overlap_ratio = len(overlap) / max(len(current_words), 1)
            
            print(f"[Context DEBUG] Word overlap: {len(overlap)} words, ratio: {overlap_ratio:.2f}")
            print(f"[Context DEBUG] Overlapping words: {overlap}")
            
            if len(overlap) >= 2 or overlap_ratio > 0.3:
                print(f"[Context DEBUG] Matched word overlap criteria")
                return True
                
        print(f"[Context DEBUG] No follow-up patterns detected")
        return False
    
    def _extract_key_context(self, messages: List[Dict]) -> str:
        """
        Extract key context from previous messages with token optimization
        Prioritizes recent messages and agricultural terms, now supporting 5 message history
        """
        if not messages:
            return ""
            
        # Use all 5 messages if available
        recent_messages = messages[-self.max_context_pairs:]
        
        context_parts = []
        total_tokens = 0
        
        # Enhanced agricultural keywords for better context extraction
        agri_keywords = {
            'crop', 'farming', 'soil', 'fertilizer', 'pesticide', 'irrigation', 
            'seed', 'harvest', 'disease', 'pest', 'weather', 'climate', 
            'yield', 'production', 'agriculture', 'plant', 'growth', 'cultivation',
            'planting', 'sowing', 'organic', 'compost', 'nutrients', 'water',
            'rice', 'wheat', 'cotton', 'maize', 'sugarcane', 'tomato', 'potato',
            'farm', 'farmer', 'field', 'land', 'season', 'kharif', 'rabi'
        }
        
        # Process messages with weighted importance (most recent gets highest priority)
        for i, msg in enumerate(reversed(recent_messages)):
            question = msg.get('question', '')
            answer = msg.get('answer', '')
            
            # Extract the most relevant sentences from answer
            answer_sentences = re.split(r'[.!?]+', answer)
            key_sentences = []
            
            # Prioritize sentences with agricultural terms
            for sentence in answer_sentences[:4]:  # Look at more sentences for better context
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:  # Filter out very short sentences
                    sentence_lower = sentence.lower()
                    agri_score = sum(1 for keyword in agri_keywords if keyword in sentence_lower)
                    
                    # Prioritize agricultural sentences
                    if agri_score > 0:
                        key_sentences.insert(0, sentence)
                    else:
                        key_sentences.append(sentence)
            
            # Create context entry with appropriate weight indicators
            if i == 0:
                weight_indicator = "Most Recent"
            elif i == 1:
                weight_indicator = "Recent"
            elif i == 2:
                weight_indicator = "Previous"
            elif i == 3:
                weight_indicator = "Earlier"
            else:
                weight_indicator = "Historical"
            
            # Build context entry with more detailed information for recent messages
            if key_sentences:
                if i <= 1:  # For the 2 most recent messages, include more detail
                    context_entry = f"{weight_indicator}: Q: {question[:120]} A: {'. '.join(key_sentences[:3])}"
                else:  # For older messages, be more concise
                    context_entry = f"{weight_indicator}: Q: {question[:80]} A: {key_sentences[0][:100]}"
            else:
                context_entry = f"{weight_indicator}: Q: {question[:100]}"
            
            # Check token limit
            entry_tokens = self._estimate_tokens(context_entry)
            if total_tokens + entry_tokens > self.max_context_tokens:
                # Try to fit a shorter version
                if i <= 1:  # Always try to include the most recent 2 messages
                    short_entry = f"{weight_indicator}: Q: {question[:60]} A: {key_sentences[0][:50] if key_sentences else 'No specific answer'}"
                    short_tokens = self._estimate_tokens(short_entry)
                    if total_tokens + short_tokens <= self.max_context_tokens:
                        context_parts.append(short_entry)
                        total_tokens += short_tokens
                break
                
            context_parts.append(context_entry)
            total_tokens += entry_tokens
        
        return "\n".join(context_parts)
    
    def should_use_context(self, current_query: str, conversation_history: List[Dict]) -> bool:
        """
        Determine if context should be used for the current query
        Now considers up to 5 previous messages for better context detection
        """
        if not conversation_history:
            return False
            
        # Check the last 5 messages for context clues
        recent_messages = conversation_history[-5:] if len(conversation_history) >= 5 else conversation_history
        
        return self._is_followup_query(current_query, recent_messages)
    
    def build_contextual_query(self, current_query: str, conversation_history: List[Dict]) -> str:
        """
        Build a contextual query incorporating relevant conversation history
        
        Args:
            current_query: The current user question
            conversation_history: List of previous Q&A pairs
            
        Returns:
            Enhanced query with context or original query if no context needed
        """
        if not self.should_use_context(current_query, conversation_history):
            return current_query
            
        context = self._extract_key_context(conversation_history)
        
        if not context:
            return current_query
            
        contextual_query = f"""Context from our conversation:
{context}

Current question: {current_query}

Please provide a response considering the above conversation context, giving more importance to recent interactions."""
        
        return contextual_query
    
    def get_context_summary(self, conversation_history: List[Dict]) -> Optional[str]:
        """
        Get a brief summary of the conversation context for debugging
        Now analyzes up to 5 messages for better topic tracking
        """
        if not conversation_history:
            return None
            
        # Analyze the last 5 messages for topic extraction
        recent_topics = []
        crop_mentions = []
        
        for msg in conversation_history[-5:]:
            question = msg.get('question', '')
            answer = msg.get('answer', '')
            
            # Extract agricultural terms from both question and answer
            combined_text = (question + ' ' + answer).lower()
            
            # Extract general agricultural terms
            key_terms = re.findall(r'\b(?:crop|farming|soil|fertilizer|pesticide|irrigation|seed|harvest|disease|pest|weather|yield|agriculture|plant|cultivation|organic|compost)\w*\b', 
                                 combined_text)
            if key_terms:
                recent_topics.extend(key_terms[:3])  # Take more terms for better context
            
            # Extract specific crop names
            crop_terms = re.findall(r'\b(?:rice|wheat|cotton|maize|sugarcane|tomato|potato|onion|chili|groundnut|soybean|mustard|barley|millets)\w*\b',
                                  combined_text)
            if crop_terms:
                crop_mentions.extend(crop_terms[:2])
                
        topic_summary = []
        if crop_mentions:
            topic_summary.append(f"Crops: {', '.join(set(crop_mentions))}")
        if recent_topics:
            topic_summary.append(f"Topics: {', '.join(set(recent_topics))}")
            
        if topic_summary:
            return "; ".join(topic_summary)
        else:
            return f"Last {len(conversation_history)} general interactions"

    def test_context_detection(self, current_query: str, conversation_history: List[Dict]) -> Dict:
        """
        Test method to check context detection - useful for debugging
        """
        result = {
            "query": current_query,
            "should_use_context": self.should_use_context(current_query, conversation_history),
            "is_followup": self._is_followup_query(current_query, conversation_history[-3:] if conversation_history else []),
            "context_summary": self.get_context_summary(conversation_history)
        }
        
        if self.should_use_context(current_query, conversation_history):
            result["enhanced_query"] = self.build_contextual_query(current_query, conversation_history)
        
        return result
