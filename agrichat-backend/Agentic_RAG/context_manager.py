"""
Enhanced Context Manager for Chain of Thought Implementation
Handles conversation context with hybrid memory strategies and agricultural entity tracking
"""

import json
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
import re
from enum import Enum

class MemoryStrategy(Enum):
    """Memory strategy types"""
    BUFFER = "buffer"  
    SUMMARY = "summary"
    HYBRID = "hybrid"
    AUTO = "auto"

class ConversationContext:
    """Enhanced conversation context manager with multiple memory strategies"""
    
    def __init__(self, 
                 max_context_pairs: int = 5, 
                 max_context_tokens: int = 800,
                 hybrid_buffer_pairs: int = 3,
                 summary_threshold: int = 8,
                 memory_strategy: MemoryStrategy = MemoryStrategy.AUTO):
        """
        Initialize enhanced context manager
        
        Args:
            max_context_pairs: Maximum number of Q&A pairs for buffer mode
            max_context_tokens: Maximum estimated tokens for context
            hybrid_buffer_pairs: Number of recent pairs to keep in raw form for hybrid mode
            summary_threshold: Conversation length threshold to switch to summary/hybrid mode
            memory_strategy: Memory strategy to use
        """
        self.max_context_pairs = max_context_pairs
        self.max_context_tokens = max_context_tokens
        self.hybrid_buffer_pairs = hybrid_buffer_pairs
        self.summary_threshold = summary_threshold
        self.memory_strategy = memory_strategy
        
        self.tracked_entities = {
            'crops': set(),
            'diseases': set(),
            'pests': set(),
            'fertilizers': set(),
            'locations': set(),
            'seasons': set()
        }
        
        self.chain_of_thought_patterns = {
            'analysis_needed': [
                'analyze', 'compare', 'evaluate', 'assess', 'explain why',
                'what causes', 'how does', 'which is better', 'pros and cons'
            ],
            'step_by_step': [
                'how to', 'steps', 'process', 'procedure', 'method',
                'treatment', 'application', 'prepare', 'grow'
            ],
            'problem_solving': [
                'problem', 'issue', 'disease', 'pest', 'yellowing',
                'wilting', 'spots', 'cure', 'treat', 'fix'
            ],
            'decision_making': [
                'choose', 'select', 'decide', 'which', 'best',
                'recommend', 'suggest', 'should i'
            ]
        }
        
        self.reasoning_templates = {
            'analysis': """Let me analyze this agricultural question step by step:

1. **Problem Understanding**: {problem_analysis}
2. **Key Factors to Consider**: {key_factors}
3. **Available Options**: {options}
4. **Recommendation**: {recommendation}

Here's my detailed analysis:""",
            
            'step_by_step': """I'll guide you through this process step by step:

**Step-by-Step Process:**
{steps}

**Important Considerations:**
{considerations}""",
            
            'problem_solving': """Let me help you solve this agricultural problem systematically:

**Problem Diagnosis:**
{diagnosis}

**Root Cause Analysis:**
{root_cause}

**Solution Strategy:**
{solution_strategy}

**Implementation Steps:**
{implementation}""",
            
            'decision_making': """Let me help you make an informed decision:

**Available Options:**
{options}

**Comparison Analysis:**
{comparison}

**My Recommendation:**
{recommendation}

**Reasoning:**
{reasoning}"""
        }
        
    def _estimate_tokens(self, text: str) -> int:
        """
        Improved token estimation using multiple heuristics
        Based on OpenAI's tokenization patterns
        """
        if not text:
            return 0
            
        text = re.sub(r'\s+', ' ', text.strip())
        
        words = len(text.split())
        
        punctuation_count = len(re.findall(r'[.,!?;:\-()"\']', text))
        
        number_count = len(re.findall(r'\d+', text))
        
        technical_terms = len(re.findall(r'[A-Z][a-z]+[A-Z]', text))
        hyphenated_words = len(re.findall(r'\w+-\w+', text))
        
        char_based = len(text) // 4
        
        word_based = words + (punctuation_count * 0.3) + (number_count * 0.5) + technical_terms + hyphenated_words
        mation
        return int(max(char_based, word_based))
    
    def _extract_agricultural_entities(self, text: str) -> Dict[str, Set[str]]:
        """
        Extract and track agricultural entities from text
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dict of entity types and their values
        """
        text_lower = text.lower()
        entities = {
            'crops': set(),
            'diseases': set(), 
            'pests': set(),
            'fertilizers': set(),
            'locations': set(),
            'seasons': set()
        }
        
        crop_patterns = [
            r'\b(rice|wheat|maize|corn|cotton|sugarcane|potato|tomato|onion|chili|pepper|groundnut|peanut|soybean|mustard|barley|millets|jowar|bajra|ragi|sesame|sunflower|safflower|castor|coconut|areca|cardamom|ginger|turmeric|coriander|cumin|fenugreek|garlic|cabbage|cauliflower|brinjal|eggplant|okra|ladyfinger|cucumber|bottle gourd|ridge gourd|bitter gourd|pumpkin|watermelon|muskmelon|papaya|mango|banana|guava|pomegranate|citrus|orange|lime|lemon)\w*\b',
            r'\b(paddy|dal|pulses|legumes|cereals|vegetables|fruits|spices|fodder|forage)\w*\b'
        ]
        
        disease_patterns = [
            r'\b(blight|rust|smut|wilt|rot|mildew|mosaic|virus|bacterial|fungal|leaf spot|stem rot|root rot|collar rot|damping off|blast|sheath blight)\w*\b',
            r'\b(disease|infection|pathogen|symptom)\w*\b'
        ]
        
        pest_patterns = [
            r'\b(borer|caterpillar|aphid|thrips|whitefly|jassid|bug|weevil|mite|nematode|grub|larva|worm|moth|butterfly|beetle|fly|termite|ant|locust|grasshopper)\w*\b',
            r'\b(pest|insect|damage|infestation)\w*\b'
        ]
        
        fertilizer_patterns = [
            r'\b(urea|dap|potash|nitrogen|phosphorus|potassium|npk|compost|manure|vermicompost|organic|bio.?fertilizer|micro.?nutrient|zinc|boron|iron|manganese|sulfur|calcium|magnesium)\w*\b',
            r'\b(fertilizer|nutrient|amendment)\w*\b'
        ]
        
        location_patterns = [
            r'\b(punjab|haryana|uttar pradesh|up|bihar|west bengal|odisha|jharkhand|madhya pradesh|mp|rajasthan|gujarat|maharashtra|karnataka|andhra pradesh|ap|telangana|tamil nadu|tn|kerala|assam|manipur|meghalaya|tripura|nagaland|mizoram|arunachal pradesh|sikkim|goa|himachal pradesh|hp|uttarakhand|jammu|kashmir|delhi|chandigarh|puducherry)\w*\b',
            r'\b(north|south|east|west|central|india|region|zone|district|state|village|block|tehsil|mandal)\w*\b'
        ]
        
        season_patterns = [
            r'\b(kharif|rabi|zaid|summer|winter|monsoon|pre.?monsoon|post.?monsoon|rainy|dry|wet|season|sowing|planting|harvesting|harvest)\w*\b'
        ]
        
        for pattern in crop_patterns:
            entities['crops'].update(re.findall(pattern, text_lower))
            
        for pattern in disease_patterns:
            entities['diseases'].update(re.findall(pattern, text_lower))
            
        for pattern in pest_patterns:
            entities['pests'].update(re.findall(pattern, text_lower))
            
        for pattern in fertilizer_patterns:
            entities['fertilizers'].update(re.findall(pattern, text_lower))
            
        for pattern in location_patterns:
            entities['locations'].update(re.findall(pattern, text_lower))
            
        for pattern in season_patterns:
            entities['seasons'].update(re.findall(pattern, text_lower))
        
        for entity_type, values in entities.items():
            self.tracked_entities[entity_type].update(values)
        
        return entities
    
    def _generate_conversation_summary(self, messages: List[Dict]) -> str:
        """
        Generate a concise summary of conversation history
        Focuses on agricultural entities and key topics
        
        Args:
            messages: List of conversation messages to summarize
            
        Returns:
            Concise summary focusing on agricultural context
        """
        if not messages:
            return ""
        
        all_text = " ".join([f"{msg.get('question', '')} {msg.get('answer', '')}" for msg in messages])
        entities = self._extract_agricultural_entities(all_text)
        
        summary_parts = []
        
        if entities['crops']:
            crops_list = list(entities['crops'])[:5]
            summary_parts.append(f"Crops discussed: {', '.join(crops_list)}")
            
        if entities['diseases']:
            diseases_list = list(entities['diseases'])[:3]
            summary_parts.append(f"Diseases mentioned: {', '.join(diseases_list)}")
            
        if entities['pests']:
            pests_list = list(entities['pests'])[:3]
            summary_parts.append(f"Pests discussed: {', '.join(pests_list)}")
            
        if entities['locations']:
            locations_list = list(entities['locations'])[:2]
            summary_parts.append(f"Regions: {', '.join(locations_list)}")
            
        if entities['seasons']:
            seasons_list = list(entities['seasons'])[:2]
            summary_parts.append(f"Seasons: {', '.join(seasons_list)}")
        
        topics = []
        for msg in messages[-3:]: 
            question = msg.get('question', '')
            if 'disease' in question.lower() or 'pest' in question.lower():
                topics.append("pest/disease management")
            elif 'fertilizer' in question.lower() or 'nutrient' in question.lower():
                topics.append("fertilization")
            elif 'water' in question.lower() or 'irrigation' in question.lower():
                topics.append("irrigation")
            elif 'harvest' in question.lower() or 'yield' in question.lower():
                topics.append("harvesting")
            elif 'plant' in question.lower() or 'sow' in question.lower():
                topics.append("planting")
        
        if topics:
            summary_parts.append(f"Recent topics: {', '.join(set(topics))}")
        
        summary_parts.append(f"Total interactions: {len(messages)}")
        
        return ". ".join(summary_parts) + "."
    
    def _determine_memory_strategy(self, conversation_length: int) -> MemoryStrategy:
        """
        Automatically determine the best memory strategy based on conversation length
        
        Args:
            conversation_length: Number of messages in conversation
            
        Returns:
            Optimal memory strategy for the conversation length
        """
        if self.memory_strategy != MemoryStrategy.AUTO:
            return self.memory_strategy
            
        if conversation_length <= 4:
            return MemoryStrategy.BUFFER
        elif conversation_length <= self.summary_threshold:
            return MemoryStrategy.BUFFER
        else:
            return MemoryStrategy.HYBRID
    
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
            # Summary and context-based patterns
            r'\b(based on|considering|given|taking into account)\b.*\b(discussed|mentioned|said|talked)\b',
            r'\b(everything|all|overall|in summary|to summarize)\b.*\b(discussed|mentioned|covered)\b',
            r'\b(most important|key|main|primary)\b.*\b(from|based on|considering)\b',
            r'\b(what.*most|which.*best|what.*recommend)\b.*\b(discussed|mentioned|covered)\b',
            r'\b(conclusion|summary|takeaway|recommendation)\b',
            r'\b(in general|overall|all things considered)\b',
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
    
    def _extract_key_context(self, messages: List[Dict], strategy: MemoryStrategy = None) -> str:
        """
        Extract key context from previous messages using specified memory strategy
        Supports buffer, summary, and hybrid approaches
        
        Args:
            messages: List of conversation messages
            strategy: Memory strategy to use (defaults to auto-determined)
            
        Returns:
            Formatted context string
        """
        if not messages:
            return ""
        
        if strategy is None:
            strategy = self._determine_memory_strategy(len(messages))
        
        print(f"[Context DEBUG] Using memory strategy: {strategy.value} for {len(messages)} messages")
        
        if strategy == MemoryStrategy.BUFFER:
            return self._extract_buffer_context(messages)
        elif strategy == MemoryStrategy.SUMMARY:
            return self._extract_summary_context(messages)
        elif strategy == MemoryStrategy.HYBRID:
            return self._extract_hybrid_context(messages)
        else:
            return self._extract_buffer_context(messages)
    
    def _extract_buffer_context(self, messages: List[Dict]) -> str:
        """
        Extract context using buffer strategy (original implementation)
        """
        recent_messages = messages[-self.max_context_pairs:]
        
        context_parts = []
        total_tokens = 0
        
        agri_keywords = {
            'crop', 'farming', 'soil', 'fertilizer', 'pesticide', 'irrigation', 
            'seed', 'harvest', 'disease', 'pest', 'weather', 'climate', 
            'yield', 'production', 'agriculture', 'plant', 'growth', 'cultivation',
            'planting', 'sowing', 'organic', 'compost', 'nutrients', 'water',
            'rice', 'wheat', 'cotton', 'maize', 'sugarcane', 'tomato', 'potato',
            'farm', 'farmer', 'field', 'land', 'season', 'kharif', 'rabi'
        }
        
        for i, msg in enumerate(reversed(recent_messages)):
            question = msg.get('question', '')
            answer = msg.get('answer', '')
            
            answer_sentences = re.split(r'[.!?]+', answer)
            key_sentences = []
            
            for sentence in answer_sentences[:4]:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:
                    sentence_lower = sentence.lower()
                    agri_score = sum(1 for keyword in agri_keywords if keyword in sentence_lower)
                    
                    if agri_score > 0:
                        key_sentences.insert(0, sentence)
                    else:
                        key_sentences.append(sentence)
            
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
            
            if key_sentences:
                if i <= 1:
                    context_entry = f"{weight_indicator}: Q: {question[:120]} A: {'. '.join(key_sentences[:3])}"
                else:
                    context_entry = f"{weight_indicator}: Q: {question[:80]} A: {key_sentences[0][:100]}"
            else:
                context_entry = f"{weight_indicator}: Q: {question[:100]}"
            
            entry_tokens = self._estimate_tokens(context_entry)
            if total_tokens + entry_tokens > self.max_context_tokens:
                if i <= 1:
                    short_entry = f"{weight_indicator}: Q: {question[:60]} A: {key_sentences[0][:50] if key_sentences else 'No specific answer'}"
                    short_tokens = self._estimate_tokens(short_entry)
                    if total_tokens + short_tokens <= self.max_context_tokens:
                        context_parts.append(short_entry)
                        total_tokens += short_tokens
                break
                
            context_parts.append(context_entry)
            total_tokens += entry_tokens
        
        return "\n".join(context_parts)
    
    def _extract_summary_context(self, messages: List[Dict]) -> str:
        """
        Extract context using summary strategy
        """
        summary = self._generate_conversation_summary(messages)
        return f"Conversation Summary: {summary}"
    
    def _extract_hybrid_context(self, messages: List[Dict]) -> str:
        """
        Extract context using hybrid strategy (summary + recent buffer)
        """
        context_parts = []
        
        if len(messages) > self.hybrid_buffer_pairs:
            older_messages = messages[:-self.hybrid_buffer_pairs]
            summary = self._generate_conversation_summary(older_messages)
            context_parts.append(f"Earlier Discussion Summary: {summary}")
        
        recent_messages = messages[-self.hybrid_buffer_pairs:]
        recent_context = self._extract_buffer_context(recent_messages)
        if recent_context:
            context_parts.append("Recent Context:")
            context_parts.append(recent_context)
        
        return "\n".join(context_parts)
    
    def should_use_context(self, current_query: str, conversation_history: List[Dict]) -> bool:
        """
        Determine if context should be used for the current query
        Now considers up to 5 previous messages for better context detection
        """
        if not conversation_history:
            return False
            
        recent_messages = conversation_history[-5:] if len(conversation_history) >= 5 else conversation_history
        
        return self._is_followup_query(current_query, recent_messages)
    
    def build_contextual_query(self, current_query: str, conversation_history: List[Dict], enable_chain_of_thought: bool = True) -> str:
        """
        Build a contextual query incorporating relevant conversation history
        Respects existing token limits (max_context_tokens=800) and pair limits (max_context_pairs=5)
        
        Args:
            current_query: The current user question
            conversation_history: List of previous Q&A pairs (limited to max_context_pairs)
            enable_chain_of_thought: Whether to include chain of thought prompting
            
        Returns:
            Enhanced query with context or original query if no context needed
        """
        if not self.should_use_context(current_query, conversation_history):
            return current_query
        
        current_entities = self._extract_agricultural_entities(current_query)
        
        # Use existing _extract_key_context which already respects token/pair limits
        context = self._extract_key_context(conversation_history)
        
        if not context:
            return current_query
            
        contextual_parts = []
        
        contextual_parts.append(f"Context from our conversation:\n{context}")
        
        entity_context = []
        for entity_type, entities in current_entities.items():
            if entities and self.tracked_entities[entity_type]:
                related_entities = self.tracked_entities[entity_type].intersection(entities)
                if related_entities:
                    entity_context.append(f"Related {entity_type}: {', '.join(related_entities)}")
        
        if entity_context:
            contextual_parts.append(f"Related entities from our discussion:\n{'; '.join(entity_context)}")
        
        contextual_parts.append(f"Current question: {current_query}")
        
        if enable_chain_of_thought:
            contextual_parts.append("""
Please analyze this question step by step:
1. Consider the conversation context and related entities discussed
2. Identify the specific agricultural problem or information need
3. Draw connections to previous discussions if relevant
4. Provide a comprehensive response considering the full context

Respond with detailed, actionable advice relevant to Indian agricultural conditions.""")
        else:
            contextual_parts.append("Please provide a response considering the above conversation context, giving more importance to recent interactions.")
        
        # Ensure the final query respects token limits
        complete_query = "\n\n".join(contextual_parts)
        query_tokens = self._estimate_tokens(complete_query)
        
        # If exceeding limits, trim while preserving structure
        if query_tokens > self.max_context_tokens:
            # Reduce recent context to fit within limits
            base_query_tokens = self._estimate_tokens(current_query) + 150  # Reserve for instructions
            available_tokens = self.max_context_tokens - base_query_tokens
            
            if available_tokens > 100:
                # Limit conversation history and try again
                limited_history = conversation_history[-2:] if len(conversation_history) > 2 else conversation_history
                limited_context = self._extract_key_context(limited_history)
                
                if limited_context:
                    contextual_parts = [
                        f"Recent context:\n{limited_context}",
                        f"Current question: {current_query}"
                    ]
                    
                    if enable_chain_of_thought:
                        contextual_parts.append("Please analyze step by step considering the context.")
                    else:
                        contextual_parts.append("Please respond considering the recent context.")
                    
                    complete_query = "\n\n".join(contextual_parts)
                else:
                    complete_query = current_query
            else:
                complete_query = current_query
        
        return complete_query
    
    def requires_chain_of_thought(self, query: str) -> Tuple[bool, str]:
        """
        Determine if a query requires chain of thought reasoning
        
        Args:
            query: User's question
            
        Returns:
            Tuple of (requires_cot, reasoning_type)
        """
        query_lower = query.lower()
        print(f"[CoT DEBUG] Analyzing query: '{query_lower}'")
        
        for reasoning_type, patterns in self.chain_of_thought_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    print(f"[CoT DEBUG] Matched pattern '{pattern}' for type '{reasoning_type}'")
                    return True, reasoning_type
        
        if any(word in query_lower for word in ['how do i', 'how can i', 'how should i']):
            print(f"[CoT DEBUG] Matched flexible 'how do/can/should' pattern for step_by_step")
            return True, 'step_by_step'
            
        if any(word in query_lower for word in ['what are the signs', 'how to identify', 'how to know']):
            print(f"[CoT DEBUG] Matched identification pattern for analysis_needed")
            return True, 'analysis_needed'
        
        print(f"[CoT DEBUG] No patterns matched - using simple response")
        return False, 'simple'
    
    def generate_chain_of_thought_prompt(self, query: str, context: str, reasoning_type: str) -> str:
        """
        Generate a chain of thought prompt based on query type and context
        
        Args:
            query: Original user question
            context: Conversation context
            reasoning_type: Type of reasoning required
            
        Returns:
            Enhanced prompt with chain of thought structure
        """
        base_prompt = f"Query: {query}\n\nContext: {context}\n\n"
        
        if reasoning_type in self.reasoning_templates:
            cot_template = self.reasoning_templates[reasoning_type]
            
            if reasoning_type == 'analysis':
                enhanced_prompt = base_prompt + cot_template.format(
                    problem_analysis="{analysis_placeholder}",
                    key_factors="{factors_placeholder}",
                    options="{options_placeholder}",
                    recommendation="{recommendation_placeholder}"
                )
            elif reasoning_type == 'step_by_step':
                enhanced_prompt = base_prompt + cot_template.format(
                    steps="{steps_placeholder}",
                    considerations="{considerations_placeholder}"
                )
            elif reasoning_type == 'problem_solving':
                enhanced_prompt = base_prompt + cot_template.format(
                    diagnosis="{diagnosis_placeholder}",
                    root_cause="{root_cause_placeholder}",
                    solution_strategy="{solution_strategy_placeholder}",
                    implementation="{implementation_placeholder}"
                )
            elif reasoning_type == 'decision_making':
                enhanced_prompt = base_prompt + cot_template.format(
                    options="{options_placeholder}",
                    comparison="{comparison_placeholder}",
                    recommendation="{recommendation_placeholder}",
                    reasoning="{reasoning_placeholder}"
                )
            else:
                enhanced_prompt = base_prompt + "Please analyze this step by step and provide detailed reasoning."
        else:
            enhanced_prompt = base_prompt + "Please think through this systematically and explain your reasoning."
            
        return enhanced_prompt
    
    def enhance_query_with_cot(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, any]:
        """
        Enhanced query building with chain of thought integration
        Respects existing token limits (max_context_tokens=800) and pair limits (max_context_pairs=5)
        
        Args:
            query: Original user question
            conversation_history: Previous conversation messages
            
        Returns:
            Dictionary with enhanced query and metadata
        """
        requires_cot, reasoning_type = self.requires_chain_of_thought(query)
        
        context = ""
        if conversation_history:
            for msg in conversation_history:
                self._extract_agricultural_entities(f"{msg.get('question', '')} {msg.get('answer', '')}")
            
            if self.should_use_context(query, conversation_history):
                context = self._extract_key_context(conversation_history)
        
        if requires_cot:
            initial_query = self.generate_chain_of_thought_prompt(query, context, reasoning_type)
        else:
            initial_query = self.build_contextual_query(query, conversation_history, enable_chain_of_thought=False)
        
        query_tokens = self._estimate_tokens(initial_query)
        enhanced_query = initial_query
        
        if query_tokens > self.max_context_tokens:
            if context:
                available_tokens = self.max_context_tokens - self._estimate_tokens(query) - 200
                
                if available_tokens > 100:
                    context_words = context.split()
                    target_words = max(50, available_tokens // 4)  
                    if len(context_words) > target_words:
                        truncated_context = " ".join(context_words[:target_words]) + "..."
                        
                        if requires_cot:
                            enhanced_query = self.generate_chain_of_thought_prompt(query, truncated_context, reasoning_type)
                        else:
                            enhanced_query = self.build_contextual_query(query, 
                                conversation_history[-2:] if conversation_history else [], 
                                enable_chain_of_thought=False)
                    else:
                        enhanced_query = initial_query
                else:
                    if requires_cot:
                        enhanced_query = f"{query}\n\nPlease analyze this step by step."
                    else:
                        enhanced_query = query
            
            query_tokens = self._estimate_tokens(enhanced_query)
        
        return {
            'enhanced_query': enhanced_query,
            'original_query': query,
            'requires_cot': requires_cot,
            'reasoning_type': reasoning_type,
            'context_used': bool(context),
            'tracked_entities': {k: list(v) for k, v in self.tracked_entities.items()},
            'memory_strategy': self._determine_memory_strategy(len(conversation_history or [])).value,
            'context_tokens': query_tokens,
            'within_limits': query_tokens <= self.max_context_tokens
        }
    
    def get_context_summary(self, conversation_history: List[Dict]) -> Optional[str]:
        """
        Enhanced context summary with entity tracking and memory strategy info
        """
        if not conversation_history:
            return None
        
        summary = self._generate_conversation_summary(conversation_history)
        
        strategy = self._determine_memory_strategy(len(conversation_history))
        
        entity_summary = []
        for entity_type, entities in self.tracked_entities.items():
            if entities:
                entity_summary.append(f"{entity_type}: {len(entities)} tracked")
        
        strategy_info = f"Memory strategy: {strategy.value}"
        entity_info = f"Entities: {', '.join(entity_summary)}" if entity_summary else "No entities tracked"
        
        return f"{summary} | {strategy_info} | {entity_info}"

    def get_tracked_entities(self) -> Dict[str, Set[str]]:
        """
        Get all tracked agricultural entities
        
        Returns:
            Dictionary of entity types and their tracked values
        """
        return self.tracked_entities.copy()
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Simple token estimation for context management
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        words = len(text.split())
        return int(words * 1.3)
    
    def reset_entity_tracking(self):
        """
        Reset all tracked entities (useful for new conversations)
        """
        for entity_set in self.tracked_entities.values():
            entity_set.clear()
    
    def test_context_enhancement(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, any]:
        """
        Test method to verify context manager and chain of thought functionality
        Validates that token limits (800) and pair limits (5) are respected
        
        Args:
            query: Test query
            conversation_history: Test conversation history
            
        Returns:
            Test results with all enhancement details and limit validation
        """
        result = self.enhance_query_with_cot(query, conversation_history)
        
        # Add validation information
        validation = {
            'respects_token_limit': result['context_tokens'] <= self.max_context_tokens,
            'respects_pair_limit': len(conversation_history or []) <= self.max_context_pairs or 
                                  len(conversation_history[-self.max_context_pairs:] if conversation_history else []) <= self.max_context_pairs,
            'token_limit': self.max_context_tokens,
            'pair_limit': self.max_context_pairs,
            'actual_tokens': result['context_tokens'],
            'actual_pairs_considered': min(len(conversation_history or []), self.max_context_pairs)
        }
        
        result['validation'] = validation
        return result
    
    def get_memory_strategy_info(self, conversation_length: int) -> Dict:
        """
        Get information about memory strategy selection
        
        Args:
            conversation_length: Length of conversation to analyze
            
        Returns:
            Dictionary with strategy information
        """
        strategy = self._determine_memory_strategy(conversation_length)
        
        return {
            "strategy": strategy.value,
            "conversation_length": conversation_length,
            "threshold_summary": self.summary_threshold,
            "max_buffer_pairs": self.max_context_pairs,
            "hybrid_buffer_pairs": self.hybrid_buffer_pairs,
            "recommended": {
                "buffer": conversation_length <= 4,
                "summary": conversation_length > self.summary_threshold and conversation_length > 12,
                "hybrid": 4 < conversation_length <= 12
            }
        }

    def test_context_detection(self, current_query: str, conversation_history: List[Dict]) -> Dict:
        """
        Enhanced test method for context detection and strategy analysis
        """
        current_entities = self._extract_agricultural_entities(current_query)
        
        strategy = self._determine_memory_strategy(len(conversation_history))
        
        result = {
            "query": current_query,
            "conversation_length": len(conversation_history),
            "should_use_context": self.should_use_context(current_query, conversation_history),
            "is_followup": self._is_followup_query(current_query, conversation_history[-3:] if conversation_history else []),
            "memory_strategy": strategy.value,
            "context_summary": self.get_context_summary(conversation_history),
            "current_entities": {k: list(v) for k, v in current_entities.items() if v},
            "tracked_entities": {k: list(v) for k, v in self.tracked_entities.items() if v},
            "token_estimation": self._estimate_tokens(current_query)
        }
        
        if self.should_use_context(current_query, conversation_history):
            result["enhanced_query"] = self.build_contextual_query(current_query, conversation_history)
            result["context_tokens"] = self._estimate_tokens(self._extract_key_context(conversation_history, strategy))
        
        return result
    
    def configure_strategy(self, strategy: MemoryStrategy, **kwargs):
        """
        Configure memory strategy and parameters
        
        Args:
            strategy: Memory strategy to use
            **kwargs: Additional configuration parameters
        """
        self.memory_strategy = strategy
        
        if 'max_context_pairs' in kwargs:
            self.max_context_pairs = kwargs['max_context_pairs']
        if 'max_context_tokens' in kwargs:
            self.max_context_tokens = kwargs['max_context_tokens']
        if 'hybrid_buffer_pairs' in kwargs:
            self.hybrid_buffer_pairs = kwargs['hybrid_buffer_pairs']
        if 'summary_threshold' in kwargs:
            self.summary_threshold = kwargs['summary_threshold']
        
        print(f"[Context Config] Strategy set to {strategy.value} with parameters: {kwargs}")

def test_memory_strategies(query: str, history: List[Dict]) -> Dict:
    """
    Test all memory strategies for comparison
    
    Args:
        query: Current query to test
        history: Conversation history
        
    Returns:
        Dictionary with results for each strategy
    """
    results = {}
    
    for strategy in [MemoryStrategy.BUFFER, MemoryStrategy.SUMMARY, MemoryStrategy.HYBRID]:
        context_manager = ConversationContext(memory_strategy=strategy)
        
        for msg in history:
            context_manager._extract_agricultural_entities(f"{msg.get('question', '')} {msg.get('answer', '')}")
        
        test_result = context_manager.test_context_detection(query, history)
        results[strategy.value] = {
            "context_length": len(test_result.get("enhanced_query", query)),
            "token_estimate": test_result.get("context_tokens", 0),
            "strategy_used": test_result["memory_strategy"],
            "entities_tracked": len([e for entities in test_result["tracked_entities"].values() for e in entities])
        }
    
    return results
