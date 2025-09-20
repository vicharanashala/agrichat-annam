from langchain_community.vectorstores import Chroma
import numpy as np
from numpy.linalg import norm
import re
from typing import List, Dict, Optional, Tuple
from context_manager import ConversationContext, MemoryStrategy
from local_llm_interface import local_llm, local_embeddings, run_local_llm
from datetime import datetime
import pytz
import argparse
import hashlib
from functools import lru_cache
import string

class ContentQualityScorer:
    """
    Advanced content quality scoring system for agricultural responses
    """
    
    def __init__(self):
        self.embedding_function = local_embeddings
        
        self.quantitative_indicators = [
            'kg/ha', 'kg per hectare', 'tons/ha', 'quintals', 'grams', 'ml/l', 'ppm',
            'days', 'weeks', 'months', 'years', 'cm', 'inches', 'feet', 'meters',
            '%', 'percent', 'degree', '°c', 'ph', 'ec', 'temperature'
        ]
        
        self.specific_agri_terms = [
            'variety', 'varieties', 'hybrid', 'cultivar', 'strain', 'breed',
            'fertilizer', 'pesticide', 'fungicide', 'herbicide', 'treatment',
            'application', 'dosage', 'concentration', 'spacing', 'depth'
        ]
        
        self.actionable_phrases = [
            'apply', 'sow', 'plant', 'harvest', 'spray', 'irrigate', 'fertilize',
            'transplant', 'prune', 'weed', 'mulch', 'prepare', 'treat', 'monitor'
        ]
        
        self.agricultural_topics = [
            'seed', 'soil', 'crop', 'plant', 'farming', 'agriculture', 'cultivation',
            'irrigation', 'fertilizer', 'pest', 'disease', 'harvest', 'yield'
        ]
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        return np.dot(vec1_np, vec2_np) / (np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np))
    
    def calculate_specificity_score(self, response: str) -> float:
        """
        Calculate how specific and actionable the response is
        Score: 0.0 (generic) to 1.0 (highly specific)
        """
        response_lower = response.lower()
        
        quantitative_count = sum(1 for indicator in self.quantitative_indicators 
                                if indicator in response_lower)
        
        specific_terms_count = sum(1 for term in self.specific_agri_terms 
                                  if term in response_lower)
        
        actionable_count = sum(1 for phrase in self.actionable_phrases 
                              if phrase in response_lower)
        
        response_length = len(response.split())
        if response_length == 0:
            return 0.0
        
        quantitative_density = min(quantitative_count / max(response_length / 50, 1), 1.0)
        specific_density = min(specific_terms_count / max(response_length / 30, 1), 1.0)
        actionable_density = min(actionable_count / max(response_length / 40, 1), 1.0)
        
        specificity_score = (quantitative_density * 0.4 + 
                           specific_density * 0.4 + 
                           actionable_density * 0.2)
        
        return min(specificity_score, 1.0)
    
    def extract_agricultural_topics(self, text: str) -> List[str]:
        """Extract agricultural topics from text"""
        text_lower = text.lower()
        found_topics = []
        
        for topic in self.agricultural_topics:
            if topic in text_lower:
                found_topics.append(topic)
        
        crops = [
            'tomato', 'potato', 'cotton', 'wheat', 'rice', 'maize', 'corn',
            'sugarcane', 'onion', 'garlic', 'soybean', 'groundnut', 'chickpea',
            'mustard', 'sunflower', 'cabbage', 'cauliflower', 'brinjal', 'okra'
        ]
        
        for crop in crops:
            if crop in text_lower:
                found_topics.append(f"crop_{crop}")
        
        return found_topics
    
    def calculate_topic_overlap(self, topics1: List[str], topics2: List[str]) -> float:
        """Calculate overlap between two topic lists"""
        if not topics1 or not topics2:
            return 0.0
        
        set1 = set(topics1)
        set2 = set(topics2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_relevance_score(self, question: str, response: str) -> float:
        """
        Calculate how relevant the response is to the question
        Score: 0.0 (irrelevant) to 1.0 (highly relevant)
        """
        try:
            q_embedding = self.embedding_function.embed_query(question)
            r_embedding = self.embedding_function.embed_query(response)
            semantic_similarity = self.cosine_similarity(q_embedding, r_embedding)
            
            question_topics = self.extract_agricultural_topics(question)
            response_topics = self.extract_agricultural_topics(response)
            topic_overlap = self.calculate_topic_overlap(question_topics, response_topics)
            
            q_crops = [t for t in question_topics if t.startswith('crop_')]
            r_crops = [t for t in response_topics if t.startswith('crop_')]
            
            crop_penalty = 0.0
            if q_crops and r_crops and not set(q_crops).intersection(set(r_crops)):
                crop_penalty = 0.5
            
            relevance_score = (semantic_similarity * 0.6 + topic_overlap * 0.4) - crop_penalty
            
            return max(relevance_score, 0.0)
            
        except Exception as e:
            print(f"[QUALITY SCORER] Error calculating relevance: {e}")
            return 0.0
    
    def detect_question_type(self, question: str) -> str:
        """Detect the type of question being asked"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what', 'which', 'name']):
            return 'what'
        elif any(word in question_lower for word in ['how', 'method', 'process', 'way']):
            return 'how'
        elif any(word in question_lower for word in ['when', 'time', 'timing', 'period']):
            return 'when'
        elif any(word in question_lower for word in ['where', 'location', 'place', 'region']):
            return 'where'
        elif any(word in question_lower for word in ['why', 'reason', 'cause']):
            return 'why'
        else:
            return 'general'
    
    def analyze_response_content_type(self, response: str) -> List[str]:
        """Analyze what type of content the response provides"""
        response_lower = response.lower()
        content_types = []
        
        if any(word in response_lower for word in ['days', 'weeks', 'months', 'season', 'time']):
            content_types.append('temporal')
        
        if any(word in response_lower for word in ['step', 'first', 'then', 'next', 'finally', 'process']):
            content_types.append('procedural')
        
        if any(word in response_lower for word in ['variety', 'type', 'kind', 'characteristics']):
            content_types.append('descriptive')
        
        if any(indicator in response_lower for indicator in self.quantitative_indicators):
            content_types.append('quantitative')
        
        if any(word in response_lower for word in ['region', 'area', 'zone', 'climate', 'soil']):
            content_types.append('locational')
        
        return content_types
    
    def calculate_completeness_score(self, question: str, response: str) -> float:
        """
        Calculate how completely the response answers the question
        Score: 0.0 (incomplete) to 1.0 (complete)
        """
        question_type = self.detect_question_type(question)
        response_content_types = self.analyze_response_content_type(response)
        
        expected_content = {
            'what': ['descriptive', 'quantitative'],
            'how': ['procedural', 'quantitative'],
            'when': ['temporal'],
            'where': ['locational'],
            'why': ['descriptive'],
            'general': ['descriptive', 'quantitative', 'procedural']
        }
        
        expected = expected_content.get(question_type, ['descriptive'])
        
        matches = sum(1 for content_type in expected if content_type in response_content_types)
        
        completeness_score = matches / len(expected)
    
        if len(response_content_types) > len(expected):
            completeness_score = min(completeness_score + 0.1, 1.0)
        
        return completeness_score
    
    def calculate_overall_score(self, specificity: float, relevance: float, completeness: float) -> float:
        """Calculate weighted overall quality score"""
        return (specificity * 0.3 + relevance * 0.5 + completeness * 0.2)
    
    def evaluate_response(self, question: str, response: str, context: str = None) -> Dict[str, float]:
        """
        Main method to evaluate response quality
        Returns comprehensive quality scores
        """
        specificity_score = self.calculate_specificity_score(response)
        relevance_score = self.calculate_relevance_score(question, response)
        completeness_score = self.calculate_completeness_score(question, response)
        overall_score = self.calculate_overall_score(specificity_score, relevance_score, completeness_score)
        
        should_fallback = self.should_trigger_fallback(specificity_score, relevance_score, completeness_score, overall_score)
        
        return {
            'specificity_score': specificity_score,
            'relevance_score': relevance_score,
            'completeness_score': completeness_score,
            'overall_score': overall_score,
            'should_fallback': should_fallback
        }
    
    def should_trigger_fallback(self, specificity: float, relevance: float, completeness: float, overall: float) -> bool:
        """
        Determine if response quality is too low and should trigger LLM fallback
        Conservative thresholds to ensure quality
        """
        min_specificity = 0.25
        min_relevance = 0.35
        min_completeness = 0.3
        min_overall = 0.3
        
        return (specificity < min_specificity or 
                relevance < min_relevance or 
                completeness < min_completeness or
                overall < min_overall)
    
    def check_pops_answer_relevance(self, question: str, answer: str) -> Dict[str, any]:
        """
        Use a small LLM to check if the PoPs answer is relevant to the user's question
        This provides a more nuanced relevance check specifically for PoPs responses
        
        Args:
            question: The original user question
            answer: The generated answer from PoPs database
            
        Returns:
            Dict with is_relevant (bool), confidence (float), and reasoning (str)
        """
        try:
            # Import the LLM interface
            from local_llm_interface import run_local_llm
            
            relevance_prompt = f"""You are a relevance checker for agricultural answers. Your task is to determine if the provided answer directly addresses the user's question.

USER QUESTION: {question}

GENERATED ANSWER: {answer}

INSTRUCTIONS:
1. Carefully analyze if the answer directly addresses what the user is asking about
2. Check if the answer provides information relevant to the specific topic/crop mentioned in the question  
3. Consider if the answer would be helpful to someone who asked this question
4. If the answer talks about a completely different topic, crop, or agricultural practice than what was asked, it's NOT relevant
5. If the answer provides general information that doesn't specifically address the question, it's NOT relevant

Respond in exactly this format:
RELEVANT: YES/NO
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation of why it is or isn't relevant]

Example responses:
RELEVANT: NO
CONFIDENCE: 0.9
REASONING: User asked about tomato diseases but answer discusses rice cultivation methods

RELEVANT: YES  
CONFIDENCE: 0.8
REASONING: User asked about neem tree spacing and answer provides specific spacing recommendations for neem trees"""

            llm_response = run_local_llm(
                relevance_prompt,
                temperature=0.1,
                max_tokens=150,
                use_fallback=False
            )
            
            response_lines = llm_response.strip().split('\n')
            is_relevant = False
            confidence = 0.0
            reasoning = "Failed to parse response"
            
            for line in response_lines:
                line = line.strip()
                if line.startswith('RELEVANT:'):
                    is_relevant = 'YES' in line.upper()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':')[1].strip())
                    except:
                        confidence = 0.5
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            print(f"[POPS RELEVANCE CHECK] Question: {question[:100]}...")
            print(f"[POPS RELEVANCE CHECK] Answer: {answer[:100]}...")
            print(f"[POPS RELEVANCE CHECK] Relevant: {is_relevant}, Confidence: {confidence:.2f}")
            print(f"[POPS RELEVANCE CHECK] Reasoning: {reasoning}")
            
            return {
                'is_relevant': is_relevant,
                'confidence': confidence,
                'reasoning': reasoning,
                'raw_response': llm_response
            }
            
        except Exception as e:
            print(f"[POPS RELEVANCE CHECK] Error in LLM relevance check: {e}")
            return {
                'is_relevant': True,
                'confidence': 0.5,
                'reasoning': f"Relevance check failed: {e}",
                'raw_response': ""
            }

class ChromaQueryHandler:

    REGION_INSTRUCTION = """
IMPORTANT: If a state or region is mentioned in the query, always give preference to that region. If the mentioned region is outside India, politely inform the user that you are only trained on Indian agriculture data and cannot answer for other regions. If the query is not related to Indian agriculture, politely inform the user that you are only trained on Indian agriculture data and can only answer questions related to Indian agriculture.

LOCATION FLEXIBILITY: Provide agricultural information from the available context regardless of specific location, as agricultural practices are generally applicable across similar climatic conditions in India. Only mention specific regional context if it's directly relevant to the user's query.
"""

    STRUCTURED_PROMPT = """
You are an agricultural assistant. Answer the user's question using ONLY the information provided in the context below. Do not add any external knowledge or information.

IMPORTANT: Always respond in the same language in which the query has been asked.

{region_instruction}
Current month: {current_month}

INSTRUCTIONS:
- Use ONLY the information from the provided context
- INTELLIGENT RELEVANCE: Understand the scope of what the user is asking about
- For variety questions: If user asks for "varieties of [crop]", provide ALL varieties of that crop from the context, not just sub-varieties of a specific variety
- Example: "varieties of rice" should include different rice types (Basmati, non-Basmati, aromatic, etc.), not just varieties within Basmati
- If the user asks about a specific crop (e.g., tomatoes) but context only contains information about completely different crops (e.g., potatoes), respond exactly: "I don't have enough information to answer that."
- Do NOT provide information about completely different crops as substitutes
- However, for variety questions, include ALL relevant varieties of the requested crop from the context
- If context has sufficient and RELEVANT information, provide a clear and helpful answer
- Provide agricultural information from the available context regardless of specific location, as practices are generally applicable across India
- Do NOT add any information from external sources or your own knowledge
- Do NOT mention metadata (dates, districts, states, seasons) unless specifically asked or relevant
- Structure your response clearly with bullet points or short paragraphs
- If context is insufficient or irrelevant, respond exactly: "I don't have enough information to answer that."

CRITICAL - For High Similarity Matches (Golden Database):
- When the context directly answers the user's question (indicating a high similarity match), provide the COMPLETE information from the context
- Do NOT truncate, summarize, or omit ANY part of the database content - include EVERY detail provided
- Include ALL practical information such as varieties, timing, spacing, irrigation details, yield information, management practices, precautions, and any other relevant data
- Ensure EVERY sentence and detail from the database content is included in your response
- Present the complete information in a well-structured, readable format
- NEVER leave out yield data, management practices, or cautionary information that appears in the context

VARIETY QUESTION EXAMPLES:
- Question: "What are the varieties of rice?" → Include ALL rice varieties from context (Basmati, non-Basmati, aromatic, short-grain, etc.)
- Question: "What are the varieties of Basmati rice?" → Include specific Basmati varieties (Pusa Basmati 1121, 1509, etc.)
- Question: "What are the varieties of wheat?" → Include ALL wheat varieties from context (durum, bread wheat, different cultivars, etc.)

### Context from Database:
{context}

### User Question:
{question}

### Answer:
"""

    BASE_PROMPT = """
You are an agricultural assistant. Answer the user's question using ONLY the information provided in the context below. Do not add any external knowledge or information.

IMPORTANT: Always respond in the same language in which the query has been asked.

INSTRUCTIONS:
- Use ONLY the information from the provided context
- If context has sufficient and RELEVANT information, provide a clear and helpful answer
- Provide agricultural information from the available context regardless of specific location, as practices are generally applicable across India
- Do NOT add any information from external sources or your own knowledge
- Structure your response clearly with bullet points or short paragraphs
- If context is insufficient or irrelevant, respond exactly: "I don't have enough information to answer that."
"""

    CLASSIFIER_PROMPT = """
You are a smart classifier assistant. Categorize the user query strictly into one of the following categories:

- AGRICULTURE: if the question is related to farming, crops, fertilizers, pests, soil, irrigation, harvest, agronomy, agricultural production, agricultural statistics, crop yields, farming techniques, agricultural states/regions, agricultural economics, etc. This includes questions about agricultural production statistics, which states grow which crops, agricultural data, etc.
- GREETING: if it is a greeting, salutation, polite conversational opening, or introduction. This includes messages like "hi", "hello", "good morning", "Hello [name]", "Hi there", "How are you", "Nice to meet you", etc.
- NON_AGRI: if the question is not agriculture-related or contains inappropriate, offensive, or irrelevant content.

{region_instruction}
Current month: {current_month}

Important: 
- Questions about agricultural production by state/region are AGRICULTURE
- Questions about which states produce which crops are AGRICULTURE  
- Questions about crop statistics, yields, and agricultural data are AGRICULTURE
- Greetings can include names or additional polite phrases. Focus on the intent of the message.

Respond with only one of these words: AGRICULTURE, GREETING, or NON_AGRI.

### User Query:
{question}
### Category:
"""

    GREETING_RESPONSE_PROMPT = """
You are a friendly agricultural assistant. The user has greeted you with: "{question}"

IMPORTANT: Always respond in the same language in which the query has been asked.

{region_instruction}

Respond appropriately to their greeting in a warm and natural way, then invite them to ask agricultural questions. 

Guidelines:
- If they said "Hi" or "Hello", respond with a similar greeting
- If they said "Good morning/afternoon/evening", acknowledge the time appropriately  
- If they used regional greetings like "Namaste", "Namaskaram", "Vanakkam", respond with the same
- If they mentioned their name, acknowledge it politely
- Always end by inviting them to ask farming or agricultural questions
- Keep the response natural and conversational, not robotic
- Don't use placeholder text or templates

Examples:
- User: "Hi" → Response: "Hi there! How can I help you with your farming questions today?"
- User: "Good morning" → Response: "Good morning! I hope you're having a great day. What farming topic can I assist you with?"
- User: "Namaste" → Response: "Namaste! Welcome. I'm here to help with any agricultural questions you might have."

Now respond to: "{question}"
"""

    NON_AGRI_RESPONSE_PROMPT = """
You are an agriculture assistant responding directly to the user who asked: "{question}"

{region_instruction}
Current month: {current_month}

Give a short response (2-3 sentences maximum) saying you are an agriculture assistant based on Indian agriculture context and can only help with agriculture-related questions. Then suggest 1-2 specific agricultural topics they could ask about instead (like "crop recommendations for {current_month}" or "soil preparation tips").

Keep it concise and friendly.
"""

    POPS_PROMPT = """You are an expert agricultural advisor. Answer the user's question using ONLY the relevant information from the Package of Practices (PoPs) content provided below.

IMPORTANT: Always respond in the same language in which the query has been asked.

### Package of Practices Content:
{content}

### User Question: {question}
{context_info}

### CRITICAL INSTRUCTIONS:

**STEP 1 - EXTRACT RELEVANT INFORMATION:**
- Read the PoPs content carefully
- Identify ONLY the information that is relevant to the user's specific question
- Ignore information that is not directly related to the question asked

**STEP 2 - ASSESS CONTENT AVAILABILITY:**
- If the PoPs content has the specific information requested, provide it with some explaination
- If the PoPs content is only partially relevant, provide what is available but do NOT fabricate
- If the PoPs content is relevant but not specific enough, provide what's available and note limitations ONLY if necessary
- Do NOT mention "available PoPs content doesn't include..." unless the content is completely irrelevant
- Also do not include the phrase "available PoPs content" in your response or PoPs does not contain this information.

**STEP 3 - RESPONSE FORMAT:**
- Provide a direct answer to the exact question asked without any title prefixes or headers
- Do NOT start with phrases like "Direct Answer:", "Answer:", "Response:", or any other titles
- Begin immediately with the actual answer content
- Use bullet points for multiple related points when appropriate
- Keep each point concise (1-2 sentences maximum)
- Only include what is directly available in the PoPs content
- Do NOT add disclaimers about content availability unless absolutely necessary
- Add all the information from the PoPs content that is relevant to the question asked.

**STEP 4 - CONTEXT USAGE:**
- Only mention location/region if the question specifically asks about location-specific advice OR if the crop is unsuitable for a region
- Only mention timing/season if the question asks about timing OR if current timing is relevant to the specific advice
- Do NOT automatically include location or timing context unless directly relevant to the question

**STEP 5 - QUALITY CONTROL:**
- Do NOT add information not present in the PoPs content
- Do NOT provide comprehensive guides unless asked for "complete" or "detailed" information
- Do NOT include unrelated aspects (e.g., if asked about sowing, don't include harvesting details)
- Do NOT add unnecessary disclaimers about content availability

**EXAMPLES OF GOOD RESPONSES:**
- Question: "What is the seed rate for wheat?"
  Response: "The seed rate for wheat is 100-125 kg/ha for timely sown wheat."
  
- Question: "When to sow maize?"
  Response: "• Kharif maize: June-July\n• Rabi maize: October-November"
  
- Question: "Can I grow coconut in Punjab?"
  Response: "Coconut cultivation is not suitable for Punjab's climate. Coconut requires tropical coastal conditions with high humidity and temperatures above 20°C year-round, which Punjab's continental climate cannot provide."

**IMPORTANT**: Do NOT use title prefixes like "Direct Answer:", "Answer:", "Response:", etc. Start directly with the content.

### Response:"""

    def __init__(self, chroma_path: str, gemini_api_key: str = None, embedding_model: str = None, chat_model: str = None):
        self.embedding_function = local_embeddings
        
        self.local_llm = local_llm
        
        self.quality_scorer = ContentQualityScorer()
        
        self.db = Chroma(
            persist_directory=chroma_path,
            embedding_function=self.embedding_function,
        )
        
        self.pops_available = False
        self.pops_db = None
        
        try:
            self.pops_db = Chroma(
                collection_name="package_of_practices",
                persist_directory=chroma_path,
                embedding_function=self.embedding_function,
            )
            test_results = self.pops_db.similarity_search("test", k=1)
            if test_results:
                self.pops_available = True
                print(f"[ChromaQueryHandler] PoPs database available as fallback")
            else:
                print(f"[ChromaQueryHandler] PoPs collection exists but is empty")
        except Exception as e:
            print(f"[ChromaQueryHandler] PoPs database not available: {e}")
            self.pops_db = None
            self.pops_available = False
        
        self.context_manager = ConversationContext(
            max_context_pairs=5,
            max_context_tokens=800,
            hybrid_buffer_pairs=3,
            summary_threshold=8,
            memory_strategy=MemoryStrategy.AUTO
        )
        
        self.min_cosine_threshold = 0.5
        self.good_match_threshold = 0.7
        
        self.query_cache = {}
        self.cache_max_size = 100
        
        col = self.db._collection.get()["metadatas"]
        self.meta_index = {
            field: {m[field] for m in col if field in m and m[field]}
            for field in [
                "Year", "Month", "Day",
                "Crop", "District", "Season", "Sector", "State"
            ]
        }

    def _get_query_cache_key(self, query: str, user_state: str = None, filter_dict: dict = None) -> str:
        """Generate cache key for query results"""
        cache_data = f"{query}|{user_state or ''}|{str(filter_dict or {})}"
        return hashlib.md5(cache_data.encode()).hexdigest()

    def extract_location_from_query(self, query: str) -> Optional[str]:
        """Extract location mentioned in the query that should take priority over user location"""
        indian_states = {
            'andhra pradesh': 'Andhra Pradesh', 'arunachal pradesh': 'Arunachal Pradesh', 
            'assam': 'Assam', 'bihar': 'Bihar', 'chhattisgarh': 'Chhattisgarh',
            'goa': 'Goa', 'gujarat': 'Gujarat', 'haryana': 'Haryana',
            'himachal pradesh': 'Himachal Pradesh', 'jharkhand': 'Jharkhand',
            'karnataka': 'Karnataka', 'kerala': 'Kerala', 'madhya pradesh': 'Madhya Pradesh',
            'maharashtra': 'Maharashtra', 'manipur': 'Manipur', 'meghalaya': 'Meghalaya',
            'mizoram': 'Mizoram', 'nagaland': 'Nagaland', 'odisha': 'Odisha',
            'punjab': 'Punjab', 'rajasthan': 'Rajasthan', 'sikkim': 'Sikkim',
            'tamil nadu': 'Tamil Nadu', 'telangana': 'Telangana', 'tripura': 'Tripura',
            'uttar pradesh': 'Uttar Pradesh', 'uttarakhand': 'Uttarakhand',
            'west bengal': 'West Bengal', 'delhi': 'Delhi', 'chandigarh': 'Chandigarh',
            'andaman and nicobar islands': 'Andaman and Nicobar Islands',
            'dadra and nagar haveli and daman and diu': 'Dadra and Nagar Haveli and Daman and Diu',
            'jammu and kashmir': 'Jammu and Kashmir', 'ladakh': 'Ladakh',
            'lakshadweep': 'Lakshadweep', 'puducherry': 'Puducherry',
            'ap': 'Andhra Pradesh', 'tn': 'Tamil Nadu', 'up': 'Uttar Pradesh',
            'mp': 'Madhya Pradesh', 'hp': 'Himachal Pradesh', 'wb': 'West Bengal',
            'orissa': 'Odisha'
        }
        
        query_lower = query.lower()
        location_patterns = [
            r'\bin\s+([a-zA-Z\s]+?)(?:\s|$)',
            r'\bfrom\s+([a-zA-Z\s]+?)(?:\s|$)',
            r'\bat\s+([a-zA-Z\s]+?)(?:\s|$)',
            r'\bof\s+([a-zA-Z\s]+?)(?:\s|$)',
            r'\b([a-zA-Z\s]+?)\s+state(?:\s|$)',
            r'\b([a-zA-Z\s]+?)\s+district(?:\s|$)',
        ]
        
        potential_locations = []
        for pattern in location_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                location = match.strip()
                if len(location) > 2 and location not in ['the', 'and', 'for', 'with', 'can', 'you', 'are', 'how', 'what', 'when', 'where']:
                    potential_locations.append(location)
        
        for location_key, standard_name in indian_states.items():
            if location_key in query_lower:
                if re.search(r'\b' + re.escape(location_key) + r'\b', query_lower):
                    potential_locations.append(standard_name)
        
        for location in potential_locations:
            normalized_location = location.lower().strip()
            if normalized_location in indian_states:
                return indian_states[normalized_location]
        
        return None

    def determine_effective_location(self, query: str, user_state: str = None) -> str:
        """Determine the effective location to use for the query"""
        query_location = self.extract_location_from_query(query)
        if query_location:
            return query_location
        return user_state or "India"
    
    def _cache_query_result(self, cache_key: str, results: list):
        """Cache query results with size limit"""
        if len(self.query_cache) >= self.cache_max_size:
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        self.query_cache[cache_key] = results
    
    def _get_cached_result(self, cache_key: str) -> Optional[list]:
        """Get cached query result if available"""
        return self.query_cache.get(cache_key)

    def get_region_context(self, question: str = None, region: str = None):
        """
        Get region-specific context for greetings and responses
        """
        regional_data = {
            "Andhra Pradesh": {
                "greeting": "You can use Telugu phrases like 'Namaskaram' if appropriate.",
                "context": "Andhra Pradesh is known for rice, cotton, sugarcane, and groundnut cultivation. The state has diverse agro-climatic zones with both kharif and rabi seasons being important.",
                "crops": ["rice", "cotton", "sugarcane", "groundnut", "chili", "turmeric"]
            },
            "Arunachal Pradesh": {
                "greeting": "You can use Hindi or local tribal greetings.",
                "context": "Arunachal Pradesh is known for rice, maize, millet, and horticultural crops. The state has hilly terrain and diverse tribal cultures.",
                "crops": ["rice", "maize", "millet", "orange", "apple", "ginger"]
            },
            "Assam": {
                "greeting": "You can use Assamese phrases like 'Namaskar' if appropriate.",
                "context": "Assam is famous for tea, rice, jute, and mustard. The state has fertile plains and abundant rainfall.",
                "crops": ["tea", "rice", "jute", "mustard", "sugarcane", "potato"]
            },
            "Bihar": {
                "greeting": "You can use Hindi phrases like 'Namaste' if appropriate.",
                "context": "Bihar is a major producer of rice, wheat, maize, and pulses. The state has fertile alluvial soil and is part of the Indo-Gangetic plain.",
                "crops": ["rice", "wheat", "maize", "pulses", "sugarcane", "potato"]
            },
            "Chhattisgarh": {
                "greeting": "You can use Hindi or Chhattisgarhi greetings.",
                "context": "Chhattisgarh is known as the 'Rice Bowl of India' and is a major producer of rice, maize, and pulses.",
                "crops": ["rice", "maize", "pulses", "groundnut", "soybean"]
            },
            "Goa": {
                "greeting": "You can use Konkani or Marathi greetings.",
                "context": "Goa is known for rice, coconut, cashew, and horticultural crops. The state has a coastal climate.",
                "crops": ["rice", "coconut", "cashew", "arecanut", "mango"]
            },
            "Gujarat": {
                "greeting": "You can use Gujarati phrases like 'Kem Cho' if appropriate.",
                "context": "Gujarat is a major producer of cotton, groundnut, tobacco, and various fruits. The state has both irrigated and rain-fed agriculture.",
                "crops": ["cotton", "groundnut", "tobacco", "wheat", "millet", "cumin"]
            },
            "Haryana": {
                "greeting": "You can use Hindi phrases like 'Namaste' if appropriate.",
                "context": "Haryana is known as the 'Granary of India' with extensive wheat and rice cultivation. The state has advanced agricultural practices and infrastructure.",
                "crops": ["wheat", "rice", "sugarcane", "cotton", "mustard", "barley"]
            },
            "Himachal Pradesh": {
                "greeting": "You can use Hindi or Pahari greetings.",
                "context": "Himachal Pradesh is famous for apples, plums, and other temperate fruits. The state also grows wheat, maize, and barley.",
                "crops": ["apple", "plum", "wheat", "maize", "barley", "potato"]
            },
            "Jharkhand": {
                "greeting": "You can use Hindi or local tribal greetings.",
                "context": "Jharkhand is known for rice, maize, pulses, and oilseeds. The state has a mix of plateau and forested areas.",
                "crops": ["rice", "maize", "pulses", "oilseeds", "wheat"]
            },
            "Karnataka": {
                "greeting": "You can use Kannada phrases like 'Namaskara' if appropriate.",
                "context": "Karnataka is a major producer of coffee, sugarcane, cotton, and various horticultural crops.",
                "crops": ["coffee", "sugarcane", "cotton", "millet", "ragi", "turmeric"]
            },
            "Kerala": {
                "greeting": "You can use Malayalam phrases like 'Namaskaram' if appropriate.",
                "context": "Kerala is renowned for spices, coconut, rubber, and rice cultivation. The state has unique farming practices suited to its tropical climate.",
                "crops": ["coconut", "rubber", "rice", "pepper", "cardamom", "tea"]
            },
            "Madhya Pradesh": {
                "greeting": "You can use Hindi phrases like 'Namaste' if appropriate.",
                "context": "Madhya Pradesh is known as the 'Soybean State' and is a major producer of wheat, rice, and pulses.",
                "crops": ["soybean", "wheat", "rice", "pulses", "maize", "gram"]
            },
            "Maharashtra": {
                "greeting": "You can use Marathi phrases like 'Namaskar' if appropriate.",
                "context": "Maharashtra is a major producer of sugarcane, cotton, soybeans, and various fruits. The state has both rain-fed and irrigated agriculture.",
                "crops": ["sugarcane", "cotton", "soybeans", "onion", "grapes", "pomegranate"]
            },
            "Manipur": {
                "greeting": "You can use Manipuri or Hindi greetings.",
                "context": "Manipur is known for rice, maize, pulses, and horticultural crops. The state has hilly terrain and diverse flora.",
                "crops": ["rice", "maize", "pulses", "pineapple", "orange"]
            },
            "Meghalaya": {
                "greeting": "You can use Khasi or English greetings.",
                "context": "Meghalaya is famous for oranges, pineapples, and other horticultural crops. The state also grows rice and maize.",
                "crops": ["orange", "pineapple", "rice", "maize", "potato"]
            },
            "Mizoram": {
                "greeting": "You can use Mizo or English greetings.",
                "context": "Mizoram is known for rice, maize, and horticultural crops. The state has hilly terrain and shifting cultivation.",
                "crops": ["rice", "maize", "banana", "ginger", "orange"]
            },
            "Nagaland": {
                "greeting": "You can use Naga or English greetings.",
                "context": "Nagaland is known for rice, maize, millet, and horticultural crops. The state has hilly terrain and diverse tribal cultures.",
                "crops": ["rice", "maize", "millet", "orange", "pineapple"]
            },
            "Odisha": {
                "greeting": "You can use Odia phrases like 'Namaskar' if appropriate.",
                "context": "Odisha is a major producer of rice, pulses, oilseeds, and coconut. The state has a long coastline and fertile plains.",
                "crops": ["rice", "pulses", "oilseeds", "coconut", "jute", "sugarcane"]
            },
            "Punjab": {
                "greeting": "You can use Punjabi phrases like 'Sat Sri Akal' or Hindi 'Namaste' if appropriate.",
                "context": "Punjab is known as the 'Granary of India' and 'Land of Five Rivers', famous for wheat and rice cultivation. The state has excellent irrigation infrastructure and is a major contributor to India's food grain production.",
                "crops": ["wheat", "rice", "cotton", "maize", "sugarcane", "potato"]
            },
            "Rajasthan": {
                "greeting": "You can use Hindi or Rajasthani greetings.",
                "context": "Rajasthan is known for millet, wheat, barley, and pulses. The state has arid and semi-arid regions.",
                "crops": ["millet", "wheat", "barley", "pulses", "mustard", "cotton"]
            },
            "Sikkim": {
                "greeting": "You can use Nepali or English greetings.",
                "context": "Sikkim is famous for organic farming, cardamom, and horticultural crops. The state is India's first fully organic state.",
                "crops": ["cardamom", "orange", "ginger", "maize", "potato"]
            },
            "Tamil Nadu": {
                "greeting": "You can use Tamil phrases like 'Vanakkam' if appropriate.",
                "context": "Tamil Nadu excels in rice, sugarcane, cotton, and coconut cultivation. The state has strong irrigation systems and diverse cropping patterns.",
                "crops": ["rice", "sugarcane", "cotton", "coconut", "banana", "turmeric"]
            },
            "Telangana": {
                "greeting": "You can use Telugu phrases like 'Namaskaram' if appropriate.",
                "context": "Telangana is famous for rice, cotton, maize, and various millets. The state promotes sustainable farming with initiatives like Rythu Bandhu.",
                "crops": ["rice", "cotton", "maize", "millets", "turmeric", "red gram"]
            },
            "Tripura": {
                "greeting": "You can use Bengali or Kokborok greetings.",
                "context": "Tripura is known for rice, pineapple, and other horticultural crops. The state has hilly terrain and a humid climate.",
                "crops": ["rice", "pineapple", "potato", "mustard", "jute"]
            },
            "Uttar Pradesh": {
                "greeting": "You can use Hindi phrases like 'Namaste' if appropriate.",
                "context": "Uttar Pradesh is India's largest agricultural state producing wheat, rice, sugarcane, and various other crops across diverse agro-climatic zones.",
                "crops": ["wheat", "rice", "sugarcane", "potato", "peas", "mustard"]
            },
            "Uttarakhand": {
                "greeting": "You can use Hindi or Garhwali/Kumaoni greetings.",
                "context": "Uttarakhand is known for rice, wheat, and horticultural crops. The state has hilly terrain and diverse agro-climatic zones.",
                "crops": ["rice", "wheat", "mandua", "potato", "apple"]
            },
            "West Bengal": {
                "greeting": "You can use Bengali phrases like 'Nomoskar' if appropriate.",
                "context": "West Bengal is a major producer of rice, jute, and tea. The state has fertile plains and a humid climate.",
                "crops": ["rice", "jute", "tea", "potato", "mustard", "sugarcane"]
            }
        }
        
        detected_region = None
        mentioned_region = None
        question_lower = question.lower() if question else ""
        for state_name in regional_data.keys():
            if state_name.lower() in question_lower:
                mentioned_region = state_name
                break
        if not mentioned_region and region:
            for state_name in regional_data.keys():
                if state_name.lower() == region.lower():
                    mentioned_region = state_name
                    break
        non_indian_keywords = [
            "usa", "america", "united states", "canada", "china", "pakistan", "bangladesh", "nepal", "sri lanka", "africa", "europe", "germany", "france", "uk", "england", "australia", "brazil", "mexico", "russia", "japan", "korea", "thailand", "vietnam", "indonesia", "malaysia", "philippines", "spain", "italy", "egypt", "iran", "iraq", "afghanistan", "turkey", "saudi", "uae", "oman", "qatar", "kuwait", "argentina", "chile", "peru", "colombia", "venezuela", "south africa", "nigeria", "kenya", "tanzania", "uganda", "zimbabwe", "sudan", "morocco", "algeria", "tunisia", "libya", "israel", "palestine", "jordan", "lebanon", "syria", "yemen", "sweden", "norway", "finland", "denmark", "poland", "ukraine", "belarus", "czech", "slovakia", "hungary", "romania", "bulgaria", "greece", "portugal", "switzerland", "austria", "netherlands", "belgium", "luxembourg", "ireland", "iceland", "new zealand", "fiji", "samoa", "tonga", "papua", "guinea", "maldives", "mauritius", "seychelles", "singapore", "hong kong", "taiwan", "mongolia", "kazakhstan", "uzbekistan", "turkmenistan", "kyrgyzstan", "tajikistan", "georgia", "armenia", "azerbaijan", "estonia", "latvia", "lithuania", "croatia", "serbia", "bosnia", "montenegro", "macedonia", "albania", "slovenia", "bulgaria", "cyprus", "malta", "monaco", "liechtenstein", "andorra", "san marino", "vatican", "luxembourg", "moldova", "belgium", "netherlands", "switzerland", "austria", "gibraltar", "jersey", "guernsey", "isle of man", "faroe", "greenland", "bermuda", "bahamas", "cuba", "jamaica", "haiti", "dominican", "puerto rico", "trinidad", "barbados", "saint lucia", "grenada", "antigua", "dominica", "saint kitts", "saint vincent", "aruba", "curacao", "bonaire", "sint maarten", "saba", "statia", "anguilla", "cayman", "turks", "caicos", "british virgin", "us virgin", "saint barthelemy", "saint martin", "saint pierre", "miquelon", "french guiana", "suriname", "guyana", "paraguay", "uruguay", "bolivia", "ecuador", "peru", "chile", "venezuela", "panama", "costa rica", "nicaragua", "honduras", "el salvador", "guatemala", "belize"]
        for keyword in non_indian_keywords:
            if keyword in question_lower:
                return {
                    "region": "Non-Indian",
                    "context": "Sorry, I am only trained on Indian agriculture data and can only answer questions related to Indian agriculture. Please ask about Indian states, crops, or farming practices.",
                    "greeting": "",
                    "crops": []
                }
        if mentioned_region:
            detected_region = mentioned_region
        elif region:
            detected_region = region
        else:
            detected_region = None
        normalized_states = {
            "andhra pradesh": "Andhra Pradesh",
            "ap": "Andhra Pradesh",
            "arunachal pradesh": "Arunachal Pradesh",
            "arunachal": "Arunachal Pradesh",
            "assam": "Assam",
            "bihar": "Bihar",
            "chhattisgarh": "Chhattisgarh",
            "chattisgarh": "Chhattisgarh",
            "goa": "Goa",
            "gujarat": "Gujarat",
            "haryana": "Haryana",
            "himachal pradesh": "Himachal Pradesh",
            "himachal": "Himachal Pradesh",
            "jharkhand": "Jharkhand",
            "karnataka": "Karnataka",
            "kerala": "Kerala",
            "madhya pradesh": "Madhya Pradesh",
            "mp": "Madhya Pradesh",
            "maharashtra": "Maharashtra",
            "maharastra": "Maharashtra",
            "manipur": "Manipur",
            "meghalaya": "Meghalaya",
            "mizoram": "Mizoram",
            "nagaland": "Nagaland",
            "odisha": "Odisha",
            "orissa": "Odisha",
            "punjab": "Punjab",
            "rajasthan": "Rajasthan",
            "sikkim": "Sikkim",
            "tamil nadu": "Tamil Nadu",
            "tn": "Tamil Nadu",
            "telangana": "Telangana",
            "tripura": "Tripura",
            "uttar pradesh": "Uttar Pradesh",
            "up": "Uttar Pradesh",
            "uttarakhand": "Uttarakhand",
            "uttaranchal": "Uttarakhand",
            "west bengal": "West Bengal",
            "wb": "West Bengal"
        }
        if detected_region and detected_region.lower() in normalized_states:
            detected_region = normalized_states[detected_region.lower()]
        if detected_region and detected_region in regional_data:
            return {
                "region": detected_region,
                "context": regional_data[detected_region]["context"],
                "greeting": regional_data[detected_region]["greeting"],
                "crops": regional_data[detected_region]["crops"]
            }
        else:
            return {
                "region": detected_region or "India",
                "context": "India has diverse agro-climatic zones with rich farming traditions across different states and regions.",
                "greeting": "You can use appropriate regional greetings like 'Namaste', 'Namaskaram', or 'Vanakkam'.",
                "crops": ["rice", "wheat", "cotton", "sugarcane", "pulses", "vegetables"]
            }

    def expand_query(self, query):
        """
        OPTIMIZED: Reduced query expansion for better performance
        Creates only 1-2 strategic query variants instead of 5+
        """
        variants = [query.strip()]
        
        if len(query.split()) > 4:
            simplified = self._simplify_query(query)
            if simplified != query and len(simplified) > 3:
                variants.append(simplified)
        
        return variants[:2]
    
    def _simplify_query(self, query):
        """Create simplified version of query focusing on core terms"""
        question_words = {'what', 'how', 'when', 'where', 'why', 'which', 'can', 'is', 'are', 'do', 'does', 'did'}
        words = query.lower().split()
        content_words = [word for word in words if word not in question_words and len(word) > 2]
        return " ".join(content_words) if content_words else query
    
    def _rephrase_query(self, query):
        """Create alternative phrasing of the query"""
        if "how to" in query.lower():
            return query.lower().replace("how to", "method for").replace("how can", "way to")
        elif "what is" in query.lower():
            return query.lower().replace("what is", "information about")
        elif query.endswith("?"):
            return query[:-1]
        return query

    def _create_metadata_filter(self, question, user_state=None):
        """
        Create metadata filter from question and user_state, but avoid overly restrictive filters
        that might cause ChromaDB query errors. Use proper ChromaDB filter format.
        """
        q = question.lower()
        filt = {}
        
        safe_fields = ["Crop", "District", "State", "Season"]
        
        if user_state and user_state.strip():
            if "State" in self.meta_index:
                for val in self.meta_index["State"]:
                    if str(val).lower() == user_state.lower().strip():
                        filt["State"] = val
                        break
        
        for field in safe_fields:
            if field in self.meta_index and field != "State":
                for val in self.meta_index[field]:
                    if str(val).lower() in q and str(val).lower() != "other" and str(val).lower() != "-":
                        filt[field] = val
                        break
        
        if len(filt) == 0:
            return None
        elif len(filt) == 1:
            return filt
        else:
            conditions = []
            for key, value in filt.items():
                conditions.append({key: value})
            return {"$and": conditions}

    def cosine_sim(self, a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    def extract_crop_from_query(self, query: str) -> Optional[str]:
        """
        Extract specific crop mentioned in the query
        
        Args:
            query: User's question
            
        Returns:
            Crop name if found, None otherwise
        """
        query_lower = query.lower()
        
        crops = [
            'apple', 'mango', 'orange', 'banana', 'grape', 'pomegranate', 'guava', 'papaya',
            'strawberry', 'cherry', 'peach', 'plum', 'apricot', 'kiwi', 'lemon', 'lime',
            
            'potato', 'tomato', 'onion', 'garlic', 'carrot', 'cabbage', 'cauliflower', 
            'brinjal', 'eggplant', 'okra', 'radish', 'turnip', 'beetroot', 'capsicum',
            'pepper', 'chili', 'cucumber', 'bottle gourd', 'bitter gourd', 'pumpkin',
            'spinach', 'lettuce', 'beans', 'peas',
            
            'rice', 'wheat', 'maize', 'corn', 'barley', 'oats', 'millet', 'bajra', 
            'jowar', 'sorghum', 'ragi', 'finger millet',
            
            'arhar', 'pigeon pea', 'gram', 'chickpea', 'lentil', 'masoor', 'moong', 
            'urad', 'black gram', 'cowpea', 'field pea',
            
            'cotton', 'sugarcane', 'tobacco', 'jute', 'sunflower', 'mustard', 'rapeseed',
            'groundnut', 'peanut', 'sesame', 'soybean', 'safflower',
            
            'ginger', 'turmeric', 'coriander', 'cumin', 'fenugreek', 'fennel'
        ]
        
        crops.sort(key=len, reverse=True)
        
        for crop in crops:
            if crop in query_lower:
                return crop
                
        return None

    def validate_crop_specificity(self, query: str, documents: List) -> List:
        """
        Validate that retrieved documents match the specific crop mentioned in query
        
        Args:
            query: User's question  
            documents: List of retrieved documents
            
        Returns:
            Filtered list of documents that match the crop specificity
        """
        query_crop = self.extract_crop_from_query(query)
        
        if not query_crop:
            return documents
            
        
        crop_matched_docs = []
        
        for doc in documents:
            doc_crop = doc.metadata.get('Crop', '').lower()
            doc_content = doc.page_content.lower()
            
            crop_match = False
            
            if query_crop in doc_crop:
                crop_match = True
                
            elif query_crop in doc_content:
                crop_match = True
                
            crop_aliases = {
                'corn': 'maize',
                'eggplant': 'brinjal', 
                'peanut': 'groundnut',
                'chickpea': 'gram',
                'pigeon pea': 'arhar',
                'finger millet': 'ragi'
            }
            
            for alias, standard in crop_aliases.items():
                if (query_crop == alias and standard in doc_crop) or \
                   (query_crop == standard and alias in doc_crop):
                    crop_match = True
                    break
            
            if crop_match:
                crop_matched_docs.append(doc)
            else:
                if doc_crop in ['', 'various', 'multiple', 'others', '-']:
                    crop_matched_docs.append(doc)
                else:
                    if query_crop in doc_content:
                        crop_matched_docs.append(doc)
        
        return crop_matched_docs

    def evaluate_database_match_quality(self, question: str, docs: List, scores: List[float]) -> str:
        """
        Evaluate the quality of database matches and determine response strategy
        Returns: 'GOOD_MATCH', 'WEAK_MATCH', or 'NO_MATCH'
        """
        if not docs or not scores:
            return 'NO_MATCH'
        
        best_score = max(scores) if scores else 0
        
        if best_score >= self.good_match_threshold:
            return 'GOOD_MATCH'
        
        if best_score >= self.min_cosine_threshold:
            best_doc = docs[0]
            content = best_doc.page_content.lower()
            question_lower = question.lower()
            
            question_words = set(question_lower.split())
            content_words = set(content.split())
            overlap_ratio = len(question_words.intersection(content_words)) / len(question_words)
            
            agri_keywords = {'crop', 'plant', 'disease', 'pest', 'fertilizer', 'soil', 'seed', 'farming', 'agriculture', 'harvest'}
            question_agri = question_words.intersection(agri_keywords)
            content_agri = content_words.intersection(agri_keywords)
            
            if overlap_ratio >= 0.2 or (question_agri and content_agri):
                return 'WEAK_MATCH'
        
        return 'NO_MATCH'

    def rerank_documents(self, question: str, results, top_k: int = 5):
        """Enhanced reranking with stricter similarity scoring for RAG-only approach"""
        query_embedding = self.embedding_function.embed_query(question)
        scored = []
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        for doc, original_score in results:
            content = doc.page_content.strip()
            
            if (content.lower().count('others') > 3 or 
                'Question: Others' in content or 
                'Answer: Others' in content or
                content.count('Others') > 5 or
                len(content.strip()) < 50):
                continue
            
            d_emb = self.embedding_function.embed_query(content)
            cosine_score = self.cosine_sim(query_embedding, d_emb)
            
            content_words = set(content.lower().split())
            keyword_overlap = len(question_words.intersection(content_words)) / len(question_words) if question_words else 0
            
            agri_terms = {'crop', 'plant', 'disease', 'pest', 'fertilizer', 'soil', 'water', 'harvest', 'seed', 'growth', 'yield', 'farming', 'agriculture'}
            question_agri_terms = question_words.intersection(agri_terms)
            content_agri_terms = content_words.intersection(agri_terms)
            agri_overlap = len(question_agri_terms.intersection(content_agri_terms)) / max(len(question_agri_terms), 1)
            
            combined_score = (cosine_score * 0.7) + (keyword_overlap * 0.2) + (agri_overlap * 0.1)
            
            scored.append((doc, combined_score, cosine_score, original_score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, combined_score, cosine_score, orig_score in scored[:top_k] 
                if doc.page_content.strip() and combined_score > self.min_cosine_threshold]

    def filter_response_thinking(self, response: str) -> str:
        """
        Filter out LLM internal thinking and assumptions from the response
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Cleaned response without thinking parts
        """
        
        thinking_patterns = [
            r"Since the user is located in [^,]+, I will provide information specific to that region\.\s*",
            r"However, since the question does not mention [^,]+, I will assume [^.]+\.\s*",
            r"Given that the user is in [^,]+, [^.]+\.\s*",
            r"As the user is located in [^,]+, [^.]+\.\s*",
            r"Since you are in [^,]+, [^.]+\.\s*",
            r"Based on your location in [^,]+, [^.]+\.\s*",
            r"As we discussed earlier, [^.]+\.\s*",
            r"Since this is a follow-up question, [^.]+\.\s*",
            r"Given the context of our conversation, [^.]+\.\s*",
            r"As mentioned in the previous response, [^.]+\.\s*",
            r"Following up on our discussion about [^,]+, [^.]+\.\s*",
            r"Since you asked about [^,]+ earlier, [^.]+\.\s*",
            r"I'll assume you want [^.]+\.\s*",
            r"I assume you're asking about [^.]+\.\s*",
            r"Let me assume [^.]+\.\s*",
            r"I will assume [^.]+\.\s*"
        ]
        
        cleaned_response = response
        
        for pattern in thinking_patterns:
            cleaned_response = re.sub(pattern, "", cleaned_response, flags=re.IGNORECASE)
        
        cleaned_response = re.sub(r'\n\s*\n', '\n\n', cleaned_response)
        cleaned_response = re.sub(r'^\s+', '', cleaned_response)
        
        return cleaned_response.strip()

    def construct_structured_prompt(self, context: str, question: str, user_state: str = None) -> str:
        IST = pytz.timezone("Asia/Kolkata")
        current_month = datetime.now(IST).strftime('%B')
        
        return self.STRUCTURED_PROMPT.format(
            context=context,
            question=question,
            region_instruction=self.REGION_INSTRUCTION,
            current_month=current_month
        )

    def construct_pops_prompt(self, content: str, question: str, effective_location: str = None) -> str:
        """
        Construct specialized prompt for PoPs database queries with intelligent context inclusion
        """
        IST = pytz.timezone("Asia/Kolkata")
        current_month = datetime.now(IST).strftime('%B')
        
        filtered_content = self.filter_pops_content(content, question)
        
        question_lower = question.lower()
        context_info = ""
        
        location_relevant = (
            'in ' in question_lower or 'from ' in question_lower or 'at ' in question_lower or
            'region' in question_lower or 'state' in question_lower or 'district' in question_lower or
            'climate' in question_lower or 'suitable' in question_lower or 'grow' in question_lower or
            'cultivation' in question_lower or 'can i' in question_lower or 'should i' in question_lower
        )
        
        timing_relevant = (
            'when' in question_lower or 'time' in question_lower or 'season' in question_lower or
            'month' in question_lower or 'sowing' in question_lower or 'planting' in question_lower or
            'now' in question_lower or 'current' in question_lower
        )
        
        context_parts = []
        if location_relevant and effective_location:
            context_parts.append(f"- Location: {effective_location}")
        if timing_relevant:
            context_parts.append(f"- Current month: {current_month}")
            
        if context_parts:
            context_info = f"\n\n### Context Information:\n{chr(10).join(context_parts)}"
        
        return self.POPS_PROMPT.format(
            content=filtered_content,
            question=question,
            context_info=context_info
        )
    
    def filter_pops_content(self, content: str, question: str) -> str:
        """
        Filter PoPs content to extract only the most relevant parts for the specific question
        """
        question_lower = question.lower()
        
        content_lines = [line.strip() for line in content.split('\n') if line.strip()]
        relevant_lines = []
        
        keyword_mapping = {
            'seed_rate': ['seed rate', 'seeding rate', 'kg/ha', 'quantity of seed', 'seeds per', 'kg per hectare'],
            'sowing': ['sowing', 'planting', 'transplanting', 'seeding time', 'planting time', 'sow in', 'plant in', 'sowing time'],
            'fertilizer': ['fertilizer', 'manure', 'compost', 'nutrition', 'NPK', 'nitrogen', 'phosphorus', 'potassium', 'urea', 'nutrient'],
            'irrigation': ['irrigation', 'watering', 'water management', 'moisture', 'rainfall', 'water requirement', 'water need'],
            'variety': ['variety', 'varieties', 'cultivar', 'hybrid', 'types of', 'recommended varieties', 'cultivars'],
            'pest_disease': ['pest', 'disease', 'insect', 'fungal', 'bacterial', 'viral', 'control', 'management', 'spray', 'treatment'],
            'harvest': ['harvest', 'harvesting', 'maturity', 'ready for harvest', 'cutting', 'picking', 'harvest time'],
            'yield': ['yield', 'production', 'productivity', 'tonnes per hectare', 'quintal', 'output'],
            'spacing': ['spacing', 'plant distance', 'row to row', 'plant to plant', 'distance between', 'plant spacing'],
            'soil': ['soil', 'soil preparation', 'field preparation', 'tillage', 'ploughing', 'soil type'],
            'climate': ['climate', 'temperature', 'rainfall', 'humidity', 'weather', 'suitable conditions'],
            'cultivation': ['cultivation', 'growing', 'farming', 'agriculture', 'crop management']
        }
        
        matching_types = []
        for qtype, keywords in keyword_mapping.items():
            if any(keyword in question_lower for keyword in keywords):
                matching_types.append(qtype)
        
        if matching_types:
            all_relevant_keywords = []
            for qtype in matching_types:
                all_relevant_keywords.extend(keyword_mapping[qtype])
            
            for line in content_lines:
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in all_relevant_keywords):
                    relevant_lines.append(line)
            
            if relevant_lines:
                context_keywords = ['recommended', 'suitable', 'best', 'important']
                context_lines = []
                for line in content_lines:
                    line_lower = line.lower()
                    if (any(keyword in line_lower for keyword in context_keywords) and 
                        len(line) < 150 and line not in relevant_lines):
                        context_lines.append(line)
                
                if context_lines:
                    relevant_lines = context_lines[:2] + relevant_lines
                
                if len(relevant_lines) > 6:
                    relevant_lines = relevant_lines[:6]
                
                filtered_content = '\n'.join(relevant_lines)
                print(f"[POPS CONTENT FILTER] Filtered content for '{', '.join(matching_types)}' from {len(content_lines)} to {len(relevant_lines)} lines")
                return filtered_content
        
        general_question_indicators = ['about', 'information', 'details', 'tell me', 'what is', 'how to']
        is_general_question = any(indicator in question_lower for indicator in general_question_indicators)
        
        if is_general_question:
            summary_lines = []
            for line in content_lines:
                if len(line) > 30 and len(line) < 200:
                    summary_lines.append(line)
                if len(summary_lines) >= 8:
                    break
            
            if summary_lines:
                filtered_content = '\n'.join(summary_lines)
                print(f"[POPS CONTENT FILTER] General question - using {len(summary_lines)} summary lines")
                return filtered_content
        
        if len(content) > 1200:
            truncated = content[:1200]
            last_period = truncated.rfind('.')
            if last_period > 800:
                truncated = truncated[:last_period + 1]
            else:
                truncated += "..."
            
            print(f"[POPS CONTENT FILTER] Intelligently truncated content from {len(content)} to {len(truncated)} characters")
            return truncated
        
        print(f"[POPS CONTENT FILTER] Using original content ({len(content)} characters)")
        return content

    def classify_query(self, question: str, conversation_history: Optional[List[Dict]] = None) -> str:
        IST = pytz.timezone("Asia/Kolkata")
        current_month = datetime.now(IST).strftime('%B')
        prompt = self.CLASSIFIER_PROMPT.format(question=question, region_instruction=self.REGION_INSTRUCTION, current_month=current_month)
        try:
            response_text = run_local_llm(
                prompt,
                temperature=0,
                max_tokens=20,
                use_fallback=False
            )
            category = response_text.strip().upper()
            if category in {"AGRICULTURE", "GREETING", "NON_AGRI"}:
                return category
            else:
                return "NON_AGRI"
        except Exception as e:
            return "NON_AGRI"

    def generate_dynamic_response(self, question: str, mode: str, region: str = None) -> str:
        IST = pytz.timezone("Asia/Kolkata")
        current_month = datetime.now(IST).strftime('%B')
        if mode == "GREETING":
            prompt = self.GREETING_RESPONSE_PROMPT.format(question=question, region_instruction=self.REGION_INSTRUCTION, current_month=current_month)
        else:
            prompt = self.NON_AGRI_RESPONSE_PROMPT.format(question=question, region_instruction=self.REGION_INSTRUCTION, current_month=current_month)
        try:
            response_text = run_local_llm(
                prompt,
                temperature=0.3,
                max_tokens=512,
                use_fallback=False
            )
            return self.filter_response_thinking(response_text.strip())
        except Exception as e:
            return "Sorry, I can only help with agriculture-related questions."

    def search_pops_fallback(self, question: str, effective_location: str = None) -> Dict[str, any]:
        """
        Fallback search in PoPs database with very lenient thresholds
        This is used as the final fallback before going to LLM
        
        Args:
            question: The user's question
            effective_location: User's location for context
            
        Returns:
            Dict with answer, source, cosine_similarity, and document_metadata
            Returns fallback dict if no suitable content found
        """
        if not self.pops_available:
            print("[POPS FALLBACK] PoPs database not available, proceeding to LLM")
            return {
                'answer': "__FALLBACK__",
                'source': "Fallback LLM",
                'cosine_similarity': 0.0,
                'document_metadata': None
            }
        
        try:
            print(f"[POPS FALLBACK] Searching PoPs database for: '{question}'")
            
            pops_results = self.pops_db.similarity_search_with_score(question, k=5, filter=None)
            
            if not pops_results:
                print("[POPS FALLBACK] No results found in PoPs database")
                return {
                    'answer': "__FALLBACK__",
                    'source': "Fallback LLM",
                    'cosine_similarity': 0.0,
                    'document_metadata': None
                }
            
            print(f"[POPS FALLBACK] Found {len(pops_results)} results in PoPs database")
            
            for i, (doc, score) in enumerate(pops_results[:3]):
                print(f"   PoPs Result {i+1}: Distance={score:.3f}")
                print(f"      Content Preview: {doc.page_content[:100]}...")
            
            best_doc, best_score = pops_results[0]
            content = best_doc.page_content.strip()
            
            if len(content) < 50:
                print(f"[POPS FALLBACK] Content too short ({len(content)} chars), proceeding to LLM")
                return {
                    'answer': "__FALLBACK__",
                    'source': "Fallback LLM",
                    'cosine_similarity': 0.0,
                    'document_metadata': None
                }
            
            query_embedding = self.embedding_function.embed_query(question)
            content_embedding = self.embedding_function.embed_query(content)
            cosine_score = self.cosine_sim(query_embedding, content_embedding)
            
            print(f"[POPS FALLBACK] Best result analysis:")
            print(f"   Distance: {best_score:.3f}, Cosine: {cosine_score:.3f}")
            print(f"   Content preview: {content[:150]}...")
            
            if best_score > 300.0:
                print(f"[POPS FALLBACK] Distance too high ({best_score:.3f} > 300.0), proceeding to LLM")
                return {
                    'answer': "__FALLBACK__",
                    'source': "Fallback LLM",
                    'cosine_similarity': cosine_score,
                    'document_metadata': None
                }
            
            if cosine_score < 0.4:
                print(f"[POPS FALLBACK] Cosine similarity too low ({cosine_score:.3f} < 0.4), proceeding to LLM")
                return {
                    'answer': "__FALLBACK__",
                    'source': "Fallback LLM",
                    'cosine_similarity': cosine_score,
                    'document_metadata': None
                }
            
            question_lower = question.lower()
            content_lower = content.lower()
            
            import string
            question_clean = question_lower.translate(str.maketrans('', '', string.punctuation))
            question_keywords = set()
            common_words = {'what', 'how', 'when', 'where', 'why', 'can', 'should', 'with', 'from', 'this', 'that', 'are', 'the', 'for', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'is', 'it'}
            
            for word in question_clean.split():
                if len(word) > 3 and word not in common_words:
                    question_keywords.add(word)
            
            content_keywords = set(content_lower.split())
            keyword_match = bool(question_keywords.intersection(content_keywords))
            
            print(f"[POPS FALLBACK] Keyword analysis:")
            print(f"   Question keywords: {question_keywords}")
            print(f"   Keyword match found: {keyword_match}")
            
            if not keyword_match:
                print(f"[POPS FALLBACK] No keyword match found, proceeding to LLM")
                return {
                    'answer': "__FALLBACK__",
                    'source': "Fallback LLM",
                    'cosine_similarity': cosine_score,
                    'document_metadata': None
                }
            
            print(f"[POPS FALLBACK] ✓ Content passed all quality checks, generating response")
            print(f"[POPS FALLBACK] Final scores - Distance: {best_score:.3f}, Cosine: {cosine_score:.3f}, Keywords: {keyword_match}")
            
            pops_prompt = self.construct_pops_prompt(content, question, effective_location)
            
            try:
                generated_response = run_local_llm(
                    pops_prompt,
                    temperature=0.1,  
                    max_tokens=512,   
                    use_fallback=False
                )
                
                final_response = self.filter_response_thinking(generated_response.strip())
                
                
                is_too_short = len(final_response.strip()) < 40
                is_refusal = any(phrase in final_response.lower() for phrase in [
                    "i don't have enough information",
                    "i cannot answer",
                    "not enough information",
                    "insufficient information",
                    "i don't have information",
                    "unable to provide",
                    "cannot provide information",
                    "the provided content doesn't",
                    "the content doesn't provide",
                    "no information available",
                    "doesn't contain information",
                    "cannot be answered"
                ])
                
                pops_specific_keywords = [
                    'variety', 'varieties', 'cultivar', 'hybrid', 'seed rate', 'spacing', 
                    'sowing', 'planting', 'transplanting', 'fertilizer', 'manure', 'compost',
                    'irrigation', 'watering', 'drainage', 'soil preparation', 'field preparation',
                    'pest management', 'disease control', 'weed control', 'harvesting',
                    'yield', 'productivity', 'cultivation', 'cropping', 'season', 'timing',
                    'plant protection', 'fungicide', 'insecticide', 'organic', 'nutrient'
                ]
                
                general_agri_keywords = [
                    'crop', 'plant', 'seed', 'soil', 'farming', 'agriculture', 'growth'
                ]
                
                has_pops_content = any(keyword in final_response.lower() for keyword in pops_specific_keywords)
                has_general_agri_content = any(keyword in final_response.lower() for keyword in general_agri_keywords)
                
                has_structured_info = any(pattern in final_response for pattern in [
                    '•', '-', '1.', '2.', '3.', 'Step', 'stage', 'phase',
                    'kg/ha', 'days', 'weeks', 'months', 'cm', 'inches'
                ])
                
                quality_scores = self.quality_scorer.evaluate_response(question, final_response)
                
                print(f"[POPS FALLBACK] Content Quality Scoring:")
                print(f"   Specificity Score: {quality_scores['specificity_score']:.3f}")
                print(f"   Relevance Score: {quality_scores['relevance_score']:.3f}")
                print(f"   Completeness Score: {quality_scores['completeness_score']:.3f}")
                print(f"   Overall Score: {quality_scores['overall_score']:.3f}")
                print(f"   Should Fallback: {quality_scores['should_fallback']}")
                
                if quality_scores['should_fallback']:
                    print(f"[POPS FALLBACK] Content quality too low, proceeding to LLM fallback")
                    return {
                        'answer': "__FALLBACK__",
                        'source': "Fallback LLM",
                        'cosine_similarity': cosine_score,
                        'document_metadata': None
                    }
                
                print(f"[POPS FALLBACK] Performing LLM-based relevance check...")
                relevance_check = self.quality_scorer.check_pops_answer_relevance(question, final_response)
                
                if not relevance_check['is_relevant'] and relevance_check['confidence'] > 0.6:
                    print(f"[POPS FALLBACK] LLM relevance check failed - Answer not relevant to question")
                    print(f"[POPS FALLBACK] Confidence: {relevance_check['confidence']:.2f}, Reasoning: {relevance_check['reasoning']}")
                    return {
                        'answer': "__FALLBACK__",
                        'source': "Fallback LLM",
                        'cosine_similarity': cosine_score,
                        'document_metadata': None
                    }
                
                print(f"[POPS FALLBACK] LLM relevance check passed - Answer is relevant")
                print(f"[POPS FALLBACK] Relevance confidence: {relevance_check['confidence']:.2f}")
                
                is_response_relevant = not quality_scores['should_fallback'] and relevance_check['is_relevant']
                
                print(f"[POPS FALLBACK] Response validation:")
                print(f"   Length: {len(final_response.strip())} chars (min: 40)")
                print(f"   Has PoPs-specific content: {has_pops_content}")
                print(f"   Has general agri content: {has_general_agri_content}")
                print(f"   Has structured info: {has_structured_info}")
                print(f"   Is response relevant: {is_response_relevant}")
                print(f"   LLM relevance check: {relevance_check['is_relevant']} (confidence: {relevance_check['confidence']:.2f})")
                print(f"   Is refusal: {is_refusal}")
                print(f"   Response preview: {final_response[:100]}...")
                
                is_valid_pops_response = (
                    not is_too_short and 
                    not is_refusal and 
                    is_response_relevant and
                    (has_pops_content or (has_general_agri_content and has_structured_info))
                )
                
                if is_valid_pops_response:
                    source_text = "\n\n<small><i>Source: Fallback to Package of Practices Database</i></small>"
                    final_response += source_text
                    
                    print(f"[POPS FALLBACK] ✓ Successfully generated valid PoPs response")
                    return {
                        'answer': final_response,
                        'source': "PoPs Database",
                        'cosine_similarity': cosine_score,
                        'document_metadata': best_doc.metadata
                    }
                else:
                    print(f"[POPS FALLBACK] Response validation failed")
                    print(f"   Reasons: too_short={is_too_short}, refusal={is_refusal}, invalid_pops_response={not is_valid_pops_response}")
                    return {
                        'answer': "__FALLBACK__",
                        'source': "Fallback LLM",
                        'cosine_similarity': cosine_score,
                        'document_metadata': None
                    }
                    
            except Exception as e:
                print(f"[POPS FALLBACK] LLM generation failed: {e}")
                return {
                    'answer': "__FALLBACK__",
                    'source': "Fallback LLM",
                    'cosine_similarity': cosine_score,
                    'document_metadata': None
                }
                
        except Exception as e:
            print(f"[POPS FALLBACK] PoPs search failed: {e}")
            return {
                'answer': "__FALLBACK__",
                'source': "Fallback LLM",
                'cosine_similarity': 0.0,
                'document_metadata': None
            }

    def get_answer(self, question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = None) -> str:
        """
        RAG-only approach - uses only database content, no fallback
        
        Strategy:
        1. Check if greeting - handle with __NO_SOURCE__
        2. Search database for relevant content
        3. If good match found, generate response from database content
        4. If no good match, return __FALLBACK__ for tools.py to handle
        
        Args:
            question: Current user question
            conversation_history: List of previous Q&A pairs for context
            user_state: User's state/region for regional context
            
        Returns:
            Generated response from RAG database or __FALLBACK__ indicator
        """
        result = self.get_answer_with_source(question, conversation_history, user_state)
        return result['answer']

    def get_answer_with_source(self, question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = None, db_filter: str = None, database_selection: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Enhanced RAG approach that returns answer with source information and cosine similarity
        
        Flow: Based on database_selection parameter:
        - ["rag"] → Golden Database → RAG Database → "No info"
        - ["pops"] → PoPs Database → "No info"  
        - ["llm"] → LLM only
        - ["rag", "pops"] → RAG → PoPs → "No info"
        - ["rag", "llm"] → RAG → LLM
        - ["pops", "llm"] → PoPs → LLM
        - ["rag", "pops", "llm"] → RAG → PoPs → LLM (current pipeline)
        - None or [] → LLM fallback
        
        Args:
            question: User question
            conversation_history: Previous conversation context
            user_state: User's state/region
            db_filter: Legacy single database filter (for backward compatibility)
            database_selection: List of databases to query in order ["rag", "pops", "llm"]
        
        Returns:
            Dict containing:
            - answer: Generated response
            - source: "Golden Database", "RAG Database", "PoPs Database", or "Fallback LLM"
            - cosine_similarity: Similarity score (0.0-1.0)
            - document_metadata: Metadata of the source document (if applicable)
        """
        if db_filter:
            return self._handle_filtered_database_query(question, conversation_history, user_state, db_filter)
        
        if database_selection is not None:
            return self._handle_database_combination(question, conversation_history, user_state, database_selection)
        
        return self._handle_database_combination(question, conversation_history, user_state, ["rag", "pops", "llm"])

    def _handle_database_combination(self, question: str, conversation_history: Optional[List[Dict]], user_state: str, database_selection: List[str]) -> Dict[str, any]:
        """Handle queries with multiple database combinations"""
        print(f"[DB_COMBINATION] ==========================================")
        print(f"[DB_COMBINATION] Processing question: {question}")
        print(f"[DB_COMBINATION] Selected databases: {database_selection}")
        print(f"[DB_COMBINATION] User state: {user_state}")
        print(f"[DB_COMBINATION] ==========================================")
        
        if not database_selection:
            print("[DB_COMBINATION] ⚠️ No databases selected, using LLM fallback")
            return self._query_llm_only(question, conversation_history, user_state)
        
        for i, db in enumerate(database_selection, 1):
            print(f"[DB_COMBINATION] Step {i}/{len(database_selection)}: Trying database '{db}'")
            
            if db == "rag":
                print(f"[DB_COMBINATION] → Querying RAG database (Golden + RAG content)...")
                result = self._query_rag_database_only(question, conversation_history, user_state)
                print(f"[DEBUG] RAG result preview: {result['answer'][:100]}...")
                
                is_no_info = result['answer'] == "I don't have enough information to answer that from the RAG database."
                is_good_match = result['cosine_similarity'] >= 0.6
                
                is_crop_relevant = True
                if not is_no_info and is_good_match:
                    query_crop = self.extract_crop_from_query(question)
                    if query_crop and 'document_metadata' in result and result['document_metadata']:
                        doc_metadata = result['document_metadata']
                        doc_crop = doc_metadata.get('Crop', '').lower() if doc_metadata else ''
                        
                        doc_content = result['answer'].lower()
                        
                        crop_match = (
                            query_crop.lower() in doc_crop or
                            query_crop.lower() in doc_content or
                            doc_crop in query_crop.lower()
                        )
                        
                        if not crop_match:
                            print(f"[DB_COMBINATION] ⚠️ Cross-crop false positive detected!")
                            print(f"[DB_COMBINATION] Query crop: '{query_crop}' vs Document crop: '{doc_crop}'")
                            print(f"[DB_COMBINATION] High similarity ({result['cosine_similarity']:.3f}) but different crops - rejecting")
                            is_crop_relevant = False
                
                if not is_no_info and is_good_match and is_crop_relevant:
                    print(f"[DB_COMBINATION] ✅ RAG database provided good quality answer (source: {result['source']}, score: {result['cosine_similarity']:.3f})")
                    return result
                else:
                    if is_no_info:
                        print(f"[DB_COMBINATION] ❌ RAG database failed - no information found")
                    else:
                        print(f"[DB_COMBINATION] ⚠️ RAG database provided low quality answer (score: {result['cosine_similarity']:.3f}), continuing to next database")
                    
            elif db == "pops":
                print(f"[DB_COMBINATION] → Querying PoPs database...")
                result = self._query_pops_database_only(question, conversation_history, user_state)
                print(f"[DEBUG] PoPs result preview: {result['answer'][:100]}...")
                if result['answer'] != "I don't have enough information to answer that from the PoPs database.":
                    print(f"[DB_COMBINATION] ✅ PoPs database provided answer (source: {result['source']})")
                    return result
                else:
                    print(f"[DB_COMBINATION] ❌ PoPs database failed")
                    print(f"[DB_COMBINATION] ❌ PoPs database failed")
                    
            elif db == "llm":
                print(f"[DB_COMBINATION] → Using LLM fallback...")
                result = self._query_llm_only(question, conversation_history, user_state)
                print(f"[DEBUG] LLM result preview: {result['answer'][:100]}...")
                print(f"[DB_COMBINATION]  LLM provided answer (source: {result['source']})")
                return result
            
            else:
                print(f"[DB_COMBINATION]  Unknown database: {db}")
        
        
        print("[DB_COMBINATION] All selected databases failed, no LLM fallback")
        return {
            'answer': "I don't have enough information to answer that from the selected databases.",
            'source': "No Database",
            'cosine_similarity': 0.0,
            'document_metadata': {}
        }
    
    def _handle_filtered_database_query(self, question: str, conversation_history: Optional[List[Dict]], user_state: str, db_filter: str) -> Dict[str, any]:
        """Handle queries filtered to specific databases"""
        print(f"[DB_FILTER] Querying only {db_filter.upper()} database")
        
        if db_filter == "rag":
            return self._query_rag_database_only(question, conversation_history, user_state)
        elif db_filter == "pops":
            return self._query_pops_database_only(question, conversation_history, user_state)
        else:
            print(f"[DB_FILTER] Unknown database filter: {db_filter}")
            return {
                'answer': "__FALLBACK__",
                'source': "Unknown Filter",
                'cosine_similarity': 0.0,
                'document_metadata': {}
            }

    def _query_rag_database_only(self, question: str, conversation_history: Optional[List[Dict]], user_state: str) -> Dict[str, any]:
        """Query unified RAG database (includes both Golden and RAG content)"""
        print("[DB_FILTER] Querying unified RAG database (Golden + RAG content)...")
        print(f"[DEBUG] Original question: {question}")
        
        processing_query = question
        if conversation_history:
            enhanced_query_result = self.context_manager.enhance_query_with_cot(question, conversation_history)
            if enhanced_query_result['requires_cot'] or enhanced_query_result['context_used']:
                processing_query = enhanced_query_result['enhanced_query']
                print(f"[DEBUG] Enhanced query: {processing_query}")
        
        print(f"[DEBUG] Final processing query: {processing_query}")
        results = self.db.similarity_search_with_score(processing_query, k=6)
        print(f"[DEBUG] Retrieved {len(results)} documents from RAG database")
        
        # Collect research data for frontend display
        research_data = []
        
        # Get query embedding for proper cosine similarity calculation
        query_embedding = self.embedding_function.embed_query(processing_query)
        
        for i, (doc, distance) in enumerate(results):
            # Get document embedding and calculate proper cosine similarity
            doc_embedding = self.embedding_function.embed_query(doc.page_content)
            cosine_score = self.cosine_sim(query_embedding, doc_embedding)
            collection_type = doc.metadata.get('collection', 'rag')
            print(f"[DEBUG] Doc {i+1}: Distance={distance:.4f}, Score={cosine_score:.3f}, Type={collection_type}, Content preview: {doc.page_content[:100]}...")
            print(f"[DEBUG] Doc {i+1} Metadata: {doc.metadata}")
            
            # Add to research data
            research_data.append({
                'rank': i + 1,
                'distance': round(distance, 4),
                'cosine_similarity': round(cosine_score, 3),
                'collection_type': collection_type,
                'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                'metadata': doc.metadata,
                'selected': False  # Will be updated if this document is selected
            })
        
        selected_doc_index = None
        for doc_idx, (doc, distance) in enumerate(results):
            # Get document embedding and calculate proper cosine similarity
            doc_embedding = self.embedding_function.embed_query(doc.page_content)
            cosine_score = self.cosine_sim(query_embedding, doc_embedding)
            
            # Combined filtering using both distance and cosine similarity
            # Distance should be low (< 0.35) AND cosine similarity should be high (>= 0.7)
            distance_ok = distance < 0.35  # Low distance means high similarity in vector space
            cosine_ok = cosine_score >= 0.5  # High cosine similarity means semantic relevance
            
            print(f"[FILTER_DEBUG] Doc: Distance={distance:.4f} (ok: {distance_ok}), Cosine={cosine_score:.3f} (ok: {cosine_ok})")
            
            if distance_ok and cosine_ok:
                # Add crop-specific validation to prevent cross-crop false positives
                query_crop = self.extract_crop_from_query(processing_query)
                is_crop_relevant = True
                
                if query_crop:
                    doc_crop = doc.metadata.get('Crop', '').lower() if doc.metadata else ''
                    doc_content = doc.page_content.lower()
                    
                    crop_match = (
                        query_crop.lower() in doc_crop or
                        query_crop.lower() in doc_content or
                        doc_crop in query_crop.lower()
                    )
                    
                    if not crop_match:
                        print(f"[CROP_VALIDATION] ⚠️ Cross-crop false positive detected!")
                        print(f"[CROP_VALIDATION] Query crop: '{query_crop}' vs Document crop: '{doc_crop}'")
                        print(f"[CROP_VALIDATION] Similarity: {cosine_score:.3f} but different crops - skipping document")
                        is_crop_relevant = False
                
                if is_crop_relevant:
                    collection_type = doc.metadata.get('collection', 'rag')
                    print(f"[DB_FILTER] ✓ RAG database match found (type: {collection_type}) with score: {cosine_score:.3f}")
                    print(f"[DEBUG] Selected document content: {doc.page_content}")
                    print(f"[DEBUG] Selected document metadata: {doc.metadata}")
                    
                    # Mark this document as selected in research data
                    selected_doc_index = doc_idx
                    research_data[doc_idx]['selected'] = True
                    research_data[doc_idx]['selection_reason'] = "Passed distance & cosine filters + crop validation"
                    
                    if cosine_score >= 0.7:
                        print(f"[DEBUG] High similarity score ({cosine_score:.3f}) - extracting direct answer from database")
                        content_lines = doc.page_content.split('\n')
                        answer_text = ""
                        capturing_answer = False
                        
                        for line in content_lines:
                            if line.startswith('Answer:'):
                                answer_text = line.replace('Answer:', '').strip()
                                capturing_answer = True
                            elif capturing_answer and line.strip():
                                answer_text += " " + line.strip()
                            elif capturing_answer and not line.strip():
                                break
                        
                        if answer_text:
                            print(f"[DEBUG] Extracted direct answer: {answer_text[:100]}...")
                            source_info = f"\n\n**Source:** RAG Database (Golden)"
                            
                            return {
                                'answer': answer_text + source_info,
                                'source': "RAG Database (Golden)",
                                'cosine_similarity': cosine_score,
                                'document_metadata': doc.metadata,
                                'research_data': research_data
                            }
                    
                    if collection_type == 'golden':
                        print(f"[DEBUG] Using Golden database response generation")
                        llm_response = self.generate_answer_with_golden_context(processing_query, doc.page_content, user_state)
                    else:
                        print(f"[DEBUG] Using RAG database response generation")
                        context = f"Context: {doc.page_content}\n\nQuestion: {processing_query}"
                        print(f"[DEBUG] LLM Context being used: {context[:200]}...")
                        llm_response = run_local_llm(
                            f"{self.REGION_INSTRUCTION}\n{self.BASE_PROMPT}\n\n{context}",
                            temperature=0.1,
                            max_tokens=1024
                        )
                    
                    print(f"[DEBUG] Generated response: {llm_response[:150]}...")
                    return {
                        'answer': llm_response,
                        'source': "RAG Database",
                        'cosine_similarity': cosine_score,
                        'document_metadata': doc.metadata,
                        'research_data': research_data
                    }
        
        print("[DB_FILTER] ✗ No suitable RAG database results found (distance >= 0.35 OR cosine similarity < 0.7)")
        return {
            'answer': "I don't have enough information to answer that from the RAG database.",
            'source': "RAG Database",
            'cosine_similarity': 0.0,
            'document_metadata': {},
            'research_data': research_data
        }
    
    def _query_pops_database_only(self, question: str, conversation_history: Optional[List[Dict]], user_state: str) -> Dict[str, any]:
        """Query only PoPs database"""
        print("[DB_FILTER] Querying PoPs database only...")
        
        processing_query = question
        if conversation_history:
            enhanced_query_result = self.context_manager.enhance_query_with_cot(question, conversation_history)
            if enhanced_query_result['requires_cot'] or enhanced_query_result['context_used']:
                processing_query = enhanced_query_result['enhanced_query']
        
        effective_location = self.determine_effective_location(question, user_state)
        result = self.search_pops_fallback(processing_query, effective_location)
        
        if result['answer'] == "__FALLBACK__":
            return {
                'answer': "I don't have enough information to answer that from the PoPs database.",
                'source': "PoPs Database",
                'cosine_similarity': 0.0,
                'document_metadata': {},
                'research_data': []
            }
        
        # Add empty research_data to PoPs result if not present
        if 'research_data' not in result:
            result['research_data'] = []
        
        return result

    def _query_llm_only(self, question: str, conversation_history: Optional[List[Dict]], user_state: str) -> Dict[str, any]:
        """Query only LLM without any database"""
        print("[DB_FILTER] Using LLM only...")
        print(f"[DEBUG] LLM-only question: {question}")
        
        processing_query = question
        if conversation_history:
            enhanced_query_result = self.context_manager.enhance_query_with_cot(question, conversation_history)
            if enhanced_query_result['requires_cot'] or enhanced_query_result['context_used']:
                processing_query = enhanced_query_result['enhanced_query']
                print(f"[DEBUG] LLM enhanced query: {processing_query}")
        
        # Use the classification system to determine response type
        category = self.classify_query(processing_query, conversation_history)
        print(f"[DEBUG] Query classified as: {category}")
        
        if category == "GREETING":
            print(f"[DEBUG] Generating greeting response")
            response = self.generate_dynamic_response(processing_query, mode="GREETING")
            source_text = "\n\n<small><i>Source: Fallback LLM</i></small>"
            final_response = response + source_text
            return {
                'answer': final_response,
                'source': "Fallback LLM",
                'cosine_similarity': 0.0,
                'document_metadata': None,
                'research_data': []
            }
        elif category == "NON_AGRI":
            print(f"[DEBUG] Generating non-agricultural response")
            response = self.generate_dynamic_response(processing_query, mode="NON_AGRI")
            source_text = "\n\n<small><i>Source: Fallback LLM</i></small>"
            final_response = response + source_text
            return {
                'answer': final_response,
                'source': "Fallback LLM",
                'cosine_similarity': 0.0,
                'document_metadata': None,
                'research_data': []
            }
        else:
            # Agriculture question - use LLM directly
            effective_location = self.determine_effective_location(question, user_state)
            context_info = f"\nUser Location: {effective_location}" if effective_location else ""
            print(f"[DEBUG] Effective location for LLM: {effective_location}")
            
            prompt = f"""You are an expert agricultural advisor specializing in Indian farming. Answer the following question with practical, actionable advice.

{self.REGION_INSTRUCTION}

Question: {processing_query}{context_info}

Provide a comprehensive answer focusing on Indian agricultural practices, varieties, and conditions. Include specific recommendations where possible."""
            
            print(f"[DEBUG] LLM prompt preview: {prompt[:200]}...")
            
            try:
                print(f"[DEBUG] Calling LLM with use_fallback=False (should use llama3.1:latest)")
                llm_response = run_local_llm(
                    prompt,
                    temperature=0.3,
                    max_tokens=1024,
                    use_fallback=False
                )
                
                print(f"[DEBUG] LLM response preview: {llm_response[:150]}...")
                
                # Add source attribution to the response
                final_response = llm_response.strip()
                source_text = "\n\n<small><i>Source: Fallback LLM</i></small>"
                final_response += source_text
                
                return {
                    'answer': final_response,
                    'source': "Fallback LLM",
                    'cosine_similarity': 0.0,
                    'document_metadata': None,
                    'research_data': []
                }
            except Exception as e:
                print(f"[DB_FILTER] LLM error: {e}")
                error_response = "I'm sorry, I'm unable to process your request at the moment. Please try again later."
                source_text = "\n\n<small><i>Source: Fallback LLM</i></small>"
                final_response = error_response + source_text
                return {
                    'answer': final_response,
                    'source': "Fallback LLM",
                    'cosine_similarity': 0.0,
                    'document_metadata': None,
                    'research_data': []
                }

    def _get_answer_with_source_main(self, question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = None) -> Dict[str, any]:
        """
        Main RAG approach for Golden/RAG databases only
        
        Returns:
            Dict containing:
            - answer: Generated response or "__FALLBACK__"
            - source: "Golden Database", "RAG Database", or "Fallback LLM"
            - cosine_similarity: Similarity score (0.0-1.0)
            - document_metadata: Metadata of the source document (if applicable)
        """
        import time
        chroma_handler_start = time.time()
        
        effective_location = self.determine_effective_location(question, user_state)
        
        try:
            greeting_check_start = time.time()
            question_lower = question.lower().strip()
            simple_greetings = [
                'hi', 'hello', 'hey', 'namaste', 'namaskaram', 'vanakkam', 
                'good morning', 'good afternoon', 'good evening', 'good day',
                'howdy', 'greetings', 'salaam', 'adaab'
            ]
            
            if len(question_lower) < 20 and any(greeting in question_lower for greeting in simple_greetings):
                greeting_time = time.time() - greeting_check_start
                if 'namaste' in question_lower:
                    fast_response = "Namaste! Welcome to AgriChat. I'm here to help you with all your farming and agriculture questions. What would you like to know about?"
                elif 'namaskaram' in question_lower:
                    fast_response = "Namaskaram! I'm your agricultural assistant. Feel free to ask me anything about crops, farming techniques, or agricultural practices."
                elif 'vanakkam' in question_lower:
                    fast_response = "Vanakkam! I'm here to assist you with farming and agriculture. What agricultural topic would you like to discuss today?"
                elif any(time in question_lower for time in ['morning', 'afternoon', 'evening']):
                    fast_response = f"Good {question_lower.split()[-1]}! I'm your agricultural assistant. How can I help you with your farming questions today?"
                else:
                    fast_response = "Hello! I'm your agricultural assistant. I'm here to help with farming, crops, and agricultural practices. What would you like to know?"
                
                return {
                    'answer': fast_response,
                    'source': "Fallback LLM",
                    'cosine_similarity': 0.0,
                    'document_metadata': None
                }
            greeting_time = time.time() - greeting_check_start
            
            context_processing_start = time.time()
            processing_query = question
            context_used = False
            
            if conversation_history:
                enhanced_query_result = self.context_manager.enhance_query_with_cot(question, conversation_history)
                
                if enhanced_query_result['requires_cot'] or enhanced_query_result['context_used']:
                    processing_query = enhanced_query_result['enhanced_query']
                    context_used = True
            context_processing_time = time.time() - context_processing_start
            
            query_classification_start = time.time()
            category = self.classify_query(question, conversation_history)
            query_classification_time = time.time() - query_classification_start
            
            if category == "GREETING":
                response = self.generate_dynamic_response(question, mode="GREETING")
                return {
                    'answer': response,
                    'source': "Fallback LLM",
                    'cosine_similarity': 0.0,
                    'document_metadata': None
                }
            
            if category == "NON_AGRI":
                response = self.generate_dynamic_response(question, mode="NON_AGRI")
                return {
                    'answer': response,
                    'source': "Fallback LLM",
                    'cosine_similarity': 0.0,
                    'document_metadata': None
                }

            database_search_start = time.time()
            try:
                query_expansion_start = time.time()
                expanded_queries = self.expand_query(processing_query)
                primary_query = expanded_queries[0]
                query_expansion_time = time.time() - query_expansion_start
                
                filter_creation_start = time.time()
                metadata_filter = self._create_metadata_filter(primary_query, effective_location)
                filter_creation_time = time.time() - filter_creation_start
                
                db_query_start = time.time()
                raw_results = self.db.similarity_search_with_score(primary_query, k=10, filter=metadata_filter)
                db_query_time = time.time() - db_query_start
                
                print(f"\n[RAG SEARCH] Query: '{primary_query}'")
                print(f"[RAG SEARCH] Database returned {len(raw_results)} results")
                
                for i, (doc, score) in enumerate(raw_results[:3]):  # Show top 3
                    print(f"   Result {i+1}: Distance={score:.3f}")
                    print(f"      Crop={doc.metadata.get('Crop', 'N/A')} | State={doc.metadata.get('State', 'N/A')}")
                    print(f"      Content Preview: {doc.page_content[:120]}...")
                
                if len(raw_results) < 3:
                    print(f"[RAG SEARCH] Few results found, trying unfiltered search...")
                    unfiltered_search_start = time.time()
                    fallback_results = self.db.similarity_search_with_score(primary_query, k=10, filter=None)
                    
                    for doc, score in fallback_results:
                        if not any(existing_doc.page_content == doc.page_content for existing_doc, _ in raw_results):
                            raw_results.append((doc, score))
                
            except Exception as e:
                print(f"[RAG SEARCH] Database search failed: {e}")
                return {
                    'answer': "__FALLBACK__",
                    'source': "Fallback LLM",
                    'cosine_similarity': 0.0,
                    'document_metadata': None
                }
            
                
            relevant_docs = self.rerank_documents(processing_query, raw_results)
            
            print(f"[SOURCE ATTRIBUTION] Reranked Results Count: {len(relevant_docs)}")
            
            crop_filtered_docs = self.validate_crop_specificity(processing_query, relevant_docs)
            
            relevant_docs = crop_filtered_docs
            
            print("=" * 70)
            for i, doc in enumerate(relevant_docs[:3]):
                print(f"DOCUMENT {i+1}:")
                print(f"   Metadata: Crop={doc.metadata.get('Crop', 'N/A')}, State={doc.metadata.get('State', 'N/A')}, District={doc.metadata.get('District', 'N/A')}")
                print(f"   Content Preview: {doc.page_content[:200]}...")
                if i < len(relevant_docs) - 1:
                    print("-" * 50)
            print("=" * 70)
            
            if not relevant_docs or not relevant_docs[0].page_content.strip():
                print(f"[EARLY QUALITY CHECK] ✗ No relevant docs or empty content")
                return {
                    'answer': "__FALLBACK__",
                    'source': "Fallback LLM",
                    'cosine_similarity': 0.0,
                    'document_metadata': None
                }
            
            content = relevant_docs[0].page_content.strip()
            print(f"[EARLY QUALITY CHECK] Content length: {len(content)} characters")
            
            if (content.lower().count('others') > 3 or 
                'Question: Others' in content or 
                'Answer: Others' in content or
                len(content) < 50):
                print(f"[EARLY QUALITY CHECK] Content quality issues - length: {len(content)}")
                return {
                    'answer': "__FALLBACK__",
                    'source': "Fallback LLM",
                    'cosine_similarity': 0.0,
                    'document_metadata': None
                }
            
            similarity_score = None
            for doc, score in raw_results:
                if doc.page_content.strip() == content:
                    similarity_score = score
                    break
            
            query_embedding = self.embedding_function.embed_query(processing_query)
            content_embedding = self.embedding_function.embed_query(content)
            cosine_score = self.cosine_sim(query_embedding, content_embedding)
            
            if similarity_score is not None and similarity_score > 0.7 and not context_used:
                print(f"[DEBUG] Rejecting due to high distance: {similarity_score:.3f} > 0.7")
                return {
                    'answer': "__FALLBACK__",
                    'source': "Fallback LLM",
                    'cosine_similarity': cosine_score,
                    'document_metadata': None
                }
            
            if cosine_score < 0.2 and not context_used:
                print(f"[DEBUG] Rejecting due to low cosine similarity: {cosine_score:.3f} < 0.2")
                return {
                    'answer': "__FALLBACK__",
                    'source': "Fallback LLM",
                    'cosine_similarity': cosine_score,
                    'document_metadata': None
                }
            
            question_lower = processing_query.lower()
            content_lower = content.lower()

            import string
            question_clean = question_lower.translate(str.maketrans('', '', string.punctuation))
            question_keywords = set()
            for word in question_clean.split():
                if len(word) > 3 and word not in {'what', 'are', 'the', 'improved', 'varieties', 'which', 'how', 'when', 'where', 'why', 'can', 'should', 'with', 'from', 'this', 'that'}:
                    question_keywords.add(word)
            
            content_keywords = set(content_lower.split())
            keyword_match = bool(question_keywords.intersection(content_keywords))
            
            print(f"[DEBUG] Question keywords: {question_keywords}")
            print(f"[DEBUG] Keyword match found: {keyword_match}")
            print(f"[DEBUG] Distance: {similarity_score:.3f}, Cosine: {cosine_score:.3f}")
            
            if not keyword_match and any(crop_word in question_lower for crop_word in ['apple', 'mango', 'orange', 'banana', 'grape', 'pomegranate']):
                fruits_in_question = [fruit for fruit in ['apple', 'mango', 'orange', 'banana', 'grape', 'pomegranate'] if fruit in question_lower]
                fruits_in_content = [fruit for fruit in ['apple', 'mango', 'orange', 'banana', 'grape', 'pomegranate'] if fruit in content_lower]
                
                if not any(fruit in fruits_in_content for fruit in fruits_in_question):
                    return {
                        'answer': "__FALLBACK__",
                        'source': "Fallback LLM",
                        'cosine_similarity': cosine_score,
                        'document_metadata': None
                    }
            
            elif not keyword_match and not context_used:
                return {
                    'answer': "__FALLBACK__",
                    'source': "Fallback LLM",
                    'cosine_similarity': cosine_score,
                    'document_metadata': None
                }
            
            document_metadata = relevant_docs[0].metadata
            
            try:
                prompt = self.construct_structured_prompt(content, processing_query, effective_location)
                
                print(f"[DEBUG] Context being sent to LLM: {content[:500]}...")
                print(f"[DEBUG] User query: {processing_query}")
                
                actual_llm_start = time.time()
                generated_response = run_local_llm(
                    prompt,
                    temperature=0.3,
                    max_tokens=1024,
                    use_fallback=False
                )
                
                print(f"[DEBUG] LLM raw response: {generated_response}")
                
                final_response = self.filter_response_thinking(generated_response.strip())
                
                is_high_quality_match = cosine_score > 0.7
                is_very_short = len(final_response.strip()) < 20
                has_insufficient_info = "I don't have enough information" in final_response
                is_rejection = final_response.strip().lower() in ["sorry", "i cannot", "not available"]
                
                if is_high_quality_match and has_insufficient_info:
                    useful_keywords = ['sown', 'sowing', 'plant', 'cultivation', 'variety', 'season', 'month', 'temperature', 'spacing', 'irrigation', 'yield', 'management']
                    has_useful_content = any(keyword in final_response.lower() for keyword in useful_keywords)
                    response_length = len(final_response.strip())
                    
                    if has_useful_content and response_length > 50:
                        print(f"[DEBUG] High-quality match (cosine={cosine_score:.3f}) with useful content despite disclaimer, proceeding...")
                    else:
                        return {
                            'answer': "__FALLBACK__",
                            'source': "Fallback LLM",
                            'cosine_similarity': cosine_score,
                            'document_metadata': document_metadata
                        }
                elif is_very_short or is_rejection:
                    return {
                        'answer': "__FALLBACK__",
                        'source': "Fallback LLM",
                        'cosine_similarity': cosine_score,
                        'document_metadata': document_metadata
                    }
                elif has_insufficient_info and not is_high_quality_match:
                    return {
                        'answer': "__FALLBACK__",
                        'source': "Fallback LLM",
                        'cosine_similarity': cosine_score,
                        'document_metadata': document_metadata
                    }
                
                quality_scores = self.quality_scorer.evaluate_response(processing_query, final_response)
                
                print(f"[MAIN RAG] Content Quality Scoring:")
                print(f"   Specificity Score: {quality_scores['specificity_score']:.3f}")
                print(f"   Relevance Score: {quality_scores['relevance_score']:.3f}")
                print(f"   Completeness Score: {quality_scores['completeness_score']:.3f}")
                print(f"   Overall Score: {quality_scores['overall_score']:.3f}")
                print(f"   Should Fallback: {quality_scores['should_fallback']}")
                if quality_scores['should_fallback'] and cosine_score < 0.6:
                    print(f"[MAIN RAG] Content quality low for non-high-confidence match, proceeding to fallback")
                    return {
                        'answer': "__FALLBACK__",
                        'source': "Fallback LLM",
                        'cosine_similarity': cosine_score,
                        'document_metadata': document_metadata
                    }
                
                determined_source = "Using RAG Database"
                
                if cosine_score > 0.7:
                    determined_source = "Using Golden Database"
                
                source_text = f"\n\n<small><i>Source: {determined_source}</i></small>"
                final_response += source_text
                
                return {
                    'answer': final_response,
                    'source': determined_source,
                    'cosine_similarity': cosine_score,
                    'document_metadata': document_metadata
                }
                
            except Exception as e:
                return {
                    'answer': "__FALLBACK__",
                    'source': "Fallback LLM",
                    'cosine_similarity': 0.0,
                    'document_metadata': None
                }
            
        except Exception as e:
            return {
                'answer': "__FALLBACK__",
                'source': "Fallback LLM",
                'cosine_similarity': 0.0,
                'document_metadata': None
            }

    def get_context_summary(self, conversation_history: List[Dict]) -> Optional[str]:
        """
        Get a brief summary of conversation context for debugging
        """
        return None


