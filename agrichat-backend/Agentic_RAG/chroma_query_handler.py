from langchain_community.vectorstores import Chroma
import numpy as np
from numpy.linalg import norm
import logging
from typing import List, Dict, Optional
from context_manager import ConversationContext
from local_llm_interface import local_llm, local_embeddings

logger = logging.getLogger("uvicorn.error")

class ChromaQueryHandler:
    STRUCTURED_PROMPT = """
You are an expert agricultural advisor specializing in helping Indian farmers with their farming questions. The user is located in {user_state}.

{region_context}

User Question: {question}

{rag_context}

Provide a comprehensive, practical answer based on the relevant information. If the information is specific to certain states or regions, tailor your response accordingly. Include actionable advice where possible.

IMPORTANT INSTRUCTIONS:
- Do NOT use placeholder text like [Your Name], [Your Region/Organization], [Company Name], or any text in square brackets
- Do NOT introduce yourself with a name or organization 
- Do NOT make assumptions about the user's specific farm, location details, or personal circumstances beyond the provided state
- Provide direct, helpful advice without template language
- If you mention the user's region, use the actual state name provided: {user_state}

Response:
"""

    CLASSIFIER_PROMPT = """
You are an intelligent agricultural advisor. Classify this query into one category based on intent and context:

AGRICULTURE - Any question about farming, crops, cultivation, soil, fertilizers, pests, diseases, irrigation, harvest, seeds, planting, agricultural practices, farm equipment, livestock, or anything related to agriculture and farming

GREETING - Simple greetings like hello, hi, namaste, good morning, how are you (without agricultural content)

NON_AGRI - Everything else (politics, sports, entertainment, general knowledge not related to farming)

Important: Consider the full context and intent. Questions like "Can I grow rice in Punjab?" or "Help me with farming" should be AGRICULTURE even if they don't contain obvious keywords.

Query: "{question}"

Classification:

Category:
"""

    GREETING_RESPONSE_PROMPT = """
User greeting: "{question}"

Respond with ONLY this format:

 Welcome! What farming question can I help you with today?

End response there. No additional sentences. No explanations.
"""

    NON_AGRI_RESPONSE_PROMPT = """
You are an agricultural advisor designed to help Indian farmers. Someone just asked you: "{question}"

Politely tell them that you specialize in helping with farming and agricultural questions only. Be friendly and respectful. Suggest they ask about crops, diseases, fertilizers, soil, irrigation, or any farming-related topics instead. Keep it brief and redirect to agricultural topics.

Speak directly to them, not about what to say to someone else. Do not provide incorrect information about non-agricultural topics.
"""

    def __init__(self, chroma_path: str, gemini_api_key: str = None, embedding_model: str = None, chat_model: str = None):
        # Use local embeddings instead of Google's API
        self.embedding_function = local_embeddings
        
        # Use local LLM instead of Gemini
        self.local_llm = local_llm
        
        self.db = Chroma(
            persist_directory=chroma_path,
            embedding_function=self.embedding_function,
        )
        
        self.context_manager = ConversationContext(
            max_context_pairs=3,
            max_context_tokens=500
        )
        
        col = self.db._collection.get()["metadatas"]
        self.meta_index = {
            field: {m[field] for m in col if field in m and m[field]}
            for field in [
                "Year", "Month", "Day",
                "Crop", "District", "Season", "Sector", "State"
            ]
        }

    def get_region_context(self, question: str = None, region: str = None):
        """
        Get region-specific context for greetings and responses
        """
        # Regional farming context and greetings
        regional_data = {
            "Andhra Pradesh": {
                "greeting": "You can use Telugu phrases like 'Namaskaram' if appropriate.",
                "context": "Andhra Pradesh is known for rice, cotton, sugarcane, and groundnut cultivation. The state has diverse agro-climatic zones with both kharif and rabi seasons being important.",
                "crops": ["rice", "cotton", "sugarcane", "groundnut", "chili", "turmeric"]
            },
            "Telangana": {
                "greeting": "You can use Telugu phrases like 'Namaskaram' if appropriate.",
                "context": "Telangana is famous for rice, cotton, maize, and various millets. The state promotes sustainable farming with initiatives like Rythu Bandhu.",
                "crops": ["rice", "cotton", "maize", "millets", "turmeric", "red gram"]
            },
            "Tamil Nadu": {
                "greeting": "You can use Tamil phrases like 'Vanakkam' if appropriate.",
                "context": "Tamil Nadu excels in rice, sugarcane, cotton, and coconut cultivation. The state has strong irrigation systems and diverse cropping patterns.",
                "crops": ["rice", "sugarcane", "cotton", "coconut", "banana", "turmeric"]
            },
            "Kerala": {
                "greeting": "You can use Malayalam phrases like 'Namaskaram' if appropriate.",
                "context": "Kerala is renowned for spices, coconut, rubber, and rice cultivation. The state has unique farming practices suited to its tropical climate.",
                "crops": ["coconut", "rubber", "rice", "pepper", "cardamom", "tea"]
            },
            "Maharashtra": {
                "greeting": "You can use Marathi phrases like 'Namaskar' if appropriate.",
                "context": "Maharashtra is a major producer of sugarcane, cotton, soybeans, and various fruits. The state has both rain-fed and irrigated agriculture.",
                "crops": ["sugarcane", "cotton", "soybeans", "onion", "grapes", "pomegranate"]
            },
            "Haryana": {
                "greeting": "You can use Hindi phrases like 'Namaste' if appropriate.",
                "context": "Haryana is known as the 'Granary of India' with extensive wheat and rice cultivation. The state has advanced agricultural practices and infrastructure.",
                "crops": ["wheat", "rice", "sugarcane", "cotton", "mustard", "barley"]
            },
            "Uttar Pradesh": {
                "greeting": "You can use Hindi phrases like 'Namaste' if appropriate.",
                "context": "Uttar Pradesh is India's largest agricultural state producing wheat, rice, sugarcane, and various other crops across diverse agro-climatic zones.",
                "crops": ["wheat", "rice", "sugarcane", "potato", "peas", "mustard"]
            },
            "Punjab": {
                "greeting": "You can use Punjabi phrases like 'Sat Sri Akal' or Hindi 'Namaste' if appropriate.",
                "context": "Punjab is known as the 'Granary of India' and 'Land of Five Rivers', famous for wheat and rice cultivation. The state has excellent irrigation infrastructure and is a major contributor to India's food grain production.",
                "crops": ["wheat", "rice", "cotton", "maize", "sugarcane", "potato"]
            }
        }
        
        detected_region = region
        
        # If no explicit region, try to detect from question
        if not detected_region and question:
            question_lower = question.lower()
            for state_name in self.meta_index.get('State', set()):
                if state_name.lower() in question_lower:
                    detected_region = state_name
                    break
            
            # Also check districts to infer state
            if not detected_region:
                for district_name in self.meta_index.get('District', set()):
                    if district_name.lower() in question_lower:
                        # You could add district to state mapping here
                        detected_region = f"your region ({district_name})"
                        break
        
        # Normalize state names
        normalized_states = {
            "andhra pradesh": "Andhra Pradesh",
            "telangana": "Telangana", 
            "tamil nadu": "Tamil Nadu",
            "kerala": "Kerala",
            "maharastra": "Maharashtra",
            "maharashtra": "Maharashtra",
            "haryana": "Haryana",
            "uttar pradesh": "Uttar Pradesh"
        }
        
        if detected_region and detected_region.lower() in normalized_states:
            detected_region = normalized_states[detected_region.lower()]
        
        # Get regional context
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
        Expand query with variations to handle 'twisted' questions better
        """
        expanded = [query]
        
        try:
            agri_keywords = self._extract_agricultural_terms(query)
            
            if agri_keywords:
                keyword_query = " ".join(agri_keywords)
                expanded.append(keyword_query)
            
            simplified = self._simplify_query(query)
            if simplified and simplified != query:
                expanded.append(simplified)
            
            alternative = self._rephrase_query(query)
            if alternative and alternative not in expanded:
                expanded.append(alternative)
            
        except Exception as e:
            logger.warning(f"[RAG] Error expanding query: {e}")
        
        return expanded[:5]
    
    def _extract_agricultural_terms(self, text):
        """Extract agricultural terms from text"""
        agricultural_terms = {
            'crop', 'crops', 'farming', 'agriculture', 'cultivation', 'harvest', 'seeds', 'soil',
            'fertilizer', 'pesticide', 'irrigation', 'planting', 'sowing', 'yield', 'farm', 'field',
            'rice', 'wheat', 'corn', 'maize', 'cotton', 'sugarcane', 'vegetables', 'fruits',
            'organic', 'disease', 'pest', 'insect', 'weed', 'water', 'rain', 'drought', 'season',
            'growth', 'production', 'farmer', 'agricultural', 'land', 'machinery', 'equipment'
        }
        
        words = text.lower().split()
        found_terms = [word for word in words if word in agricultural_terms]
        return found_terms
    
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

    def _create_metadata_filter(self, question):
        """
        Create metadata filter from question, but avoid overly restrictive filters
        that might cause ChromaDB query errors
        """
        q = question.lower()
        filt = {}
        
        safe_fields = ["Crop", "District", "State", "Season"]
        
        for field in safe_fields:
            if field in self.meta_index:
                for val in self.meta_index[field]:
                    if str(val).lower() in q and str(val).lower() != "other" and str(val).lower() != "-":
                        filt[field] = val
                        break
        
        if len(filt) > 2:
            filt = dict(list(filt.items())[:2])
            
        return filt or None

    def cosine_sim(self, a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    def rerank_documents(self, question: str, results, top_k: int = 5):
        """Enhanced reranking with better similarity scoring and multiple strategies"""
        query_embedding = self.embedding_function.embed_query(question)
        scored = []
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        for doc, original_score in results:
            content = doc.page_content.strip()
            
            if (content.lower().count('others') > 3 or 
                'Question: Others' in content or 
                'Answer: Others' in content or
                content.count('Others') > 5):
                continue
            
            d_emb = self.embedding_function.embed_query(content)
            cosine_score = self.cosine_sim(query_embedding, d_emb)
            
            content_words = set(content.lower().split())
            keyword_overlap = len(question_words.intersection(content_words)) / len(question_words) if question_words else 0
            
            agri_terms = {'crop', 'plant', 'disease', 'pest', 'fertilizer', 'soil', 'water', 'harvest', 'seed', 'growth', 'yield'}
            question_agri_terms = question_words.intersection(agri_terms)
            content_agri_terms = content_words.intersection(agri_terms)
            agri_overlap = len(question_agri_terms.intersection(content_agri_terms)) / max(len(question_agri_terms), 1)
            
            combined_score = (cosine_score * 0.6) + (keyword_overlap * 0.25) + (agri_overlap * 0.15)
            
            scored.append((doc, combined_score, cosine_score, original_score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, combined_score, cosine_score, orig_score in scored[:top_k] 
                if doc.page_content.strip() and combined_score > 0.3]

    def construct_structured_prompt(self, context: str, question: str, user_state: str = None) -> str:
        # Get regional context for the user's actual state
        region_data = self.get_region_context(None, user_state)
        user_region = region_data["region"]
        regional_context = region_data["context"] 
        regional_crops = ", ".join(region_data["crops"])
        
        # Build state-specific prompt
        state_instruction = ""
        if user_state and user_state != "India":
            state_instruction = f"""
IMPORTANT: The user is from {user_region}. {regional_context} 
Focus your advice specifically for {user_region} farmers. The main crops in {user_region} are: {regional_crops}.
Do NOT mention other states like West Bengal, Hooghly, or any other region in your response. 
Speak directly about {user_region} conditions and crops.
"""
        
        full_prompt = f"""
You are a knowledgeable agricultural advisor who understands the challenges Indian farmers face. Respond in a warm, friendly, and supportive manner. Use the provided information to give practical, actionable advice that farmers can actually implement.

{state_instruction}

### How to respond:
- Start with a friendly acknowledgment like "That's a great question!" or "I understand your concern"  
- Use simple, clear language that any farmer can understand - avoid too much technical jargon
- Give practical solutions that work in Indian conditions and are affordable
- Show empathy for farming challenges and offer encouragement
- Include both modern and traditional methods when helpful
- Never mention any system names, databases, or technical details
- Don't make personal assumptions about the user's work or background
- NEVER mention states other than the user's actual state

### For Disease, Pest, or Input-related Questions:
- Always provide BOTH expensive and affordable solutions:
  * **Professional solutions**: Chemical fertilizers, branded pesticides, fungicides
  * **Budget-friendly solutions**: Home remedies, organic methods, local ingredients (neem, turmeric, cow dung, etc.)
- Explain which option works best in different situations
- Mention timing and weather considerations important in India

### Keep it practical and friendly:
- Use bullet points or simple headings for clarity
- Give step-by-step advice when needed
- End with encouragement or invitation for follow-up questions
- Focus on solutions that work in Indian villages and farms

**If the context doesn't have relevant information, reply exactly:**
`I don't have enough information to answer that.`

### Context
{context}

### User Question  
{question}
---
### Your Advice:
"""
        return full_prompt

    def classify_query(self, question: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """Enhanced classifier that uses LLM intelligence with conversation context"""
        
        question_lower = question.lower().strip()
        
        # Handle obvious simple greetings first (single words/phrases)
        simple_greetings = {'hello', 'hi', 'hey', 'namaste', 'namaskar', 'namaskaram', 'vanakkam', 'good morning', 'good afternoon', 'good evening'}
        if question_lower in simple_greetings:
            return "GREETING"
        
        # Check conversation context for follow-up questions
        if conversation_history and len(conversation_history) > 0:
            # Check if this looks like a follow-up question
            follow_up_indicators = ['yes', 'please help', 'tell me more', 'what about', 'i want to know', 'help me']
            is_follow_up = any(indicator in question_lower for indicator in follow_up_indicators)
            
            if is_follow_up:
                # Check if the previous question was agricultural
                last_interaction = conversation_history[-1]
                if last_interaction and 'question' in last_interaction:
                    prev_question = last_interaction['question'].lower()
                    # Use LLM to classify the previous question
                    prev_classification = self._classify_with_llm(prev_question)
                    if prev_classification == "AGRICULTURE":
                        return "AGRICULTURE"
        
        # Use LLM for intelligent classification
        return self._classify_with_llm(question)
    
    def _classify_with_llm(self, question: str) -> str:
        """Use LLM for intelligent classification"""
        prompt = self.CLASSIFIER_PROMPT.format(question=question)
        try:
            response_text = self.local_llm.generate_content(
                prompt=prompt,
                temperature=0,
                max_tokens=20
            )
            category = response_text.strip().upper()
            
            # Validate response
            valid_categories = {"AGRICULTURE", "GREETING", "NON_AGRI"}
            if category in valid_categories:
                return category
            
            # If LLM returned something else, try to parse it
            if "AGRICULTURE" in response_text.upper():
                return "AGRICULTURE"
            elif "GREETING" in response_text.upper():
                return "GREETING"
            else:
                return "NON_AGRI"
                
        except Exception as e:
            logger.error(f"[Classifier Error] {e}")
            # Fallback: Basic keyword detection
            agri_keywords = ['crop', 'farm', 'plant', 'soil', 'seed', 'grow', 'cultivation', 'rice', 'wheat', 'fertilizer']
            if any(word in question.lower() for word in agri_keywords):
                return "AGRICULTURE"
            return "NON_AGRI"

    def generate_dynamic_response(self, question: str, mode: str, region: str = None) -> str:
        if mode == "GREETING":
            # Get region-specific context
            region_data = self.get_region_context(question, region)
            prompt = self.GREETING_RESPONSE_PROMPT.format(
                question=question,
                region=region_data["region"],
                region_context=region_data["context"],
                regional_greeting=region_data["greeting"]
            )
        else:
            prompt = self.NON_AGRI_RESPONSE_PROMPT.format(question=question)
        try:
            response_text = self.local_llm.generate_content(
                prompt=prompt,
                temperature=0.5,
                max_tokens=300
            )
            return response_text.strip()
        except Exception as e:
            logger.error(f"[Dynamic Response Error] {e}")
            return "Sorry, I can only help with agriculture-related questions."

    def get_answer(self, question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = None) -> str:
        """
        FIXED VERSION - Get answer with optional conversation context and user location
        
        Args:
            question: Current user question
            conversation_history: List of previous Q&A pairs for context
            user_state: User's state/region detected from frontend location
        """
        try:
            # Step 1: Handle conversation context
            processing_query = question
            context_used = False
            
            if conversation_history:
                should_use_context = self.context_manager.should_use_context(question, conversation_history)
                if should_use_context:
                    processing_query = self.context_manager.build_contextual_query(question, conversation_history)
                    context_used = True
            
            # Step 2: Classify the question (now with conversation context)
            category = self.classify_query(question, conversation_history)
            
            # Step 3: Detect region - prioritize frontend state over question parsing
            region_data = self.get_region_context(question, user_state)
            detected_region = region_data["region"]
            
            # Handle non-agricultural questions
            if category == "GREETING":
                response = self.generate_dynamic_response(question, mode="GREETING", region=detected_region)
                return f"__NO_SOURCE__{response}"
            
            if category == "NON_AGRI":
                response = self.generate_dynamic_response(question, mode="NON_AGRI", region=detected_region)
                return f"__NO_SOURCE__{response}"
            
            # Step 3: Handle agricultural questions with RAG
            raw_results = self.db.similarity_search_with_score(processing_query, k=15, filter=None)
            relevant_docs = self.rerank_documents(processing_query, raw_results)
            
            # Check if we have good content
            if relevant_docs and len(relevant_docs) > 0:
                content = relevant_docs[0].page_content.strip()
                
                # Skip low quality content
                if (content.lower().count('others') > 3 or 
                    'Question: Others' in content or 
                    'Answer: Others' in content or
                    len(content) < 50):
                    return "I don't have enough information to answer that."
                
                # Generate RAG response
                final_question = question if context_used else processing_query
                prompt = self.construct_structured_prompt(content, final_question, user_state)
                
                generated_response = self.local_llm.generate_content(
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=1024
                )
                
                return generated_response
            
            # Fallback if no good content found
            return "I don't have enough information to answer that."
            
        except Exception as e:
            logger.error(f"[Error in get_answer] {e}")
            return "I don't have enough information to answer that."

    def get_context_summary(self, conversation_history: List[Dict]) -> Optional[str]:
        """
        Get a brief summary of conversation context for debugging
        """
        return self.context_manager.get_context_summary(conversation_history)
