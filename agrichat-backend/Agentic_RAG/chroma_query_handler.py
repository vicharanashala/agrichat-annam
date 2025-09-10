from langchain_community.vectorstores import Chroma
import numpy as np
from numpy.linalg import norm
import logging
import re
from typing import List, Dict, Optional
from context_manager import ConversationContext, MemoryStrategy
from local_llm_interface import local_llm, local_embeddings, run_local_llm
from datetime import datetime
import pytz
import argparse
import hashlib
from functools import lru_cache

logger = logging.getLogger("uvicorn.error")

class ChromaQueryHandler:

    REGION_INSTRUCTION = """
IMPORTANT: If a state or region is mentioned in the query, always give preference to that region over the user's location. If the mentioned region is outside India, politely inform the user that you are only trained on Indian agriculture data and cannot answer for other regions. If the query is not related to Indian agriculture, politely inform the user that you are only trained on Indian agriculture data and can only answer questions related to Indian agriculture.
"""

    STRUCTURED_PROMPT = """
You are an expert agricultural assistant. Using only the provided context (do not mention or reveal any metadata such as date, district, state, or season unless the user asks for it), answer the user's question in a detailed and structured manner. Stay strictly within the scope of the user's question and do not introduce unrelated information.

IMPORTANT: Always respond in the same language in which the query has been asked.

The user is located in {user_state}. {region_context}
{region_instruction}
Current month: {current_month}

### Detailed Explanation
- Provide a comprehensive, step-by-step explanation using both the context and your own agricultural knowledge, but only as it directly relates to the user's question.
- Use bullet points, sub-headings, or tables to clarify complex information.
- Reference and explain all relevant data points from the context.
- Briefly define technical terms inline if needed.
- Avoid botanical or scientific explanations that are not relevant to farmers unless explicitly asked.
- While focusing on {user_state} conditions, you can also provide information about crops not typically grown in {user_state} if the user specifically asks about them.

### Special Instructions for Disease, Fertilizer, Fungicide, or Tonic Queries
- Whenever a question relates to disease, pest attacks, use of fertilizers, fungicides, plant tonics, or similar agricultural inputs—even if not explicitly stated—include both:
- Standard recommendations involving chemical fertilizers, fungicides, or plant protection chemicals.
- Quick low-cost household/natural solutions suitable for farmers seeking alternative approaches.
- For each method, explain when and why it may be preferable, and note any precautions if relevant.
- Ensure both professional and practical solutions are always offered unless the question strictly forbids one or the other.

### Additional Guidance for General Crop Management Questions (e.g., maximizing yield, disease prevention, necessary precautions)
- If a general question is asked about growing a particular crop and the database contains information related to that crop, analyze the context of the user's question (such as disease prevention, yield maximization, or best practices).
- Retrieve and provide all relevant guidance from the database about that crop, including:
    - Disease management
    - Best agronomic practices to maximize yield
    - Important precautions and crop requirements
    - Fertilizer and input recommendations
    - Any risks and general crop care tips

### To keep responses concise and focused, the answer should:
    - Only address the specific question asked, using clear bullet points, tables, or short sub-headings as needed.
    - Make sure explanations are actionable, practical, and relevant for farmers—avoiding lengthy background or scientific context unless requested.
    - For questions about diseases, fertilizers, fungicides, or tonics:
    - Briefly provide both standard (chemical) and quick low-cost/natural solutions, each with a very short explanation and clear usage or caution notes.
    - For broader crop management questions, summarize key data points (disease management, input use, care tips, risks) in a succinct, easy-to-use manner—only including what's relevant to the query.
    - Never add unrelated information, avoid detailed paragraphs unless multiple issues are asked, and always keep the response direct and farmer-friendly.

- Even if the database information does not directly match the question, use context and reasoning to include all data points from the database that could help answer the user's general query about that crop.
- Your response should synthesize the relevant parts of the database connected to the user's request, offering a complete, actionable answer.

IMPORTANT INSTRUCTIONS:
- Do NOT use placeholder text like [Your Name], [Your Region/Organization], [Company Name], or any text in square brackets
- Do NOT introduce yourself with a name or organization 
- Do NOT make assumptions about the user's specific farm, location details, or personal circumstances beyond the provided state
- Do NOT include your reasoning, thinking process, or assumptions in the answer (e.g., "Since the user is located in...", "I will assume...", "However, since the question does not mention...")
- Do NOT explain why you're providing certain information or how you interpreted the question
- Start directly with the agricultural advice and information
- If this appears to be a follow-up question, acknowledge the previous context naturally
- Provide direct, helpful advice without template language or meta-commentary

**CRITICAL: If the context does not contain relevant information to properly answer the question, reply exactly:**   
`I don't have enough information to answer that.`

**Do NOT attempt to provide general agricultural advice or information if it's not specifically supported by the context provided above.**

### Context
{context}

### User Question
{question}
---
### Your Answer:
"""

    CLASSIFIER_PROMPT = """
You are a smart classifier assistant. Categorize the user query strictly into one of the following categories:

- AGRICULTURE: if the question is related to farming, crops, fertilizers, pests, soil, irrigation, harvest, agronomy, etc.
- GREETING: if it is a greeting, salutation, polite conversational opening, or introduction. This includes messages like "hi", "hello", "good morning", "Hello [name]", "Hi there", "How are you", "Nice to meet you", etc.
- NON_AGRI: if the question is not agriculture-related or contains inappropriate, offensive, or irrelevant content.

{region_instruction}
Current month: {current_month}

Important: Greetings can include names or additional polite phrases. Focus on the intent of the message.

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

Politely tell them that you can only help with agriculture-related questions. Be friendly and respectful.

Respond as if you are talking directly to the user, not giving advice on what to say to someone else.
"""

    def __init__(self, chroma_path: str, gemini_api_key: str = None, embedding_model: str = None, chat_model: str = None):
        self.embedding_function = local_embeddings
        
        self.local_llm = local_llm
        
        self.db = Chroma(
            persist_directory=chroma_path,
            embedding_function=self.embedding_function,
        )
        
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
            
            # More conservative scoring for stricter filtering
            combined_score = (cosine_score * 0.7) + (keyword_overlap * 0.2) + (agri_overlap * 0.1)
            
            scored.append((doc, combined_score, cosine_score, original_score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Use stricter threshold for filtering documents
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
        if user_state:
            region_data = self.get_region_context(None, user_state)
            user_region = region_data["region"]
            regional_context = region_data["context"]
            
            return self.STRUCTURED_PROMPT.format(
                context=context,
                question=question,
                user_state=user_region,
                region_context=regional_context,
                region_instruction=self.REGION_INSTRUCTION,
                current_month=current_month
            )
        else:
            return self.STRUCTURED_PROMPT.format(
                context=context,
                question=question,
                user_state="India",
                region_context="India has diverse agro-climatic zones with rich farming traditions across different states and regions.",
                region_instruction=self.REGION_INSTRUCTION,
                current_month=current_month
            )

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
            logger.error(f"[Classifier Error] {e}")
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
            logger.error(f"[Dynamic Response Error] {e}")
            return "Sorry, I can only help with agriculture-related questions."

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
        import time
        chroma_handler_start = time.time()
        logger.info(f"[TIMING] ChromaQueryHandler.get_answer started for: {question[:50]}...")
        
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
                logger.info(f"[TIMING] Greeting detection took: {greeting_time:.3f}s")
                logger.info(f"[FAST GREETING] Detected simple greeting: {question}")
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
                
                return f"__NO_SOURCE__{fast_response}"
            greeting_time = time.time() - greeting_check_start
            logger.info(f"[TIMING] Greeting check took: {greeting_time:.3f}s")
            
            context_processing_start = time.time()
            processing_query = question
            context_used = False
            
            if conversation_history:
                logger.info(f"[Context DEBUG] Received conversation history with {len(conversation_history)} entries")
                enhanced_query_result = self.context_manager.enhance_query_with_cot(question, conversation_history)
                
                if enhanced_query_result['requires_cot'] or enhanced_query_result['context_used']:
                    processing_query = enhanced_query_result['enhanced_query']
                    context_used = True
                    logger.info(f"[CoT DEBUG] Using enhanced query with context")
            context_processing_time = time.time() - context_processing_start
            logger.info(f"[TIMING] Context processing took: {context_processing_time:.3f}s")
            
            query_classification_start = time.time()
            category = self.classify_query(question, conversation_history)
            query_classification_time = time.time() - query_classification_start
            logger.info(f"[TIMING] Query classification took: {query_classification_time:.3f}s")
            
            if category == "GREETING":
                response = self.generate_dynamic_response(question, mode="GREETING")
                return f"__NO_SOURCE__{response}"
            
            if category == "NON_AGRI":
                response = self.generate_dynamic_response(question, mode="NON_AGRI")
                return f"__NO_SOURCE__{response}"

            database_search_start = time.time()
            try:
                query_expansion_start = time.time()
                expanded_queries = self.expand_query(processing_query)
                primary_query = expanded_queries[0]
                query_expansion_time = time.time() - query_expansion_start
                logger.info(f"[TIMING] Query expansion took: {query_expansion_time:.3f}s")
                
                try:
                    filter_creation_start = time.time()
                    metadata_filter = self._create_metadata_filter(primary_query, user_state)
                    filter_creation_time = time.time() - filter_creation_start
                    logger.info(f"[TIMING] Filter creation took: {filter_creation_time:.3f}s")
                    
                    filtered_search_start = time.time()
                    raw_results = self.db.similarity_search_with_score(primary_query, k=10, filter=metadata_filter)
                    filtered_search_time = time.time() - filtered_search_start
                    logger.info(f"[TIMING] Filtered search took: {filtered_search_time:.3f}s")
                    
                    if len(raw_results) < 3:
                        unfiltered_search_start = time.time()
                        raw_results = self.db.similarity_search_with_score(primary_query, k=10, filter=None)
                        unfiltered_search_time = time.time() - unfiltered_search_start
                        logger.info(f"[TIMING] Unfiltered search took: {unfiltered_search_time:.3f}s")
                except Exception as e:
                    logger.warning(f"[RAG] Error with filtered search: {e}")
                    fallback_search_start = time.time()
                    raw_results = self.db.similarity_search_with_score(primary_query, k=10, filter=None)
                    fallback_search_time = time.time() - fallback_search_start
                    logger.info(f"[TIMING] Fallback search took: {fallback_search_time:.3f}s")
                
            except Exception as e:
                logger.error(f"[RAG] Database search failed: {e}")
                return "__FALLBACK__"
            
            database_search_time = time.time() - database_search_start
            logger.info(f"[TIMING] Database search total took: {database_search_time:.3f}s")
                
            rerank_start = time.time()
            relevant_docs = self.rerank_documents(processing_query, raw_results)
            rerank_time = time.time() - rerank_start
            logger.info(f"[TIMING] Document reranking took: {rerank_time:.3f}s")
            
            quality_check_start = time.time()
            if not relevant_docs or not relevant_docs[0].page_content.strip():
                logger.info(f"[RAG] No relevant documents found")
                return "__FALLBACK__"
            
            content = relevant_docs[0].page_content.strip()
            
            if (content.lower().count('others') > 3 or 
                'Question: Others' in content or 
                'Answer: Others' in content or
                len(content) < 50):
                logger.info(f"[RAG] Content failed quality filter")
                return "__FALLBACK__"
            
            similarity_score = None
            for doc, score in raw_results:
                if doc.page_content.strip() == content:
                    similarity_score = score
                    break
            
            # Much stricter threshold - similarity scores are distances (lower is better)
            # ChromaDB returns distances, not similarities, so lower values indicate better matches
            max_distance_threshold = 0.5  # Maximum allowed distance for RAG response
            
            if similarity_score is not None and similarity_score > max_distance_threshold and not context_used:
                logger.info(f"[RAG] Document distance too high: {similarity_score:.3f} > {max_distance_threshold}")
                return "__FALLBACK__"
            
            # Additional check: ensure document has sufficient relevance through combined scoring
            query_embedding = self.embedding_function.embed_query(processing_query)
            content_embedding = self.embedding_function.embed_query(content)
            cosine_score = self.cosine_sim(query_embedding, content_embedding)
            
            min_relevance_threshold = 0.3  # Minimum cosine similarity for relevance
            if cosine_score < min_relevance_threshold and not context_used:
                logger.info(f"[RAG] Content relevance too low: {cosine_score:.3f} < {min_relevance_threshold}")
                return "__FALLBACK__"
            
            # NEW: Check if the content actually contains information about the queried crop/topic
            question_lower = processing_query.lower()
            content_lower = content.lower()
            
            # Extract key topics from the question
            question_keywords = set()
            for word in question_lower.split():
                if len(word) > 3 and word not in {'what', 'are', 'the', 'improved', 'varieties', 'which', 'how', 'when', 'where', 'why', 'can', 'should'}:
                    question_keywords.add(word)
            
            # Check if any important keywords from question appear in content
            content_keywords = set(content_lower.split())
            keyword_match = bool(question_keywords.intersection(content_keywords))
            
            # Special check for crop-specific queries
            if not keyword_match and any(crop_word in question_lower for crop_word in ['apple', 'mango', 'orange', 'banana', 'grape', 'pomegranate']):
                # For fruit queries, ensure the content mentions the same fruit
                fruits_in_question = [fruit for fruit in ['apple', 'mango', 'orange', 'banana', 'grape', 'pomegranate'] if fruit in question_lower]
                fruits_in_content = [fruit for fruit in ['apple', 'mango', 'orange', 'banana', 'grape', 'pomegranate'] if fruit in content_lower]
                
                if not any(fruit in fruits_in_content for fruit in fruits_in_question):
                    logger.info(f"[RAG] Content topic mismatch - Question fruits: {fruits_in_question}, Content fruits: {fruits_in_content}")
                    return "__FALLBACK__"
            
            elif not keyword_match and not context_used:
                logger.info(f"[RAG] No keyword overlap between question and content")
                logger.info(f"[RAG] Question keywords: {question_keywords}")
                logger.info(f"[RAG] Content preview: {content[:100]}...")
                return "__FALLBACK__"
            quality_check_time = time.time() - quality_check_start
            logger.info(f"[TIMING] Quality check took: {quality_check_time:.3f}s")
            
            llm_generation_start = time.time()
            try:
                prompt_construction_start = time.time()
                prompt = self.construct_structured_prompt(content, processing_query, user_state)
                prompt_construction_time = time.time() - prompt_construction_start
                logger.info(f"[TIMING] Prompt construction took: {prompt_construction_time:.3f}s")
                
                actual_llm_start = time.time()
                generated_response = run_local_llm(
                    prompt,
                    temperature=0.3,
                    max_tokens=1024,
                    use_fallback=False
                )
                actual_llm_time = time.time() - actual_llm_start
                logger.info(f"[TIMING] LLM generation took: {actual_llm_time:.3f}s")
                
                response_processing_start = time.time()
                logger.info(f"[RAG] Response generated from database (distance: {similarity_score:.3f}, relevance: {cosine_score:.3f})")
                final_response = self.filter_response_thinking(generated_response.strip())
                
                # Check if the generated response is meaningful
                if (len(final_response.strip()) < 20 or 
                    "I don't have enough information" in final_response or
                    final_response.strip().lower() in ["sorry", "i cannot", "not available"]):
                    logger.info(f"[RAG] Generated response insufficient, falling back to tools")
                    return "__FALLBACK__"
                
                final_response += "\n\n<small><i>Source: RAG Database</i></small>"
                response_processing_time = time.time() - response_processing_start
                logger.info(f"[TIMING] Response processing took: {response_processing_time:.3f}s")
                
                llm_generation_time = time.time() - llm_generation_start
                logger.info(f"[TIMING] Total LLM generation took: {llm_generation_time:.3f}s")
                
                total_chroma_time = time.time() - chroma_handler_start
                logger.info(f"[TIMING] TOTAL ChromaQueryHandler took: {total_chroma_time:.3f}s")
                
                return final_response
                
            except Exception as e:
                logger.error(f"[RAG] Response generation failed: {e}")
                return "__FALLBACK__"
            
        except Exception as e:
            logger.error(f"[RAG] Unexpected error: {e}")
            return "__FALLBACK__"

    def get_context_summary(self, conversation_history: List[Dict]) -> Optional[str]:
        """
        Get a brief summary of conversation context for debugging
        """
        return None



if __name__ == "__main__":
    handler = ChromaQueryHandler(chroma_path=r"/home/ubuntu/agrichat-annam/agrichat-backend/Agentic_RAG/Knowledge_base/chromaDb")
    test_question = "What are the improved varieties of Apple?"
    response = handler.get_answer(test_question, user_state="haryana")
    print(response)