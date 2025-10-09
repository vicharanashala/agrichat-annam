from fastapi import FastAPI, Request, Form, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pymongo import MongoClient
from uuid import uuid4
from datetime import datetime
import sys
import os
import pytz
import logging
import csv
import time
from typing import Optional, List, Dict
import asyncio
from contextlib import asynccontextmanager
from io import StringIO
from local_whisper_interface import get_whisper_instance
import re
from Agentic_RAG.fast_response_handler import FastResponseHandler
from Agentic_RAG import main as rag_main
from Agentic_RAG.database_config import DatabaseConfig
import json
import markdown
from fastapi.responses import StreamingResponse
from langchain.memory import ConversationBufferWindowMemory


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")
current_dir = os.path.dirname(os.path.abspath(__file__))
agentic_rag_path = os.path.join(current_dir, "Agentic_RAG")
sys.path.insert(0, agentic_rag_path)

USE_FAST_MODE = os.getenv("USE_FAST_MODE", "true").lower() == "true"

class QueryRequest(BaseModel):
    question: str
    device_id: str
    state: str = ""
    language: str = "en"
    database_config: Optional[Dict] = None

class SessionQueryRequest(BaseModel):
    question: str
    device_id: str
    state: str = ""
    database_config: Optional[Dict] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    authenticated: bool
    username: Optional[str] = None
    role: Optional[str] = None
    full_name: Optional[str] = None
    message: str = ""

fast_handler = None
if USE_FAST_MODE:
    try:
        fast_handler = FastResponseHandler()
    except Exception as e:
        logger.warning(f"[CONFIG] Fast response handler initialization failed: {e}")
        USE_FAST_MODE = False

from Agentic_RAG.tools import is_agricultural_query

MONGO_URI = os.getenv("MONGO_URI")
try:
    client = MongoClient(MONGO_URI)
    db = client.get_database("agrichat")
    sessions_collection = db["sessions"]
    logger.info("[DB] MongoDB client initialized and 'sessions' collection ready")
except Exception as e:
    logger.error(f"[DB] Failed to initialize MongoDB client: {e}")
    sessions_collection = None

def load_users_from_csv():
    """Load users from CSV file"""
    users = {}
    csv_path = os.path.join(os.path.dirname(__file__), "users.csv")
    
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                users[row['username']] = {
                    'password': row['password'],
                    'role': row['role'],
                    'full_name': row['full_name']
                }
        logger.info(f"[AUTH] Loaded {len(users)} users from CSV")
    except FileNotFoundError:
        logger.error(f"[AUTH] Users CSV file not found at {csv_path}")
    except Exception as e:
        logger.error(f"[AUTH] Error loading users CSV: {e}")
    
    return users

def authenticate_user(username: str, password: str) -> dict:
    """Authenticate user credentials"""
    users = load_users_from_csv()
    
    if username in users and users[username]['password'] == password:
        return {
            'authenticated': True,
            'username': username,
            'role': users[username]['role'],
            'full_name': users[username]['full_name']
        }
    
    return {'authenticated': False}

recommendations_cache = {}
CACHE_EXPIRY_SECONDS = 3600
session_memories = {}

def get_session_memory(session_id: str):
    """Get or create conversation memory for a session"""
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferWindowMemory(
            k=8,  # Tune this number: 3=light, 5=default, 8-10=extended, 15+=heavy
            return_messages=True
        )
    return session_memories[session_id]

def format_conversation_context(memory: ConversationBufferWindowMemory) -> str:
    """Enhanced formatting of conversation history for better context"""
    if not memory.chat_memory.messages:
        return "This is the start of the conversation."
    
    context_parts = []
    messages = memory.chat_memory.messages[-10:]  # Last 10 messages
    
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            human_msg = messages[i]
            ai_msg = messages[i + 1]
            
            human_content = human_msg.content
            ai_content = ai_msg.content[:300] + "..." if len(ai_msg.content) > 300 else ai_msg.content
            
            context_parts.append(f"Previous Q: {human_content}")
            context_parts.append(f"Previous A: {ai_content}")
    
    all_content = " ".join([msg.content for msg in messages])
    topics = extract_topics_from_context(all_content)
    
    if topics:
        context_parts.append(f"Main topics discussed: {', '.join(topics[:3])}")
    
    return "\n".join(context_parts)

def convert_langchain_memory_to_history(memory: ConversationBufferWindowMemory) -> List[Dict]:
    """
    Convert LangChain memory messages to the format expected by the RAG system.
    Returns list of {'question': str, 'answer': str} dictionaries.
    """
    try:
        if not memory or not memory.chat_memory or not memory.chat_memory.messages:
            return []
        
        conversation_history = []
        messages = memory.chat_memory.messages
        
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i]
                ai_msg = messages[i + 1]
                
                if (hasattr(human_msg, 'content') and hasattr(ai_msg, 'content') and
                    human_msg.content and ai_msg.content):
                    conversation_history.append({
                        'question': str(human_msg.content),
                        'answer': str(ai_msg.content)
                    })
        
        logger.info(f"[LANGCHAIN] Converted {len(messages)} messages to {len(conversation_history)} conversation pairs")
        return conversation_history
    
    except Exception as e:
        logger.error(f"[LANGCHAIN] Error converting memory to history: {e}")
        return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Startup] App initialized.")
    yield
    logger.info("[Shutdown] App shutting down...")

app = FastAPI(lifespan=lifespan)

origins = [
    "https://agri-annam.vercel.app",
    "https://agrichat.annam.ai",
    "https://be1f1f2ed9db.ngrok-free.app",
    "https://*.ngrok-free.app",
    "https://localhost:3000",
    "https://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

def extract_thinking_process(text: str) -> tuple:
    """Extract thinking process from <think> tags and return (thinking, clean_answer)"""
    if not text:
        return "", text
    
    think_pattern = r'<think>(.*?)</think>'
    matches = re.findall(think_pattern, text, re.DOTALL | re.IGNORECASE)
    thinking = ""
    if matches:
        thinking = matches[0].strip()
        clean_answer = re.sub(think_pattern, '', text, flags=re.DOTALL | re.IGNORECASE).strip()
    else:
        clean_answer = text
    
    return thinking, clean_answer

async def get_answer(question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = None, session_id: str = None, db_config: DatabaseConfig = None) -> Dict:
    """Get answer using rag_main.get_answer to match main.py structure"""
    try:
        question_lower = question.lower().strip()
        greetings = ['hi', 'hello', 'hey', 'namaste', 'good morning', 'good afternoon', 'good evening', 'how are you']
        if any(g in question_lower for g in greetings):
            return {
                'answer': ("Hello! I'm your agricultural assistant specializing in Indian farming. "
                           "I can help you with crop management, soil health, pest control, fertilizers, "
                           "irrigation, farming techniques, and agricultural practices. What would you like to know?"),
                'source': 'Greeting',
                'confidence': 1.0,
                'thinking': ''
            }

        if not is_agricultural_query(question):
            return {
                'answer': ("I'm an agricultural assistant focused on Indian farming. I can only help with agriculture-related questions. "
                           "Please ask about crops, farming practices, soil management, pest control, or other agricultural topics."),
                'source': 'Non-Agricultural',
                'confidence': 1.0,
                'thinking': ''
            }
    except Exception:
        pass

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: rag_main.get_answer(question, conversation_history or [], user_state))

        if isinstance(result, dict):
            answer_text = result.get('answer', '')
            thinking, clean_answer = extract_thinking_process(answer_text)
            
            response = {
                'answer': clean_answer,
                'thinking': thinking,
                'source': result.get('source', 'Unknown'),
                'confidence': result.get('confidence', result.get('similarity', 0.0)),
                'research_data': result.get('research_data', []),
                'reasoning_steps': result.get('reasoning_steps', []),
                'metadata': result.get('metadata', {})
            }
        else:
            thinking, clean_answer = extract_thinking_process(str(result))
            response = {
                'answer': clean_answer,
                'thinking': thinking,
                'source': 'Unknown',
                'confidence': 0.0,
                'research_data': [],
                'reasoning_steps': [],
                'metadata': {}
            }

        return response
    except Exception as e:
        logger.error(f"[RAG] rag_main.get_answer failed: {e}")
        return {
            'answer': "Service temporarily unavailable: internal error.",
            'thinking': '',
            'source': 'Error',
            'confidence': 0.0,
            'research_data': [],
            'reasoning_steps': [],
            'metadata': {}
        }

def resolve_context_question(question: str, context: str) -> str:
    """
    Enhanced context resolution for agricultural follow-up questions
    """
    question_lower = question.lower().strip()
    context_lower = context.lower()
    
    context_patterns = [
        "how do i cure it", "how to cure it", "cure it", "treat it", "how to treat it",
        "what should i do", "how to fix it", "fix it", "prevent it", "how to prevent it",
        "what medicine", "which medicine", "what treatment", "what chemical", "what spray",
        "how much", "when to apply", "when should i", "how often", "what dosage",
        "side effects", "precautions", "what about", "tell me more", "more details",
        "how to grow it", "grow it", "plant it", "sow it", "when to sow", "when to plant",
        "what fertilizer", "which fertilizer", "fertilizer for it", "nutrients needed",
        "watering schedule", "how much water", "irrigation", "harvest time", "when to harvest",
        "what variety", "which variety", "best variety", "seed rate", "spacing",
        "market price", "selling price", "profit", "yield", "production"
    ]
    
    if any(pattern in question_lower for pattern in context_patterns):
        recent_topics = extract_topics_from_context(context)
        
        if recent_topics:
            topic = recent_topics[0]
            
            if "cure" in question_lower or "treat" in question_lower:
                if "disease" in topic or "blight" in topic or "spot" in topic or "rot" in topic:
                    return f"How to cure {topic}?"
                elif "pest" in topic or "insect" in topic or "worm" in topic:
                    return f"How to control {topic}?"
                else:
                    return f"How to treat {topic}?"
            
            elif "prevent" in question_lower:
                return f"How to prevent {topic}?"
            
            elif "medicine" in question_lower or "chemical" in question_lower or "spray" in question_lower:
                return f"What medicines or chemicals to use for {topic}?"
            
            elif "fertilizer" in question_lower or "nutrient" in question_lower:
                return f"What fertilizer is best for {topic}?"
            
            elif "grow" in question_lower or "cultivation" in question_lower:
                return f"How to grow {topic}?"
            
            elif "plant" in question_lower or "sow" in question_lower:
                return f"When and how to plant {topic}?"
            
            elif "water" in question_lower or "irrigation" in question_lower:
                return f"Watering schedule and irrigation for {topic}?"
            
            elif "harvest" in question_lower:
                return f"When to harvest {topic} and harvesting methods?"
            
            elif "variety" in question_lower or "seed" in question_lower:
                return f"Best varieties and seeds for {topic}?"
            
            elif "price" in question_lower or "market" in question_lower or "profit" in question_lower:
                return f"Market price and profitability of {topic}?"
            
            elif "yield" in question_lower or "production" in question_lower:
                return f"How to increase yield and production of {topic}?"
            
            elif "dosage" in question_lower or "how much" in question_lower or "quantity" in question_lower:
                return f"Dosage and application rates for {topic}?"
            
            else:
                return f"{question} for {topic}"
    
    return question

def extract_topics_from_context(context: str) -> List[str]:
    """Enhanced extraction of agricultural topics from conversation context"""
    topics = []
    context_lower = context.lower()
    
    disease_patterns = [
        "late blight", "early blight", "powdery mildew", "downy mildew", 
        "bacterial wilt", "fungal infection", "leaf spot", "root rot", "stem rot",
        "collar rot", "blast", "sheath blight", "rust", "smut", "mosaic virus",
        "yellowing", "wilting", "damping off", "canker", "scab"
    ]
    
    crop_patterns = [
        "potato", "tomato", "wheat", "rice", "cotton", "sugarcane", "maize", "corn",
        "onion", "garlic", "chili", "pepper", "brinjal", "eggplant", "okra", "cucumber",
        "cabbage", "cauliflower", "carrot", "radish", "beans", "peas", "groundnut",
        "soybean", "mustard", "sesame", "sunflower", "mango", "banana", "guava",
        "papaya", "coconut", "tea", "coffee", "spices", "turmeric", "ginger"
    ]
    
    pest_patterns = [
        "aphid", "thrips", "whitefly", "bollworm", "stem borer", "fruit borer",
        "leaf miner", "scale insect", "mealybug", "spider mite", "nematode",
        "caterpillar", "grub", "weevil", "beetle", "locust", "grasshopper"
    ]
    
    problem_patterns = [
        "nutrient deficiency", "nitrogen deficiency", "phosphorus deficiency",
        "potassium deficiency", "iron deficiency", "zinc deficiency", "magnesium deficiency",
        "water stress", "drought stress", "waterlogging", "poor growth", "stunted growth",
        "low yield", "poor germination", "flower drop", "fruit drop"
    ]
    
    practice_patterns = [
        "organic farming", "crop rotation", "intercropping", "mulching", "pruning",
        "grafting", "seed treatment", "soil preparation", "land preparation",
        "transplanting", "direct sowing", "drip irrigation", "sprinkler irrigation"
    ]
    
    for disease in disease_patterns:
        if disease in context_lower:
            for crop in crop_patterns:
                if crop in context_lower:
                    topics.append(f"{disease} in {crop}")
                    break
            else:
                topics.append(disease)
            break
    
    if not topics:
        for pest in pest_patterns:
            if pest in context_lower:
                for crop in crop_patterns:
                    if crop in context_lower:
                        topics.append(f"{pest} in {crop}")
                        break
                else:
                    topics.append(pest)
                break
    
    if not topics:
        for problem in problem_patterns:
            if problem in context_lower:
                for crop in crop_patterns:
                    if crop in context_lower:
                        topics.append(f"{problem} in {crop}")
                        break
                else:
                    topics.append(problem)
                break
    
    if not topics:
        for practice in practice_patterns:
            if practice in context_lower:
                for crop in crop_patterns:
                    if crop in context_lower:
                        topics.append(f"{practice} for {crop}")
                        break
                else:
                    topics.append(practice)
                break
    
    if not topics:
        for crop in crop_patterns:
            if crop in context_lower:
                topics.append(crop)
                break
    
    return topics

async def get_fast_answer(question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = None, db_config: DatabaseConfig = None) -> Dict:
    """
    Fast mode using enhanced FastResponseHandler with source attribution.
    Returns complete result object.
    """
    logger.info(f"[FAST] Processing question with enhanced fast handler: {question}")
    try:
        if db_config:
            database_selection = db_config.get_enabled_databases()
        else:
            database_selection = ["rag", "pops", "llm"]  # default
        
        result = await get_answer(question, conversation_history, user_state, None, db_config)

        try:
            if isinstance(result, dict) and 'ragas_score' in result:
                logger.info(f"[RAGAS] Fast mode produced ragas_score={result['ragas_score']}")
        except Exception:
            pass

        try:
            if isinstance(result, dict) and result.get('source') == 'Fallback LLM':
                candidate_paths = [
                    os.path.abspath(os.path.join(current_dir, '..', 'fallback_queries.csv')),
                    os.path.abspath(os.path.join(current_dir, 'fallback_queries.csv')),
                    '/tmp/fallback_queries.csv'
                ]

                timestamp = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
                question_text = question.replace('\n', ' ').replace(',', ' ').strip()
                answer_text = result.get('answer', '').replace('\n', ' ').replace(',', ' ').strip()
                fallback_reason = 'Database search failed - using LLM fallback (Source: Fallback LLM)'

                wrote = False
                header = 'timestamp,question,answer,fallback_reason\n'
                for csv_path in candidate_paths:
                    try:
                        file_exists = os.path.exists(csv_path)
                        parent_dir = os.path.dirname(csv_path)
                        if parent_dir and not os.path.exists(parent_dir):
                            os.makedirs(parent_dir, exist_ok=True)

                        with open(csv_path, 'a', encoding='utf-8') as fh:
                            if not file_exists:
                                fh.write(header)
                            fh.write(f"{timestamp},{question_text},{answer_text},{fallback_reason}\n")
                        logger.info(f"[FALLBACK LOG] Written fallback query to {csv_path}")
                        wrote = True
                        break
                    except Exception:
                        continue

                if not wrote:
                    logger.warning('[FALLBACK LOG] All candidate paths failed for writing fallback_queries.csv')
        except Exception as e:
            logger.warning(f"[FALLBACK LOG] Failed to write fallback query to CSV: {e}")

        return result
    except Exception as e:
        logger.error(f"[FAST] Error in fast mode: {e}")
        logger.info("[FAST] Falling back to full RAG pipeline")
        try:
            return query_handler.get_answer_with_source(
                question=question,
                conversation_history=conversation_history,
                user_state=user_state,
                database_selection=(db_config.get_enabled_databases() if db_config else ["rag", "pops", "llm"])
            )
        except Exception:
            return { 'answer': "I couldn't process your question right now.", 'source': 'Error' }


def preprocess_question(question: str) -> str:
    """
    Preprocess question text for similarity comparison
    """
    question = question.lower()
    question = re.sub(r'[^a-zA-Z0-9\s]', '', question)
    question = re.sub(r'\s+', ' ', question).strip()
    return question

def get_question_recommendations(user_question: str, user_state: str = None, limit: int = 4) -> List[Dict]:
    """
    Get question recommendations based on similarity to user's question and state
    Only returns recommendations for agriculture-related questions
    Optimized for performance with caching and reduced database queries
    
    Args:
        user_question: The user's current question
        user_state: User's state for state-specific recommendations
        limit: Number of recommendations to return (default 4)
    
    Returns:
        List of recommended questions with their details (empty if not agriculture-related)
    """
    try:
        cache_key = hashlib.md5(f"{user_question.lower()}_{user_state}_{limit}".encode()).hexdigest()
        current_time = time.time()
        
        if cache_key in recommendations_cache:
            cached_data, timestamp = recommendations_cache[cache_key]
            if current_time - timestamp < CACHE_EXPIRY_SECONDS:
                logger.info(f"[CACHE] Returning cached recommendations for: {user_question[:50]}...")
                return cached_data
            else:
                del recommendations_cache[cache_key]
        
        question_category = query_handler.classify_query(user_question)
        logger.info(f"[REC] Question classified as: {question_category} for: {user_question[:50]}...")
        if question_category != "AGRICULTURE":
            logger.info(f"[REC] Question classified as {question_category}, skipping recommendations: {user_question}")
            recommendations_cache[cache_key] = ([], current_time)
            return []
        
        pipeline = [
            {"$unwind": "$messages"},
            {"$match": {
                "state": {"$exists": True},
                "messages.question": {"$exists": True, "$ne": "", "$type": "string"},
                "messages.answer": {"$exists": True, "$type": "string"}
            }},
            {"$group": {
                "_id": {
                    "question": "$messages.question",
                    "state": "$state"
                },
                "count": {"$sum": 1},
                "sample_answer": {"$first": "$messages.answer"},  # Keep full answer, truncate in Python
                "sample_source": {"$first": "$messages.source"}
            }},
            {"$match": {"count": {"$gte": 1}}},
            {"$project": {
                "question": "$_id.question",
                "state": "$_id.state",
                "count": 1,
                "sample_answer": 1,
                "sample_source": 1
            }},
            {"$sort": {"count": -1}},
            {"$limit": 100} 
        ]
        
        try:
            questions_data = list(sessions_collection.aggregate(pipeline))
            logger.info(f"[REC] Found {len(questions_data)} questions in database for recommendations")
        except Exception as db_error:
            logger.error(f"[REC] Database aggregation failed: {db_error}")
            try:
                logger.info("[REC] Attempting fallback simple query...")
                simple_docs = list(sessions_collection.find(
                    {"state": {"$exists": True}, "messages.question": {"$exists": True}},
                    {"messages.question": 1, "state": 1}
                ).limit(50))
                
                questions_data = []
                for doc in simple_docs:
                    for msg in doc.get("messages", []):
                        if msg.get("question"):
                            questions_data.append({
                                "question": msg["question"],
                                "state": doc.get("state", "unknown"),
                                "count": 1,
                                "sample_answer": "Answer available on request..."
                            })
                
                logger.info(f"[REC] Fallback query found {len(questions_data)} questions")
            except Exception as fallback_error:
                logger.error(f"[REC] Fallback query also failed: {fallback_error}")
                recommendations_cache[cache_key] = ([], current_time)
                return []
        
        if not questions_data:
            logger.info("[REC] No questions found in database for recommendations")
            recommendations_cache[cache_key] = ([], current_time)
            return []
        
        processed_user_question = preprocess_question(user_question)

        candidates = []

        for item in questions_data:
            question = item['question']
            processed_question = preprocess_question(question)

            if processed_question == processed_user_question:
                continue

            candidates.append({
                'original_question': question,
                'processed_question': processed_question,
                'state': item.get('state', 'unknown'),
                'count': item['count'],
                'sample_answer': item['sample_answer'][:200] + "..." if len(item['sample_answer']) > 200 else item['sample_answer'],
                'source': item.get('source', 'unknown') if isinstance(item, dict) else 'unknown'
            })

            if len(candidates) >= 200:
                break

        if not candidates:
            logger.info("[REC] No suitable questions found for recommendations after filtering")
            recommendations_cache[cache_key] = ([], current_time)
            return []

        processed_questions = [c['original_question'] for c in candidates]
        try:
            tfidf_prefilter = TfidfVectorizer(max_features=300, stop_words='english', ngram_range=(1,2))
            tfidf_matrix_pref = tfidf_prefilter.fit_transform([user_question] + processed_questions)
            pref_similarities = cosine_similarity(tfidf_matrix_pref[0:1], tfidf_matrix_pref[1:]).flatten()
            top_n = min(50, len(candidates))
            top_idx = pref_similarities.argsort()[::-1][:top_n]
            prefiltered = [candidates[i] for i in top_idx]
            logger.info(f"[REC] Prefiltered {len(prefiltered)} candidates for embedding from {len(candidates)} total")
        except Exception as e:
            logger.warning(f"[REC] Prefilter TF-IDF failed, falling back to full candidate set: {e}")
            prefiltered = candidates

        try:
            user_emb = np.array(local_embeddings.embed_query(user_question))
        except Exception as e:
            logger.error(f"[REC] Failed to compute user embedding: {e}")
            recommendations_cache[cache_key] = ([], current_time)
            return []

        if not hasattr(get_question_recommendations, "_emb_cache"):
            get_question_recommendations._emb_cache = {}

        for cand in prefiltered:
            key = cand['processed_question']
            if key in get_question_recommendations._emb_cache:
                cand_emb = get_question_recommendations._emb_cache[key]
            else:
                try:
                    cand_emb = np.array(local_embeddings.embed_query(cand['original_question']))
                    get_question_recommendations._emb_cache[key] = cand_emb
                except Exception as e:
                    logger.warning(f"[REC] Failed to embed candidate question '{cand['original_question'][:50]}': {e}")
                    cand_emb = None
            cand['embedding'] = cand_emb

        def cosine(a, b):
            try:
                return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            except Exception:
                return 0.0

        ranked = []
        for cand in prefiltered:
            if cand.get('embedding') is None:
                continue
            sim = cosine(user_emb, cand['embedding'])
            if sim >= 0.90 or sim < 0.15:
                continue
            ranked.append({
                'question': cand['original_question'],
                'processed_question': cand['processed_question'],
                'state': cand['state'],
                'similarity_score': sim,
                'popularity': cand['count'],
                'sample_answer': cand['sample_answer'],
                'source': cand.get('sample_source', cand.get('source', 'unknown'))
            })

        unique_by_processed = {}
        for r in ranked:
            key = r.get('processed_question') or preprocess_question(r.get('question', ''))
            existing = unique_by_processed.get(key)
            if not existing:
                unique_by_processed[key] = r
            else:
                if r['similarity_score'] > existing['similarity_score'] or (
                    r['similarity_score'] == existing['similarity_score'] and r.get('popularity', 0) > existing.get('popularity', 0)
                ):
                    unique_by_processed[key] = r

        ranked = list(unique_by_processed.values())

        if not ranked:
            logger.info("[REC] No candidates passed semantic filtering")
            recommendations_cache[cache_key] = ([], current_time)
            return []

        rag_candidates = [r for r in ranked if 'rag' in (r.get('source') or '').lower() or 'golden' in (r.get('source') or '').lower()]
        other_candidates = [r for r in ranked if r not in rag_candidates]

        rag_candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
        other_candidates.sort(key=lambda x: x['similarity_score'], reverse=True)

        final_recs = []
        for r in rag_candidates:
            final_recs.append(r)
            if len(final_recs) >= limit:
                break
        if len(final_recs) < limit:
            for r in other_candidates:
                final_recs.append(r)
                if len(final_recs) >= limit:
                    break

        if user_state:
            for rec in final_recs:
                rec['boosted_score'] = rec['similarity_score'] + (0.12 if rec['state'].lower() == user_state.lower() else 0)
            final_recs.sort(key=lambda x: x['boosted_score'], reverse=True)
        else:
            final_recs.sort(key=lambda x: x['similarity_score'], reverse=True)

        def canonicalize(q: str) -> str:
            q = (q or "").lower()
            q = re.sub(r"[^a-z0-9\s]", "", q)
            q = re.sub(r"\s+", " ", q).strip()
            return q

        top_recommendations = []
        seen = set()
        for rec in final_recs:
            canon = canonicalize(rec.get('question', ''))
            if canon in seen:
                continue
            seen.add(canon)
            top_recommendations.append(rec)
            if len(top_recommendations) >= limit:
                break

        logger.info(f"[REC] Generated {len(top_recommendations)} recommendations from {len(prefiltered)} candidates for question: {user_question[:50]}...")
        if top_recommendations:
            logger.info(f"[REC] Top recommendation similarity scores: {[rec['similarity_score'] for rec in top_recommendations]}")

        recommendations_cache[cache_key] = (top_recommendations, current_time)

        if len(recommendations_cache) > 100:
            sorted_cache = sorted(recommendations_cache.items(), key=lambda x: x[1][1])
            for old_key, _ in sorted_cache[:50]:  # Remove 50 oldest entries
                del recommendations_cache[old_key]

        return top_recommendations
        
    except Exception as e:
        logger.error(f"[REC] get_question_recommendations failed: {e}")
        try:
            recommendations_cache[cache_key] = ([], current_time)
        except Exception:
            pass
        return []


async def get_question_recommendations_async(user_question: str, user_state: str = None, limit: int = 4):
    """Async wrapper to run `get_question_recommendations` in a thread pool for `await` usage."""
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, get_question_recommendations, user_question, user_state, limit)
    except Exception as e:
        logger.error(f"[REC] Async recommendations failed: {e}")
        return []
def ensure_database_indexes():
    try:
        sessions_collection.create_index([("state", 1)])
        sessions_collection.create_index([("state", 1), ("messages.question", 1)])
        sessions_collection.create_index([("timestamp", -1)])
        logger.info("[PERF] Database indexes ensured for optimal performance")
    except Exception as e:
        logger.error(f"[PERF] Failed to create indexes: {e}")

ensure_database_indexes()

def clean_session(s):
    s["_id"] = str(s["_id"])
    return s

@app.get("/")
async def root():
    return {"message": "AgriChat backend is running."}

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "AgriChat backend is running."}

@app.options("/{full_path:path}")
async def options_handler(request: Request):
    origin = request.headers.get("origin")
    allowed_origin = origin if origin in origins else origins[0]
    
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": allowed_origin,
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "86400"
        }
    )

@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """Authenticate user with CSV-based credentials"""
    try:
        auth_result = authenticate_user(request.username, request.password)
        
        if auth_result['authenticated']:
            logger.info(f"[AUTH] Successful login for user: {request.username}")
            return AuthResponse(
                authenticated=True,
                username=auth_result['username'],
                role=auth_result['role'],
                full_name=auth_result['full_name'],
                message="Login successful"
            )
        else:
            logger.warning(f"[AUTH] Failed login attempt for user: {request.username}")
            return AuthResponse(
                authenticated=False,
                message="Invalid username or password"
            )
    
    except Exception as e:
        logger.error(f"[AUTH] Login error: {e}")
        return AuthResponse(
            authenticated=False,
            message="Authentication service unavailable"
        )

@app.get("/api/sessions")
async def list_sessions(request: Request):
    device_id = request.query_params.get("device_id")
    if not device_id:
        return JSONResponse(status_code=400, content={"error": "Device ID is required"})

    sessions = list(
        sessions_collection.find({"device_id": device_id},
            {"session_id": 1, "timestamp": 1, "crop": 1, "state": 1, "status": 1, "has_unread": 1, "messages": {"$slice": 1}})
        .sort("timestamp", -1).limit(20)
    )
    return {"sessions": [clean_session(s) for s in sessions]}

@app.post("/api/query")
async def new_session(request: QueryRequest):
    api_start_time = time.time()
    session_id = str(uuid4())
    logger.info(f"[TIMING] API endpoint started for question: {request.question[:50]}...")
    
    db_config = None
    if request.database_config:
        db_config = DatabaseConfig(**request.database_config)
        logger.info(f"[DB_CONFIG] Received database configuration: {db_config.get_enabled_databases()}")
    
    async def process_answer():
        try:
            answer_processing_start = time.time()
            logger.info(f"[DEBUG] Starting get_answer call...")
            
            memory = get_session_memory(session_id)
            logger.info(f"[LANGCHAIN] Initialized new memory for session {session_id}")

            memory.chat_memory.add_user_message(request.question)

            loop = asyncio.get_event_loop()
            try:
                raw_answer = await loop.run_in_executor(
                    None,
                    rag_main.get_answer,
                    request.question,
                    [],
                    request.state
                )
            except Exception as e:
                logger.error(f"[PIPELINE] rag_main.get_answer failed: {e}")
                raise
            
            if isinstance(raw_answer, dict) and raw_answer.get('answer'):
                memory.chat_memory.add_ai_message(raw_answer['answer'])
                logger.info(f"[LANGCHAIN] Added AI response to new session memory {session_id}")
            else:
                memory.chat_memory.add_ai_message(str(raw_answer))
                logger.info(f"[LANGCHAIN] Added AI response (string) to new session memory {session_id}")
            
            answer_processing_time = time.time() - answer_processing_start
            logger.info(f"[TIMING] Answer processing took: {answer_processing_time:.3f}s")
            logger.info(f"[DEBUG] Raw answer received: {str(raw_answer.get('answer', ''))[:100]}...")
            
            return raw_answer
            
        except Exception as e:
            logger.error(f"[DEBUG] Error in get_answer: {e}")
            logger.error(f"[DEBUG] Exception type: {type(e)}")
            raise e
    
    tasks = [process_answer()]
    
    # if not DISABLE_RECOMMENDATIONS and PARALLEL_RECOMMENDATIONS:
    #     tasks.append(get_question_recommendations_async(request.question, request.state, 4))
    
    try:
        if len(tasks) > 1:
            parallel_start = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            parallel_time = time.time() - parallel_start
            logger.info(f"[TIMING] Parallel processing took: {parallel_time:.3f}s")
            
            answer_result = results[0]
            if isinstance(answer_result, Exception):
                raise answer_result
                
            recommendations = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else []
            if isinstance(results[1], Exception):
                logger.error(f"Parallel recommendations failed: {results[1]}")
                recommendations = []
        else:
            answer_result = await tasks[0]
            recommendations = []
    except Exception as e:
        logger.error(f"[DEBUG] Error in parallel processing: {e}")
        return JSONResponse(status_code=500, content={"error": "LLM processing failed."})

    answer_only = answer_result.get('answer', '') if isinstance(answer_result, dict) else str(answer_result)

    markdown_processing_start = time.time()
    html_answer = markdown.markdown(answer_only, extensions=["extra", "nl2br"])
    markdown_processing_time = time.time() - markdown_processing_start
    logger.info(f"[TIMING] Markdown processing took: {markdown_processing_time:.3f}s")

    session_creation_start = time.time()
    
    message = {
        "question": request.question,
        "thinking": answer_result.get('thinking', '') if isinstance(answer_result, dict) else '',
        "final_answer": html_answer,
        "answer": html_answer,
        "rating": None
    }
    
    if isinstance(answer_result, dict):
        if 'research_data' in answer_result:
            message['research_data'] = answer_result['research_data']
        if 'source' in answer_result:
            message['source'] = answer_result['source']
        if 'ragas_score' in answer_result:
            message['ragas_score'] = answer_result['ragas_score']
        sources = []
        if answer_result.get('research_data'):
            for d in answer_result.get('research_data'):
                src = d.get('source') or d.get('collection_type') or 'unknown'
                preview = d.get('content_preview') or (d.get('page_content') or '')[:300]
                sources.append({"source": src, "preview": preview, "metadata": d.get('metadata', {})})
        elif answer_result.get('source'):
            sources.append({"source": answer_result.get('source'), "preview": ''})
        if sources:
            message['sources'] = sources
    
    session = {
        "session_id": session_id,
        "timestamp": datetime.now(IST).isoformat(),
        "messages": [message],
        "crop": "unknown",
        "state": request.state,
        "status": "active",
        "language": request.language,
        "has_unread": True,
        "device_id": request.device_id
    }

    sessions_collection.insert_one(session)
    session.pop("_id", None)
    session_creation_time = time.time() - session_creation_start
    logger.info(f"[TIMING] Session creation took: {session_creation_time:.3f}s")
    
    recommendations_start = time.time()
    # if not DISABLE_RECOMMENDATIONS:
    #     if not PARALLEL_RECOMMENDATIONS:
    #         try:
    #             recommendations = get_question_recommendations(
    #                 user_question=request.question,
    #                 user_state=request.state,
    #                 limit=4
    #             )
    #             logger.info(f"Added {len(recommendations)} recommendations to session response (sequential)")
    #         except Exception as e:
    #             logger.error(f"Failed to get sequential recommendations: {e}")
    #             recommendations = []
    #     else:
    #         logger.info(f"Added {len(recommendations)} recommendations to session response (parallel)")
    #     
    #     session["recommendations"] = recommendations
    # else:
    session["recommendations"] = []
    #     logger.info(f"[PERFORMANCE] Recommendations disabled for speed optimization")
    # recommendations_time = time.time() - recommendations_start
    # logger.info(f"[TIMING] Recommendations handling took: {recommendations_time:.3f}s")
    
    total_api_time = time.time() - api_start_time
    logger.info(f"[TIMING] TOTAL API processing took: {total_api_time:.3f}s")
    
    return {"session": session}

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    session = sessions_collection.find_one({"session_id": session_id})
    if session:
        session["_id"] = str(session["_id"])
        sessions_collection.update_one({"session_id": session_id}, {"$set": {"has_unread": False}})
        session["has_unread"] = False
    return {"session": session}

@app.post("/api/session/{session_id}/query")
async def continue_session(session_id: str, request: SessionQueryRequest):
    session = sessions_collection.find_one({"session_id": session_id})
    if not session or session.get("status") == "archived" or session.get("device_id") != request.device_id:
        return JSONResponse(status_code=403, content={"error": "Session is archived, missing or unauthorized"})

    db_config = None
    if request.database_config:
        db_config = DatabaseConfig(**request.database_config)
        logger.info(f"[DB_CONFIG] Received database configuration for session: {db_config.get_enabled_databases()}")

    async def process_session_answer():
        try:
            memory = get_session_memory(session_id)
            logger.info(f"[LANGCHAIN] Retrieved memory for session {session_id}: {len(memory.chat_memory.messages)} messages")
            
            conversation_history = convert_langchain_memory_to_history(memory)
            logger.info(f"[LANGCHAIN] Using {len(conversation_history)} conversation pairs from memory")
            
            current_state = request.state or session.get("state", "unknown")
            
            memory.chat_memory.add_user_message(request.question)
            loop = asyncio.get_event_loop()
            try:
                raw_answer = await loop.run_in_executor(None, rag_main.get_answer, request.question, conversation_history, current_state)
            except Exception as e:
                logger.error(f"[PIPELINE] rag_main.get_answer failed for session: {e}")
                raise

            if isinstance(raw_answer, dict) and raw_answer.get('answer'):
                memory.chat_memory.add_ai_message(raw_answer['answer'])
                logger.info(f"[LANGCHAIN] Added AI response to memory for session {session_id}")
            else:
                memory.chat_memory.add_ai_message(str(raw_answer))
                logger.info(f"[LANGCHAIN] Added AI response (string) to memory for session {session_id}")

            return raw_answer
            
        except Exception as e:
            logger.error(f"[DEBUG] Error in get_answer: {e}")
            raise e
    
    tasks = [process_session_answer()]
    
    current_state = request.state or session.get("state", "unknown")
    # if not DISABLE_RECOMMENDATIONS and PARALLEL_RECOMMENDATIONS:
    #     logger.info("[PARALLEL] Starting session recommendations in parallel with answer processing")
    # tasks.append(get_question_recommendations_async(request.question, current_state, 4))
    
    try:
        if len(tasks) > 1:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            answer_result = results[0]
            if isinstance(answer_result, Exception):
                raise answer_result
                
            recommendations = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else []
            if isinstance(results[1], Exception):
                logger.error(f"Parallel session recommendations failed: {results[1]}")
                recommendations = []
        else:
            answer_result = await tasks[0]
            recommendations = []
            
    except Exception as e:
        logger.error(f"[DEBUG] Error in parallel session processing: {e}")
        return JSONResponse(status_code=500, content={"error": "LLM processing failed."})

    answer_only = answer_result.get('answer', '') if isinstance(answer_result, dict) else str(answer_result)
    html_answer = markdown.markdown(answer_only, extensions=["extra", "nl2br"])

    new_message = {
        "question": request.question,
        "thinking": answer_result.get('thinking', '') if isinstance(answer_result, dict) else '',
        "final_answer": html_answer,
        "answer": html_answer,
        "rating": None
    }
    
    if isinstance(answer_result, dict):
        if 'research_data' in answer_result:
            new_message['research_data'] = answer_result['research_data']
        if 'source' in answer_result:
            new_message['source'] = answer_result['source']
        if 'ragas_score' in answer_result:
            new_message['ragas_score'] = answer_result['ragas_score']

        sources = []
        if answer_result.get('research_data'):
            for d in answer_result.get('research_data'):
                src = d.get('source') or d.get('collection_type') or 'unknown'
                preview = d.get('content_preview') or (d.get('page_content') or '')[:300]
                sources.append({"source": src, "preview": preview, "metadata": d.get('metadata', {})})
        elif answer_result.get('source'):
            sources.append({"source": answer_result.get('source'), "preview": ''})
        if sources:
            new_message['sources'] = sources

    crop = session.get("crop", "unknown")
    current_state = request.state or session.get("state", "unknown")

    sessions_collection.update_one(
        {"session_id": session_id},
        {
            "$push": {"messages": new_message},
            "$set": {
                "has_unread": True,
                "crop": crop,
                "state": current_state,
                "timestamp": datetime.now(IST).isoformat()
            },
        }
    )
    updated = sessions_collection.find_one({"session_id": session_id})
    if updated:
        updated.pop("_id", None)
        
        # if not DISABLE_RECOMMENDATIONS:
        #     if not PARALLEL_RECOMMENDATIONS:
        #         try:
        #             recommendations = get_question_recommendations(
        #                 user_question=request.question,
        #                 user_state=current_state,
        #                 limit=4
        #             )
        #             updated["recommendations"] = recommendations
        #             logger.info(f"Added {len(recommendations)} session recommendations (sequential)")
        #         except Exception as e:
        #             logger.error(f"Failed to get sequential session recommendations: {e}")
        #             updated["recommendations"] = []
        #     else:
        #         updated["recommendations"] = recommendations
        #         logger.info(f"Added {len(recommendations)} session recommendations (parallel)")
        updated["recommendations"] = []
        # else:
        #     updated["recommendations"] = []
        #     logger.info(f"[PERFORMANCE] Session recommendations disabled for speed optimization")
    
    return {"session": updated}

async def enhance_answer_with_context_questions(question: str, answer: str, user_state: str, thinking: str) -> str:
    """
    Enhance answer with contextual questions when more context is needed
    """
    generic_indicators = [
        "depends on", "varies", "generally", "typically", "usually", 
        "can vary", "it depends", "specific to", "without knowing",
        "need more information", "requires specific", "can range"
    ]
    
    context_indicators = [
        "need to know", "depends on the", "varies by", "specific variety",
        "location matters", "soil type", "climate", "season", "region"
    ]
    
    answer_lower = answer.lower()
    thinking_lower = thinking.lower() if thinking else ""
    
    needs_context = (
        any(indicator in answer_lower for indicator in generic_indicators) or
        any(indicator in thinking_lower for indicator in context_indicators) or
        len(answer.split()) < 30  # Very short answers might be generic
    )
    
    if needs_context:
        context_questions = generate_context_questions(question, user_state)
        
        if context_questions:
            enhanced_answer = f"{answer}\n\n**To provide more specific guidance, I'd like to know:**\n"
            for i, q in enumerate(context_questions, 1):
                enhanced_answer += f"{i}. {q}\n"
            enhanced_answer += "\nFeel free to share any of these details for more targeted advice!"
            return enhanced_answer
    
    return answer

def generate_context_questions(question: str, user_state: str) -> List[str]:
    """
    Generate relevant context questions based on the agricultural topic
    """
    question_lower = question.lower()
    questions = []
    
    if any(crop in question_lower for crop in ['rice', 'wheat', 'maize', 'sugarcane', 'cotton', 'soybean']):
        questions.append("What variety of the crop are you planning to grow?")
        questions.append("What's your soil type (clay, sandy, loamy)?")
        
    if any(word in question_lower for word in ['sowing', 'planting', 'harvest', 'when']):
        questions.append("Which season/month are you planning this activity?")
        if not user_state or user_state == "unknown":
            questions.append("Which state/region are you located in?")
            
    if any(word in question_lower for word in ['fertilizer', 'nutrient', 'manure', 'compost']):
        questions.append("What's your current soil pH and nutrient status?")
        questions.append("What's the size of your field (in acres/hectares)?")
        
    if any(word in question_lower for word in ['disease', 'pest', 'insect', 'fungus', 'virus']):
        questions.append("Can you describe the symptoms you're seeing?")
        questions.append("What stage is your crop currently in?")
        
    if any(word in question_lower for word in ['water', 'irrigation', 'drought', 'flood']):
        questions.append("What's your current irrigation method?")
        questions.append("What's the rainfall pattern in your area?")
        
    return questions[:4]

@app.post("/api/query/thinking-stream")
async def thinking_stream_query(request: QueryRequest):
    """
    Streaming endpoint that shows real-time thinking process followed by answer
    """
    async def generate_stream():
        import json
        import asyncio
        
        session_id = str(uuid4())
        logger.info(f"[THINKING_STREAM] Starting stream for question: {request.question[:50]}...")
        
        try:
            yield f"data: {json.dumps({'type': 'session_start', 'session_id': session_id})}\n\n"
            
            if USE_FAST_MODE and fast_handler:
                logger.info("[THINKING_STREAM] Using fast mode with fast_handler")
                yield f"data: {json.dumps({'type': 'thinking_start'})}\n\n"
                
                thinking_content = ''
                if fast_handler.reasoner_llm:
                    logger.info("[THINKING_STREAM] Reasoner LLM available, starting thinking generation")
                    try:
                        reasoner_prompt = (
                            f"You are an expert agricultural assistant. Analyze this question step-by-step.\n\n"
                            f"Question: {request.question}\n"
                            f"User State: {request.state if request.state else 'Not specified'}\n\n"
                            f"Think about:\n"
                            f"1. Is this question specific enough to provide a detailed answer?\n"
                            f"2. What context is missing (location, crop variety, soil type, season, etc.)?\n"
                            f"3. Can I provide a helpful general answer while identifying what additional context would be valuable?\n\n"
                            f"Keep your thinking concise and focused on what information is needed."
                        )
                        
                        current_text = ''
                        for ev in fast_handler.reasoner_llm.stream_generate(reasoner_prompt, model=fast_handler.reasoner_llm.model_name, temperature=0.1):
                            if ev.get('type') == 'token':
                                token = ev.get('text', '')
                                current_text += token
                                try:
                                    json_data = json.dumps({'type': 'thinking_token', 'token': token, 'text': current_text}, ensure_ascii=False)
                                    yield f"data: {json_data}\n\n"
                                except (TypeError, ValueError) as json_error:
                                    logger.error(f"[THINKING_STREAM] JSON serialization error: {json_error}")
                                    sanitized_text = current_text.encode('utf-8', errors='ignore').decode('utf-8')
                                    json_data = json.dumps({'type': 'thinking_token', 'token': token, 'text': sanitized_text}, ensure_ascii=False)
                                    yield f"data: {json_data}\n\n"
                        
                        thinking_content = current_text.strip()
                        
                    except Exception as e:
                        logger.error(f"[THINKING_STREAM] Reasoner streaming failed: {e}")
                        thinking_content = ''
                else:
                    logger.warning("[THINKING_STREAM] No reasoner LLM available, skipping thinking generation")
                    thinking_content = ''
                
                logger.info(f"[THINKING_STREAM] Thinking complete, content length: {len(thinking_content)}")
                try:
                    json_data = json.dumps({'type': 'thinking_complete', 'thinking': thinking_content}, ensure_ascii=False)
                    yield f"data: {json_data}\n\n"
                except (TypeError, ValueError) as json_error:
                    logger.error(f"[THINKING_STREAM] JSON serialization error for thinking_complete: {json_error}")
                    sanitized_thinking = thinking_content.encode('utf-8', errors='ignore').decode('utf-8')
                    json_data = json.dumps({'type': 'thinking_complete', 'thinking': sanitized_thinking}, ensure_ascii=False)
                    yield f"data: {json_data}\n\n"
                
                logger.info("[THINKING_STREAM] Starting answer generation with consistent pipeline")
                yield f"data: {json.dumps({'type': 'answer_start'})}\n\n"
                
                try:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: fast_handler.get_answer(request.question, [], request.state)
                    )
                    logger.info(f"[THINKING_STREAM] Got consistent answer result type: {type(result)}")
                    logger.info(f"[THINKING_STREAM] Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                    logger.info(f"[THINKING_STREAM] Answer content length: {len(result.get('answer', '')) if isinstance(result, dict) else 'N/A'}")
                    logger.info(f"[THINKING_STREAM] Answer preview: {str(result.get('answer', ''))[:100] if isinstance(result, dict) else 'N/A'}...")
                except Exception as e:
                    logger.error(f"[THINKING_STREAM] Answer generation failed: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to generate answer'})}\n\n"
                    return
                
                if isinstance(result, dict):
                    answer_text = result.get('answer', '')
                    source = result.get('source', '')
                    logger.info(f"[THINKING_STREAM] Extracted answer_text length: {len(answer_text)}")
                    logger.info(f"[THINKING_STREAM] Extracted answer_text preview: {answer_text[:100]}...")
                    logger.info(f"[THINKING_STREAM] Source detected: '{source}' (type: {type(source)})")
                    
                    if source == 'RAG Database (Golden)':
                        logger.info("[THINKING_STREAM] Golden Database answer - skipping enhancement to preserve exact content")
                        enhanced_answer = answer_text  # Use exact Golden Database content
                    else:
                        logger.info(f"[THINKING_STREAM] Not Golden Database (source: '{source}') - applying enhancement")
                        enhanced_answer = await enhance_answer_with_context_questions(
                            request.question, answer_text, request.state, thinking_content
                        )
                    
                    try:
                        json_data = json.dumps({'type': 'answer', 'answer': enhanced_answer, 'source': result.get('source', ''), 'confidence': result.get('confidence', 0.0)}, ensure_ascii=False)
                        yield f"data: {json_data}\n\n"
                    except (TypeError, ValueError) as json_error:
                        logger.error(f"[THINKING_STREAM] JSON serialization error for answer: {json_error}")
                        sanitized_answer = enhanced_answer.encode('utf-8', errors='ignore').decode('utf-8')
                        json_data = json.dumps({'type': 'answer', 'answer': sanitized_answer, 'source': result.get('source', ''), 'confidence': result.get('confidence', 0.0)}, ensure_ascii=False)
                        yield f"data: {json_data}\n\n"
                    
                    # Debug metadata processing
                    logger.info(f"[THINKING_STREAM] Checking metadata - Source: '{source}', Has metadata: {bool(result.get('metadata'))}")
                    if result.get('metadata'):
                        logger.info(f"[THINKING_STREAM] Available metadata keys: {list(result.get('metadata', {}).keys())}")
                    
                    if source in ['RAG Database', 'RAG Database (Golden)', 'PoPs Database'] and result.get('metadata'):
                        logger.info(f"[THINKING_STREAM] Processing metadata for source: {source}")
                        try:
                            metadata = result.get('metadata', {})
                            metadata_tags = []
                            
                            if 'file_path' in metadata:
                                metadata_tags.append(f"**File:** {metadata['file_path']}")
                            
                            if 'section' in metadata:
                                metadata_tags.append(f"**Section:** {metadata['section']}")
                            
                            if 'document_id' in metadata:
                                metadata_tags.append(f"**Document ID:** {metadata['document_id']}")
                            
                            if 'content_type' in metadata:
                                metadata_tags.append(f"**Content Type:** {metadata['content_type']}")
                            
                            if 'category' in metadata:
                                metadata_tags.append(f"**Category:** {metadata['category']}")
                            
                            if 'subcategory' in metadata:
                                metadata_tags.append(f"**Subcategory:** {metadata['subcategory']}")
                            
                            if result.get('similarity') or result.get('cosine_similarity'):
                                similarity = result.get('similarity') or result.get('cosine_similarity')
                                metadata_tags.append(f"**Similarity Score:** {similarity:.3f}")
                            
                            if metadata_tags:
                                metadata_content = "\n".join(metadata_tags)
                                logger.info(f"[THINKING_STREAM] Sending metadata_tags with {len(metadata_tags)} items")
                                metadata_json = json.dumps({'type': 'metadata_tags', 'content': metadata_content}, ensure_ascii=False)
                                yield f"data: {metadata_json}\n\n"
                            else:
                                logger.info(f"[THINKING_STREAM] No metadata tags generated from available metadata")
                                
                        except Exception as metadata_error:
                            logger.error(f"[THINKING_STREAM] Error processing metadata: {metadata_error}")
                    
                    html_answer = markdown.markdown(enhanced_answer, extensions=["extra", "nl2br"])
                    
                    message = {
                        "question": request.question,
                        "thinking": thinking_content,
                        "final_answer": html_answer,
                        "answer": html_answer,
                        "rating": None
                    }
                    
                    if 'research_data' in result:
                        message['research_data'] = result['research_data']
                    if 'source' in result:
                        message['source'] = result['source']
                    
                    session = {
                        "session_id": session_id,
                        "timestamp": datetime.now(IST).isoformat(),
                        "messages": [message],
                        "crop": "unknown",
                        "state": request.state,
                        "status": "active",
                        "language": request.language,
                        "has_unread": True,
                        "device_id": request.device_id
                    }
                    
                    sessions_collection.insert_one(session)
                    session.pop("_id", None)
                    
                    yield f"data: {json.dumps({'type': 'session_complete', 'session': session})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to get response'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Fast mode not available'})}\n\n"
                
        except Exception as e:
            logger.error(f"[STREAM] Error in thinking stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Processing failed'})}\n\n"
        
        yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/api/toggle-status/{session_id}/{status}")
async def toggle_status(session_id: str, status: str):
    new_status = "archived" if status == "active" else "active"
    sessions_collection.update_one(
        {"session_id": session_id},
        {"$set": {"status": new_status}}
    )
    return {"status": new_status}

@app.get("/api/export/csv/{session_id}")
async def export_csv(session_id: str):
    session = sessions_collection.find_one({"session_id": session_id})
    if not session:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Question", "Answer", "Rating", "Timestamp"])
    for i, msg in enumerate(session.get("messages", [])):
        q = msg["question"]
        a = BeautifulSoup(msg["answer"], "html.parser").get_text()
        r = msg.get("rating", "")
        t = parser.isoparse(session["timestamp"]).astimezone(IST).strftime("%Y-%m-%d %H:%M:%S") if i == 0 else ""
        writer.writerow([q, a, r, t])

    output.seek(0)
    filename = f"session_{session_id}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

@app.delete("/api/delete-session/{session_id}")
async def delete_session(session_id: str):
    result = sessions_collection.delete_one({"session_id": session_id})
    if result.deleted_count == 0:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    return {"message": "Session deleted successfully"}

@app.post("/api/session/{session_id}/rate")
async def rate_answer(session_id: str, question_index: int = Form(...), rating: str = Form(...)):
    if rating not in ["up", "down"]:
        return JSONResponse(status_code=400, content={"error": "Invalid rating"})

    session = sessions_collection.find_one({"session_id": session_id})
    if not session:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    if question_index < 0 or question_index >= len(session["messages"]):
        return JSONResponse(status_code=400, content={"error": "Invalid message index"})

    update_field = f"messages.{question_index}.rating"
    sessions_collection.update_one({"session_id": session_id}, {"$set": {update_field: rating}})
    return {"message": "Rating Updated"}

@app.post("/api/update-language")
async def update_language(data: dict = Body(...)):
    device_id = data.get("device_id")
    state = data.get("state")
    language = data.get("language")
    print(state, language)

    if not device_id:
        return JSONResponse(status_code=400, content={"error": "Device ID is required"})

    result = sessions_collection.update_many(
        {"device_id": device_id},
        {"$set": {"state": state, "language": language}}
    )
    return {"status": "success", "matched": result.matched_count, "updated": result.modified_count}


@app.post("/api/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...), language: str = Form("English")):
    try:
        logger.info(f"Received file: {file.filename}")
        
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")

        logger.info(f"Using language: {language} (Local Whisper)")

        whisper_instance = get_whisper_instance()
        transcript = whisper_instance.transcribe_audio(contents, file.filename)

        if transcript.startswith("Error:"):
            logger.error(f"Local Whisper transcription failed: {transcript}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Transcription failed: {transcript}"}
            )

        logger.info(f"Transcript: {transcript[:100]}...")

        return {"transcript": transcript}

    except Exception as e:
        logger.exception("Local Whisper transcription failed")
        return JSONResponse(status_code=500, content={"error": f"Local Whisper transcription failed: {str(e)}"})

@app.post("/api/recommendations")
async def get_recommendations(data: dict = Body(...)):
    """
    Get question recommendations based on the user's current question and state
    
    Expected input:
    {
        "question": "How to grow tomatoes in summer?",
        "state": "Maharashtra",
        "limit": 2
    }
    """
    try:
        user_question = data.get('question', '')
        user_state = data.get('state', '')
        limit = data.get('limit', 2)
        
        if not user_question.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Question is required"}
            )
        
        # recommendations = get_question_recommendations(
        #     user_question=user_question,
        #     user_state=user_state,
        #     limit=min(limit, 5)
        # )
        recommendations = []
        
        return {
            "recommendations": recommendations,
            "total_found": len(recommendations),
            "based_on": {
                "question": user_question[:100] + "..." if len(user_question) > 100 else user_question,
                "state": user_state or "any"
            }
        }
        
    except Exception as e:
        logger.error(f"Error in recommendations endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to generate recommendations"}
        )


















@app.post("/api/test-database-toggle")
async def test_database_toggle(
    question: str = Form(...),
    golden_db: bool = Form(False),
    rag_db: bool = Form(False), 
    pops_db: bool = Form(False),
    llm_fallback: bool = Form(False)
):
    """Test endpoint for database toggle functionality"""
    db_config = DatabaseConfig(
        golden_db_enabled=golden_db,
        rag_db_enabled=rag_db,
        pops_db_enabled=pops_db,
        llm_fallback_enabled=llm_fallback
    )
    
    try:
        response = await get_answer(question, None, "unknown", None, db_config if not db_config.is_traditional_mode() else None)
        
        return {
            "question": question,
            "answer": response,
            "database_config": {
                "golden_db_enabled": golden_db,
                "rag_db_enabled": rag_db,
                "pops_db_enabled": pops_db,
                "llm_fallback_enabled": llm_fallback,
                "mode": "traditional" if db_config.is_traditional_mode() else "selective",
                "enabled_databases": db_config.get_enabled_databases()
            }
        }
    except Exception as e:
        logger.error(f"[TEST] Error in database toggle test: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Database toggle test failed: {str(e)}"}
        )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
