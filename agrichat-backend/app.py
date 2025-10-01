from fastapi import FastAPI, Request, Form, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pymongo import MongoClient
from uuid import uuid4
from datetime import datetime
import sys
import os
import csv
from io import StringIO
import certifi
import pytz
from contextlib import asynccontextmanager
import logging
from typing import Optional, List, Dict
import time
import asyncio
import hashlib
from functools import lru_cache
from local_whisper_interface import get_whisper_instance
import re
from Agentic_RAG.fast_response_handler import FastResponseHandler
from Agentic_RAG.database_config import DatabaseConfig
from Agentic_RAG import main as rag_main
import json
import markdown
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.memory import ConversationBufferWindowMemory
from fastapi.responses import StreamingResponse
from bs4 import BeautifulSoup
from dateutil import parser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")
current_dir = os.path.dirname(os.path.abspath(__file__))
agentic_rag_path = os.path.join(current_dir, "Agentic_RAG")
sys.path.insert(0, agentic_rag_path)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

USE_FAST_MODE = os.getenv("USE_FAST_MODE", "true").lower() == "true"
DISABLE_RECOMMENDATIONS = os.getenv("DISABLE_RECOMMENDATIONS", "false").lower() == "true"
PARALLEL_RECOMMENDATIONS = os.getenv("PARALLEL_RECOMMENDATIONS", "true").lower() == "true"

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
from Agentic_RAG.chroma_query_handler import ChromaQueryHandler
from Agentic_RAG.tools import is_agricultural_query

chroma_db_path = os.getenv('CHROMA_DB_PATH')
if not chroma_db_path:
    if os.path.exists('/app') and os.path.exists('/app/chromaDb'):
        chroma_db_path = '/app/chromaDb'
        environment = 'Docker'
    else:
        chroma_db_path = '/home/ubuntu/agrichat-annam/agrichat-backend/chromaDb'
        environment = 'Local'

logger.info(f"[CONFIG] Environment: {environment}")
logger.info(f"[CONFIG] ChromaDB path: {chroma_db_path}")
logger.info(f"[CONFIG] ChromaDB exists: {os.path.exists(chroma_db_path)}")

query_handler = ChromaQueryHandler(chroma_path=chroma_db_path)

pipeline = None

local_llm = None
local_embeddings = None
response_formatter = None
try:
    from Agentic_RAG.local_llm_interface import LocalLLMInterface
    from Agentic_RAG.local_llm_interface import LocalEmbeddings
    local_llm = LocalLLMInterface()
    local_embeddings = LocalEmbeddings()
except Exception as e:
    logger.warning(f"[CONFIG] Local LLM/embeddings initialization failed: {e}")

try:
    from response_formatter import AgriculturalResponseFormatter
    response_formatter = AgriculturalResponseFormatter()
except Exception as e:
    logger.warning(f"[CONFIG] Response formatter initialization failed: {e}")

MONGO_URI = os.getenv("MONGO_URI")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
try:
    client = MongoClient(MONGO_URI) if ENVIRONMENT == "development" else MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client.get_database("agrichat")
    sessions_collection = db["sessions"]
    logger.info("[DB] MongoDB client initialized and 'sessions' collection ready")
except Exception as e:
    logger.error(f"[DB] Failed to initialize MongoDB client: {e}")
    sessions_collection = None

session_memories = {}

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
    "https://c455816e214e.ngrok-free.app", 
    "https://localhost:3000",
    "https://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
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
    
    if not DISABLE_RECOMMENDATIONS and PARALLEL_RECOMMENDATIONS:
        tasks.append(get_question_recommendations_async(request.question, request.state, 4))
    
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
    if not DISABLE_RECOMMENDATIONS:
        if not PARALLEL_RECOMMENDATIONS:
            try:
                recommendations = get_question_recommendations(
                    user_question=request.question,
                    user_state=request.state,
                    limit=4
                )
                logger.info(f"Added {len(recommendations)} recommendations to session response (sequential)")
            except Exception as e:
                logger.error(f"Failed to get sequential recommendations: {e}")
                recommendations = []
        else:
            logger.info(f"Added {len(recommendations)} recommendations to session response (parallel)")
        
        session["recommendations"] = recommendations
    else:
        session["recommendations"] = []
        logger.info(f"[PERFORMANCE] Recommendations disabled for speed optimization")
    recommendations_time = time.time() - recommendations_start
    logger.info(f"[TIMING] Recommendations handling took: {recommendations_time:.3f}s")
    
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
    if not DISABLE_RECOMMENDATIONS and PARALLEL_RECOMMENDATIONS:
        logger.info("[PARALLEL] Starting session recommendations in parallel with answer processing")
    tasks.append(get_question_recommendations_async(request.question, current_state, 4))
    
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
        
        if not DISABLE_RECOMMENDATIONS:
            if not PARALLEL_RECOMMENDATIONS:
                try:
                    recommendations = get_question_recommendations(
                        user_question=request.question,
                        user_state=current_state,
                        limit=4
                    )
                    updated["recommendations"] = recommendations
                    logger.info(f"Added {len(recommendations)} session recommendations (sequential)")
                except Exception as e:
                    logger.error(f"Failed to get sequential session recommendations: {e}")
                    updated["recommendations"] = []
            else:
                updated["recommendations"] = recommendations
                logger.info(f"Added {len(recommendations)} session recommendations (parallel)")
        else:
            updated["recommendations"] = []
            logger.info(f"[PERFORMANCE] Session recommendations disabled for speed optimization")
    
    return {"session": updated}

@app.post("/api/query/stream")
async def stream_query(question: str = Form(...), device_id: str = Form(...), state: str = Form(...), language: str = Form(...)):
    """
    Streaming endpoint that shows thinking progress
    """
    async def generate_thinking_stream():
        import json
        import asyncio
        
        thinking_states = [
            "Understanding your question...",
            "Searching agricultural database...", 
            "Processing with AI...",
            "Generating response..."
        ]
        
        for i, state in enumerate(thinking_states):
            yield f"data: {json.dumps({'type': 'thinking', 'message': state, 'progress': (i + 1) * 25})}\n\n"
            await asyncio.sleep(0.5)
        
        try:
            session_id = str(uuid4())
            memory = get_session_memory(session_id)
            memory.chat_memory.add_user_message(question)
            logger.info(f"[STREAM] Streaming session {session_id} started for question: {question[:80]}")


            classifier_prompt = f"""You are an agricultural expert assistant. Analyze this farmer's question for completeness and classify it.

Current Question: "{question}"

CRITICAL GUIDELINES FOR COMPLETENESS:

MARK AS COMPLETE if the question is:
1. **Direct Facts**: "What is the seed rate of [crop]?", "How to control [pest/disease]?", "When to harvest [crop]?"
2. **General Practices**: "How to grow [crop]?", "What fertilizer for [crop]?", "Best varieties of [crop]?"
3. **Technical Questions**: "What causes [disease] in [crop]?", "How to prepare soil for [crop]?"

MARK AS INCOMPLETE only if:
1. **Vague Questions**: "What should I grow?", "Help with my crop problem" (no specific crop mentioned)
2. **Personal Advice Needing Context**: "What's best for my farm?" (no location/conditions given)

EXAMPLES:
- "What is seed rate of bitter gourd?" → COMPLETE (direct fact)
- "How to control aphids in cotton?" → COMPLETE (specific pest + crop)
- "When to plant tomatoes?" → COMPLETE (can give general timing)
- "What should I grow?" → INCOMPLETE (needs location/conditions)
- "My crop is dying, help!" → INCOMPLETE (vague, no crop specified)

Current Question: "{question}"

Respond in this format:
COMPLETENESS: [COMPLETE/INCOMPLETE]  
QUERY_TYPE: [direct_fact/complex_reasoning/personalized_advice]
CONFIDENCE: [0.0-1.0]
MISSING_INFO: [only list if question is truly vague and needs critical context]
AGRICULTURAL_RELEVANCE: [0.0-1.0]
REASONING: [brief explanation]"""

            classifier_output = []
            for event in local_llm.stream_generate(classifier_prompt, model=os.getenv('OLLAMA_MODEL_CLASSIFIER', 'qwen3:1.7b'), temperature=0.0):
                etype = event.get('type')
                if etype == 'model':
                    yield f"data: {json.dumps({'type': 'model', 'model': event.get('model'), 'source': event.get('source')})}\n\n"
                    continue
                if etype == 'token':
                    text = event.get('text')
                    classifier_output.append(text)
                    yield f"data: {json.dumps({'type': 'research', 'source': event.get('source'), 'model': event.get('model'), 'chunk': text})}\n\n"
                elif etype == 'raw':
                    yield f"data: {json.dumps({'type': 'research_raw', 'source': event.get('source'), 'model': event.get('model'), 'data': event.get('data')})}\n\n"
                elif etype == 'error':
                    yield f"data: {json.dumps({'type': 'research_error', 'message': event.get('message')})}\n\n"

            classifier_text = ''.join(classifier_output).strip()
            logger.info(f"[STREAM] Classifier output length={len(classifier_text)}")


            is_incomplete = False
            completeness_status = "COMPLETE"  # Default to complete for lenient approach
            

            for line in classifier_text.split('\n'):
                if 'COMPLETENESS:' in line.upper():
                    if 'INCOMPLETE' in line.upper():
                        completeness_status = "INCOMPLETE"
                        is_incomplete = True
                    break
            

            if 'MISSING_INFO' in classifier_text.upper() and completeness_status == "INCOMPLETE":
                is_incomplete = True

            try:
                try_enhanced_query = question
                try:
                    recent = memory.chat_memory.messages[-4:]
                    recent_text = ' '.join([m.content if hasattr(m, 'content') else str(m) for m in recent])
                    if recent_text:
                        try_enhanced_query = f"{question} {recent_text[:400]}"
                except Exception:
                    pass

                early_docs = []
                try:
                    early_docs = query_handler.db.similarity_search_with_score(try_enhanced_query, k=6)
                except Exception:
                    early_docs = []

                for d, dist in early_docs:
                    try:
                        def _norm_text(t: str) -> str:
                            return re.sub(r"\s+", " ", (t or '').strip().lower())

                        stored_q = None
                        try:
                            stored_q = query_handler._extract_stored_question(d)
                        except Exception:
                            stored_q = None

                        stored_q_norm = _norm_text(stored_q) if stored_q else None
                        question_norm = _norm_text(question)

                        similarity = None
                        try:
                            if dist is not None:
                                similarity = max(0.0, 1.0 - float(dist)) if float(dist) <= 1.0 else float(dist)
                        except Exception:
                            similarity = None

                        is_stored_exact = False
                        if stored_q_norm and question_norm:
                            if stored_q_norm == question_norm or query_handler._is_exact_question_match(question, stored_q, threshold=0.95):
                                is_stored_exact = True

                        high_conf_similarity = (similarity is not None and similarity >= 0.85)

                        if is_stored_exact or high_conf_similarity:
                            doc_state = ''
                            try:
                                doc_state = (d.metadata.get('State') or d.metadata.get('state') or '').strip().lower()
                            except Exception:
                                doc_state = ''

                            user_state_norm = (state or '').strip().lower() if state is not None else ''
                            if (not user_state_norm) or (doc_state and user_state_norm and doc_state == user_state_norm) or (not doc_state and not user_state_norm):
                                answer_text = ''
                                try:
                                    content_lines = d.page_content.split('\n') if hasattr(d, 'page_content') else []
                                    capturing = False
                                    for line in content_lines:
                                        if line.strip().lower().startswith('answer:'):
                                            answer_text = line.split(':', 1)[1].strip()
                                            capturing = True
                                        elif capturing and line.strip():
                                            answer_text += ' ' + line.strip()
                                        elif capturing and not line.strip():
                                            break
                                except Exception:
                                    answer_text = getattr(d, 'page_content', '')[:2000]

                                final_text = answer_text or getattr(d, 'page_content', '')

                                source_name = 'Golden FAQ' if stored_q_norm else 'RAG Database'
                                conf_val = None
                                if similarity is not None:
                                    conf_val = similarity
                                else:
                                    conf_val = 0.95 if is_stored_exact else (0.8 if high_conf_similarity else 0.7)

                                confidence = 'High' if conf_val >= 0.8 else 'Medium' if conf_val >= 0.6 else 'Low'

                                try:
                                    formatted = response_formatter.format_simple_answer(final_text, source=source_name, similarity=conf_val)
                                except Exception:
                                    formatted = f"{final_text}\n\n---\n*Source: {source_name} (Confidence: {confidence})*"


                                try:
                                    formatted_clean = re.sub(r"(Direct Answer:|Answer:|Response:)\s*", "", formatted, flags=re.IGNORECASE).strip()
                                except Exception:
                                    formatted_clean = formatted.strip()

                                yield f"data: {json.dumps({'type': 'answer', 'chunk': formatted_clean})}\n\n"

                                html_answer = markdown.markdown(formatted, extensions=["extra", "nl2br"])
                                memory.chat_memory.add_ai_message(formatted)
                                session = {"session_id": session_id, "timestamp": datetime.now(IST).isoformat(), "messages": [{"question": question, "answer": html_answer, "rating": None}], "crop": "unknown", "state": state, "status": "active", "language": language, "has_unread": True, "device_id": device_id}
                                try:
                                    if sessions_collection:
                                        sessions_collection.insert_one(session)
                                        session.pop("_id", None)
                                except Exception:
                                    logger.warning("[STREAM] Failed to persist golden-match session to DB")

                                yield f"data: {json.dumps({'type': 'complete', 'session': session})}\n\n"
                                return
                    except Exception:
                        logger.debug("[STREAM] Early golden-match check failed for a doc, continuing")
                        continue
            except Exception:
                pass

            if is_incomplete:
                query_lower = question.lower()
                crop_mentioned = None
                crops = ['rice', 'wheat', 'cotton', 'tomato', 'potato', 'maize', 'corn', 'sugarcane', 'onion', 'soybean']
                for crop in crops:
                    if crop in query_lower:
                        crop_mentioned = crop
                        break
                
                missing_info = []
                if 'location' in classifier_text.lower() or 'state' in classifier_text.lower():
                    missing_info.append('location')
                if 'soil' in classifier_text.lower():
                    missing_info.append('soil type')
                if 'season' in classifier_text.lower():
                    missing_info.append('season')  
                if 'space' in classifier_text.lower() or 'size' in classifier_text.lower():
                    missing_info.append('space')
                
                recommendations = []
                if 'location' in missing_info:
                    recommendations.append("• **Location**: Which state/region are you in?")
                if 'soil type' in missing_info:
                    recommendations.append("• **Soil Type**: What type of soil do you have?")
                if 'season' in missing_info:
                    recommendations.append("• **Season**: Which season are you planning to grow?")
                if 'space' in missing_info:
                    recommendations.append("• **Space**: How much area do you have available?")
                
                if not recommendations:
                    recommendations = [
                        "• **Location**: Which state/region are you in?",
                        "• **Space**: How much area do you have available?"
                    ]
                
                recommendations_text = "\n".join(recommendations)
                
                if crop_mentioned:
                    intro = f"I need some additional information to give you the best advice about {crop_mentioned} cultivation."
                else:
                    intro = "I need some additional information to give you the best agricultural advice."
                
                clarification = f"""{intro}

**To provide accurate recommendations, please tell me:**
{recommendations_text}

**Why this helps:**
Agricultural recommendations vary based on your location, available space, and growing conditions. With these details, I can provide specific advice for your situation."""
                
                chunk_size = 100
                for i in range(0, len(clarification), chunk_size):
                    chunk = clarification[i:i+chunk_size]
                    yield f"data: {json.dumps({'type': 'answer', 'chunk': chunk})}\n\n"
                    await asyncio.sleep(0.1)  # Small delay for streaming effect

                final_text = clarification
                html_answer = markdown.markdown(final_text, extensions=["extra", "nl2br"])
                memory.chat_memory.add_ai_message(final_text)
                session = {"session_id": session_id, "timestamp": datetime.now(IST).isoformat(), "messages": [{"question": question, "answer": html_answer, "rating": None}], "crop": "unknown", "state": state, "status": "active", "language": language, "has_unread": True, "device_id": device_id}
                try:
                    sessions_collection.insert_one(session)
                    session.pop("_id", None)
                except Exception:
                    logger.warning("[STREAM] Failed to persist clarification session to DB")

                yield f"data: {json.dumps({'type': 'complete', 'session': session})}\n\n"
                return

            try:
                enhanced_query = question
                try:
                    recent = memory.chat_memory.messages[-4:]
                    recent_text = ' '.join([m.content if hasattr(m, 'content') else str(m) for m in recent])
                    if recent_text:
                        enhanced_query = f"{question} {recent_text[:400]}"
                except Exception:
                    pass

                docs = []
                try:
                    docs = query_handler.db.similarity_search_with_score(enhanced_query, k=5)
                except Exception as e:
                    logger.warning(f"[STREAM] Direct Chroma query failed: {e}")

                best_doc = None
                best_score = 0.0
                if docs:
                    for d, dist in docs:
                        score = 1.0 - dist if dist is not None else 0.0
                        if score > best_score:
                            best_score = score
                            best_doc = (d, dist)

                if best_doc and best_score > 0.6:
                    doc, distance = best_doc
                    meta = doc.metadata if hasattr(doc, 'metadata') else {}
                    yield f"data: {json.dumps({'type': 'research', 'source': 'rag_direct', 'metadata': meta, 'similarity': best_score})}\n\n"
                    raw_snippet = (doc.page_content[:1000] + '...') if hasattr(doc, 'page_content') else ''
                    yield f"data: {json.dumps({'type': 'research_raw', 'source': 'rag_direct', 'data': raw_snippet})}\n\n"

                    struct_prompt = query_handler.STRUCTURED_PROMPT.format(context=doc.page_content, question=question, region_instruction=query_handler.REGION_INSTRUCTION, current_month=datetime.now(IST).strftime('%B'))

                    answer_chunks = []
                    for event in local_llm.stream_generate(struct_prompt, model=os.getenv('OLLAMA_MODEL_STRUCTURER', 'gemma:latest'), temperature=0.2):
                        etype = event.get('type')
                        if etype == 'model':
                            yield f"data: {json.dumps({'type': 'model', 'model': event.get('model'), 'source': event.get('source')})}\n\n"
                            continue
                        if etype == 'token':
                            chunk = event.get('text')
                            answer_chunks.append(chunk)
                            yield f"data: {json.dumps({'type': 'answer', 'chunk': chunk, 'model': event.get('model'), 'source': event.get('source')})}\n\n"
                        elif etype == 'raw':
                            yield f"data: {json.dumps({'type': 'research_raw', 'source': event.get('source'), 'model': event.get('model'), 'data': event.get('data')})}\n\n"

                    final_text = ''.join(answer_chunks)

                    eval_prompt = f"Evaluate this response for the question: {question}\nResponse: {final_text}\nProvide QUALITY_SCORE: [1-10] and SATISFACTORY: [YES/NO]"
                    eval_stream = []
                    for e in local_llm.stream_generate(eval_prompt, model=os.getenv('OLLAMA_MODEL_CLASSIFIER', 'qwen3:1.7b'), temperature=0.1):
                        etype = e.get('type')
                        if etype == 'model':
                            yield f"data: {json.dumps({'type': 'model', 'model': e.get('model'), 'source': e.get('source')})}\n\n"
                            continue
                        if etype == 'token':
                            yield f"data: {json.dumps({'type': 'research', 'source': e.get('source'), 'model': e.get('model'), 'chunk': e.get('text')})}\n\n"
                            eval_stream.append(e.get('text'))
                        elif etype == 'raw':
                            yield f"data: {json.dumps({'type': 'research_raw', 'source': e.get('source'), 'model': e.get('model'), 'data': e.get('data')})}\n\n"

                    eval_text = ''.join(eval_stream)
                    quality_score = 0.7
                    satisfactory = True
                    if 'QUALITY_SCORE' in eval_text.upper():
                        try:
                            score_line = [l for l in eval_text.splitlines() if 'QUALITY_SCORE' in l.upper()]
                            if score_line:
                                val = ''.join([c for c in score_line[0] if (c.isdigit() or c=='.') or c==' '])
                                quality_score = float(val.strip()[:4]) / 10.0 if val.strip() else 0.7
                        except Exception:
                            quality_score = 0.7
                    if 'SATISFACTORY' in eval_text.upper() and 'NO' in eval_text.upper():
                        satisfactory = False

                    if not satisfactory or quality_score < 0.6:
                        fallback_prompt = f"The previous structured response didn't fully satisfy the question: {question}\nPrevious response: {final_text}\nPlease provide a complete, accurate, and actionable agricultural response."
                        for ev in local_llm.stream_generate(fallback_prompt, model=os.getenv('OLLAMA_MODEL_FALLBACK', 'gpt-oss:20b'), temperature=0.3):
                            etype = ev.get('type')
                            if etype == 'model':
                                yield f"data: {json.dumps({'type': 'model', 'model': ev.get('model'), 'source': ev.get('source')})}\n\n"
                                continue
                            if etype == 'token':
                                yield f"data: {json.dumps({'type': 'answer', 'chunk': ev.get('text'), 'model': ev.get('model'), 'source': ev.get('source')})}\n\n"
                            elif etype == 'raw':
                                yield f"data: {json.dumps({'type': 'research_raw', 'source': ev.get('source'), 'model': ev.get('model'), 'data': ev.get('data')})}\n\n"

                        persisted_text = final_text
                    else:
                        persisted_text = final_text

                    html_answer = markdown.markdown(persisted_text, extensions=["extra", "nl2br"])
                    memory.chat_memory.add_ai_message(persisted_text)
                    session = {"session_id": session_id, "timestamp": datetime.now(IST).isoformat(), "messages": [{"question": question, "answer": html_answer, "rating": None}], "crop": "unknown", "state": state, "status": "active", "language": language, "has_unread": True, "device_id": device_id}
                    try:
                        sessions_collection.insert_one(session)
                        session.pop("_id", None)
                    except Exception:
                        logger.warning("[STREAM] Failed to persist session to DB")

                    yield f"data: {json.dumps({'type': 'complete', 'session': session})}\n\n"
                    return

                else:
                    fallback_prompt = f"No direct database match was found. Answer this agricultural question comprehensively: {question}"
                    for ev in local_llm.stream_generate(fallback_prompt, model=os.getenv('OLLAMA_MODEL_FALLBACK', 'gpt-oss:20b'), temperature=0.3):
                        etype = ev.get('type')
                        if etype == 'model':
                            yield f"data: {json.dumps({'type': 'model', 'model': ev.get('model'), 'source': ev.get('source')})}\n\n"
                            continue
                        if etype == 'token':
                            yield f"data: {json.dumps({'type': 'answer', 'chunk': ev.get('text'), 'model': ev.get('model'), 'source': ev.get('source')})}\n\n"
                        elif etype == 'raw':
                            yield f"data: {json.dumps({'type': 'research_raw', 'source': ev.get('source'), 'model': ev.get('model'), 'data': ev.get('data')})}\n\n"

                    final_text = ''
                    html_answer = markdown.markdown(final_text, extensions=["extra", "nl2br"])
                    memory.chat_memory.add_ai_message(final_text)
                    session = {"session_id": session_id, "timestamp": datetime.now(IST).isoformat(), "messages": [{"question": question, "answer": html_answer, "rating": None}], "crop": "unknown", "state": state, "status": "active", "language": language, "has_unread": True, "device_id": device_id}
                    try:
                        sessions_collection.insert_one(session)
                        session.pop("_id", None)
                    except Exception:
                        logger.warning("[STREAM] Failed to persist fallback session to DB")

                    yield f"data: {json.dumps({'type': 'complete', 'session': session})}\n\n"
                    return

            except Exception as e:
                logger.error(f"[STREAM] Pipeline error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Processing failed. Please try again.'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Processing failed. Please try again.'})}\n\n"
    
    return StreamingResponse(
        generate_thinking_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive", 
            "Content-Type": "text/event-stream",
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
        
        recommendations = get_question_recommendations(
            user_question=user_question,
            user_state=user_state,
            limit=min(limit, 5)
        )
        
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
