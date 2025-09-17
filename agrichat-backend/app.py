from fastapi import FastAPI, Request, Form, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pymongo import MongoClient
from uuid import uuid4
from datetime import datetime
from bs4 import BeautifulSoup
from pathlib import Path
import sys
import os
import markdown
import csv
from io import StringIO
import certifi
import pytz
from dateutil import parser
from contextlib import asynccontextmanager
import logging
from typing import Optional, List, Dict
import requests
import time
import asyncio
import concurrent.futures
import hashlib
from functools import lru_cache
from local_whisper_interface import local_whisper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.schema import HumanMessage, AIMessage
from Agentic_RAG.fast_response_handler import FastResponseHandler
from Agentic_RAG.context_manager import ConversationContext
from Agentic_RAG.database_config import DatabaseConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")
current_dir = os.path.dirname(os.path.abspath(__file__))
agentic_rag_path = os.path.join(current_dir, "Agentic_RAG")
sys.path.insert(0, agentic_rag_path)

USE_FAST_MODE = os.getenv("USE_FAST_MODE", "true").lower() == "true"
DISABLE_RECOMMENDATIONS = os.getenv("DISABLE_RECOMMENDATIONS", "false").lower() == "true"
PARALLEL_RECOMMENDATIONS = os.getenv("PARALLEL_RECOMMENDATIONS", "true").lower() == "true"
logger.info(f"[CONFIG] USE_FAST_MODE environment variable: {os.getenv('USE_FAST_MODE', 'not set')}")
logger.info(f"[CONFIG] Fast Mode Enabled: {USE_FAST_MODE}")
logger.info(f"[CONFIG] Recommendations Disabled: {DISABLE_RECOMMENDATIONS}")
logger.info(f"[CONFIG] Parallel Recommendations Enabled: {PARALLEL_RECOMMENDATIONS}")

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

fast_handler = None
if USE_FAST_MODE:
    try:
        fast_handler = FastResponseHandler()
        logger.info("[CONFIG] Fast response handler loaded successfully - 50% performance improvement enabled")
    except Exception as e:
        logger.warning(f"[CONFIG] Fast response handler initialization failed: {e}")
        logger.info("[CONFIG] Falling back to CrewAI mode")
        USE_FAST_MODE = False
else:
    logger.info("[CONFIG] CrewAI Mode ENABLED - Using multi-agent workflow")

from crewai import Crew
from Agentic_RAG.crew_agents import (
    Retriever_Agent, Grader_agent,
    hallucination_grader, answer_grader
)
from Agentic_RAG.crew_tasks import (
    retriever_task, grader_task,
    hallucination_task, answer_task
)
from Agentic_RAG.chroma_query_handler import ChromaQueryHandler

if os.path.exists("/app"):
    chroma_db_path = "/app/chromaDb"
    environment = "Docker"
else:
    chroma_db_path = "/home/ubuntu/agrichat-annam/agrichat-backend/chromaDb" 
    environment = "Local"

logger.info(f"[CONFIG] Environment: {environment}")
logger.info(f"[CONFIG] ChromaDB path: {chroma_db_path}")
logger.info(f"[CONFIG] ChromaDB exists: {os.path.exists(chroma_db_path)}")

query_handler = ChromaQueryHandler(chroma_path=chroma_db_path)

session_memories = {}

recommendations_cache = {}
CACHE_EXPIRY_SECONDS = 3600

def get_session_memory(session_id: str):
    """Get or create conversation memory for a session"""
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferWindowMemory(
            k=5,
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

def get_answer(question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = None, session_id: str = None, db_config: DatabaseConfig = None) -> str:
    """
    Main answer function that routes to fast mode or CrewAI based on configuration.
    Simplified structure aligned with main.py pipeline.
    """
    if USE_FAST_MODE and fast_handler:
        response = get_fast_answer(question, conversation_history, user_state, db_config)
    else:
        response = get_crewai_answer(question, conversation_history, user_state, db_config)
    
    return response

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
                # Generic follow-up
                return f"{question} for {topic}"
    
    return question

def extract_topics_from_context(context: str) -> List[str]:
    """Enhanced extraction of agricultural topics from conversation context"""
    topics = []
    context_lower = context.lower()
    
    # Enhanced disease patterns
    disease_patterns = [
        "late blight", "early blight", "powdery mildew", "downy mildew", 
        "bacterial wilt", "fungal infection", "leaf spot", "root rot", "stem rot",
        "collar rot", "blast", "sheath blight", "rust", "smut", "mosaic virus",
        "yellowing", "wilting", "damping off", "canker", "scab"
    ]
    
    # Enhanced crop patterns
    crop_patterns = [
        "potato", "tomato", "wheat", "rice", "cotton", "sugarcane", "maize", "corn",
        "onion", "garlic", "chili", "pepper", "brinjal", "eggplant", "okra", "cucumber",
        "cabbage", "cauliflower", "carrot", "radish", "beans", "peas", "groundnut",
        "soybean", "mustard", "sesame", "sunflower", "mango", "banana", "guava",
        "papaya", "coconut", "tea", "coffee", "spices", "turmeric", "ginger"
    ]
    
    # Pest patterns
    pest_patterns = [
        "aphid", "thrips", "whitefly", "bollworm", "stem borer", "fruit borer",
        "leaf miner", "scale insect", "mealybug", "spider mite", "nematode",
        "caterpillar", "grub", "weevil", "beetle", "locust", "grasshopper"
    ]
    
    # Problem patterns
    problem_patterns = [
        "nutrient deficiency", "nitrogen deficiency", "phosphorus deficiency",
        "potassium deficiency", "iron deficiency", "zinc deficiency", "magnesium deficiency",
        "water stress", "drought stress", "waterlogging", "poor growth", "stunted growth",
        "low yield", "poor germination", "flower drop", "fruit drop"
    ]
    
    # Practice patterns
    practice_patterns = [
        "organic farming", "crop rotation", "intercropping", "mulching", "pruning",
        "grafting", "seed treatment", "soil preparation", "land preparation",
        "transplanting", "direct sowing", "drip irrigation", "sprinkler irrigation"
    ]
    
    # Look for disease + crop combinations first
    for disease in disease_patterns:
        if disease in context_lower:
            for crop in crop_patterns:
                if crop in context_lower:
                    topics.append(f"{disease} in {crop}")
                    break
            else:
                topics.append(disease)
            break
    
    # Look for pest + crop combinations
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
    
    # Look for problem + crop combinations
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
    
    # Look for practice + crop combinations
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
    
    # If no specific combinations found, look for individual crops
    if not topics:
        for crop in crop_patterns:
            if crop in context_lower:
                topics.append(crop)
                break
    
    return topics

def get_fast_answer(question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = None, db_config: DatabaseConfig = None) -> str:
    """
    Fast mode using enhanced FastResponseHandler with source attribution.
    """
    logger.info(f"[FAST] Processing question with enhanced fast handler: {question}")
    try:
        return fast_handler.get_answer(question, conversation_history, user_state, db_config)
    except Exception as e:
        logger.error(f"[FAST] Error in fast mode: {e}")
        logger.info("[FAST] Falling back to CrewAI")
        return get_crewai_answer(question, conversation_history, user_state, db_config)

def get_crewai_answer(question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = None, db_config: DatabaseConfig = None) -> str:
    """
    CrewAI mode - original multi-agent approach for complex reasoning.
    """
    logger.info(f"[CREWAI] Processing question with CrewAI approach: {question}")
    
    # Comment out greeting handling - now handled by enhanced FastResponseHandler
    # greeting_check_start = time.time()
    # question_lower = question.lower().strip()
    # simple_greetings = [
    #     'hi', 'hello', 'hey', 'namaste', 'namaskaram', 'vanakkam', 
    #     'good morning', 'good afternoon', 'good evening', 'good day',
    #     'howdy', 'greetings', 'salaam', 'adaab', 'hi there', 'hello there'
    # ]
    # 
    # if len(question_lower) < 20 and any(greeting in question_lower for greeting in simple_greetings):
    #     greeting_time = time.time() - greeting_check_start
    #     logger.info(f"[TIMING] Greeting detection took: {greeting_time:.3f}s")
    #     logger.info(f"[GREETING] Detected simple greeting: {question}")
    #     logger.info(f"[SOURCE] Fast pattern matching used for greeting: {question}")
    #     state_context = f" in {user_state}" if user_state and user_state.lower() != "unknown" else " in India"
    #     if 'namaste' in question_lower:
    #         return f"Namaste! Welcome to AgriChat, your trusted agricultural assistant for Indian farming{state_context}. I specialize in crop management, soil health, weather patterns, and farming practices specific to Indian conditions. What agricultural challenge can I help you with today?"
    #     elif 'namaskaram' in question_lower:
    #         return f"Namaskaram! I'm your specialized agricultural assistant for Indian farmers{state_context}. Feel free to ask me about Indian crop varieties, monsoon farming, soil management, or any farming techniques suited to Indian climate and conditions."
    #     elif 'vanakkam' in question_lower:
    #         return f"Vanakkam! I'm here to assist you with Indian farming and agriculture{state_context}. What regional agricultural topic would you like to discuss - from rice cultivation to spice farming, I'm here to help with India-specific guidance!"
    #     elif any(time in question_lower for time in ['morning', 'afternoon', 'evening']):
    #         time_word = next(time for time in ['morning', 'afternoon', 'evening'] if time in question_lower)
    #         return f"Good {time_word}! I'm your agricultural assistant specializing in Indian farming practices{state_context}. How can I help you with your crop management, seasonal farming, or any agriculture-related questions specific to Indian conditions today?"
    #     else:
    #         return f"Hello! I'm your agricultural assistant specializing in Indian farming and crop management{state_context}. I'm here to help with crops, farming techniques, and agricultural practices tailored to Indian soil, climate, and regional conditions. What would you like to know?"
    # greeting_time = time.time() - greeting_check_start
    # logger.info(f"[TIMING] Greeting check took: {greeting_time:.3f}s")
    # 
    # if conversation_history:
    #     logger.info(f"[DEBUG] Using conversation context with {len(conversation_history)} previous interactions")
    # if user_state:
    #     logger.info(f"[DEBUG] Using frontend-detected state: {user_state}")
    
    try:
        # Comment out detailed timing - keep essential CrewAI functionality
        # crew_setup_start = time.time()
        rag_crew = Crew(
            agents=[
                Retriever_Agent
            ],
            tasks=[
                retriever_task
            ],
            verbose=True,
        )
        # crew_setup_time = time.time() - crew_setup_start
        # logger.info(f"[TIMING] CrewAI setup took: {crew_setup_time:.3f}s")
        
        # crew_execution_start = time.time()
        inputs = {
            "question": question,
            "conversation_history": conversation_history or []
        }
        
        result = rag_crew.kickoff(inputs=inputs)
        # crew_execution_time = time.time() - crew_execution_start
        # logger.info(f"[TIMING] CrewAI execution took: {crew_execution_time:.3f}s")
        logger.info(f"[DEBUG] CrewAI result: {result}")
        
        # Clean up source tags from CrewAI result
        # post_process_start = time.time()
        result_str = str(result).strip()
        if "Source: RAG Database" in result_str:
            logger.info(f"[SOURCE] RAG Database used for question: {question[:50]}...")
            result_str = result_str.replace("Source: RAG Database", "").strip()
        if "Source: Local LLM" in result_str:
            logger.info(f"[SOURCE] Local LLM used for question: {question[:50]}...")
            result_str = result_str.replace("Source: Local LLM", "").strip()
        # post_process_time = time.time() - post_process_start
        # logger.info(f"[TIMING] Post-processing took: {post_process_time:.3f}s")
        
        # total_crewai_time = time.time() - crewai_start
        # logger.info(f"[TIMING] TOTAL CrewAI processing took: {total_crewai_time:.3f}s")
        
        return result_str
            
    except Exception as e:
        logger.error(f"[DEBUG] Error in CrewAI execution: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please try again later."

def preprocess_question(question: str) -> str:
    """
    Preprocess question text for similarity comparison
    """
    question = question.lower()
    question = re.sub(r'[^a-zA-Z0-9\s]', '', question)
    question = re.sub(r'\s+', ' ', question).strip()
    return question

def get_question_recommendations(user_question: str, user_state: str = None, limit: int = 2) -> List[Dict]:
    """
    Get question recommendations based on similarity to user's question and state
    Only returns recommendations for agriculture-related questions
    Optimized for performance with caching and reduced database queries
    
    Args:
        user_question: The user's current question
        user_state: User's state for state-specific recommendations
        limit: Number of recommendations to return (default 2)
    
    Returns:
        List of recommended questions with their details (empty if not agriculture-related)
    """
    try:
        # Create cache key based on question and state
        cache_key = hashlib.md5(f"{user_question.lower()}_{user_state}_{limit}".encode()).hexdigest()
        current_time = time.time()
        
        # Check cache first
        if cache_key in recommendations_cache:
            cached_data, timestamp = recommendations_cache[cache_key]
            if current_time - timestamp < CACHE_EXPIRY_SECONDS:
                logger.info(f"[CACHE] Returning cached recommendations for: {user_question[:50]}...")
                return cached_data
            else:
                # Remove expired cache entry
                del recommendations_cache[cache_key]
        
        # Quick classification check first
        question_category = query_handler.classify_query(user_question)
        logger.info(f"[REC] Question classified as: {question_category} for: {user_question[:50]}...")
        if question_category != "AGRICULTURE":
            logger.info(f"[REC] Question classified as {question_category}, skipping recommendations: {user_question}")
            # Cache empty result for non-agriculture questions
            recommendations_cache[cache_key] = ([], current_time)
            return []
        
        # Optimized pipeline with indexed fields and reduced data
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
                "sample_answer": {"$first": "$messages.answer"}  # Keep full answer, truncate in Python
            }},
            {"$match": {"count": {"$gte": 1}}},
            {"$project": {
                "question": "$_id.question",
                "state": "$_id.state",
                "count": 1,
                "sample_answer": 1
            }},
            {"$sort": {"count": -1}},  # Sort by popularity for better results
            {"$limit": 100}  # Reduced from 200 for faster processing
        ]
        
        try:
            questions_data = list(sessions_collection.aggregate(pipeline))
            logger.info(f"[REC] Found {len(questions_data)} questions in database for recommendations")
        except Exception as db_error:
            logger.error(f"[REC] Database aggregation failed: {db_error}")
            # Fallback to simple query without aggregation
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
                # Cache empty result for database errors
                recommendations_cache[cache_key] = ([], current_time)
                return []
        
        if not questions_data:
            logger.info("[REC] No questions found in database for recommendations")
            # Cache empty result for no data
            recommendations_cache[cache_key] = ([], current_time)
            return []
        
        processed_user_question = preprocess_question(user_question)
        
        questions_list = []
        metadata_list = []
        
        # Skip classification for faster processing - pre-filtered by agriculture category check
        for item in questions_data:
            question = item['question']
            processed_question = preprocess_question(question)
            
            # Skip identical questions
            if processed_question == processed_user_question:
                continue
                
            questions_list.append(processed_question)
            metadata_list.append({
                'original_question': question,
                'state': item.get('state', 'unknown'),
                'count': item['count'],
                'sample_answer': item['sample_answer'][:200] + "..." if len(item['sample_answer']) > 200 else item['sample_answer']
            })
            
            # Early exit if we have enough candidates for processing
            if len(questions_list) >= 50:  # Process only top 50 for speed
                break
        
        if not questions_list:
            logger.info("[REC] No suitable questions found for recommendations after filtering")
            # Cache empty result for no suitable questions
            recommendations_cache[cache_key] = ([], current_time)
            return []
        
        logger.info(f"[REC] Processing {len(questions_list)} questions for similarity matching")
        
        # Optimized TF-IDF with reduced features for speed
        try:
            vectorizer = TfidfVectorizer(
                max_features=500,  # Reduced from 1000
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,  # Minimum document frequency
                max_df=0.95  # Maximum document frequency to filter common words
            )
            
            all_questions = [processed_user_question] + questions_list
            tfidf_matrix = vectorizer.fit_transform(all_questions)
            
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            logger.info(f"[REC] TF-IDF vectorization successful, computed {len(similarities)} similarities")
            
        except Exception as tfidf_error:
            logger.error(f"[REC] TF-IDF vectorization failed: {tfidf_error}")
            # Cache empty result for TF-IDF errors
            recommendations_cache[cache_key] = ([], current_time)
            return []
        
        recommendations = []
        for i, similarity_score in enumerate(similarities):
            if similarity_score > 0.1:
                rec = {
                    'question': metadata_list[i]['original_question'],
                    'state': metadata_list[i]['state'],
                    'similarity_score': float(similarity_score),
                    'popularity': metadata_list[i]['count'],
                    'sample_answer': metadata_list[i]['sample_answer'][:200] + "..." if len(metadata_list[i]['sample_answer']) > 200 else metadata_list[i]['sample_answer']
                }
                recommendations.append(rec)
        
        if user_state:
            def sort_key(rec):
                state_boost = 0.2 if rec['state'].lower() == user_state.lower() else 0
                return rec['similarity_score'] + state_boost
            recommendations.sort(key=sort_key, reverse=True)
        else:
            recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        top_recommendations = recommendations[:limit]
        
        logger.info(f"[REC] Generated {len(top_recommendations)} recommendations from {len(recommendations)} candidates for question: {user_question[:50]}...")
        if top_recommendations:
            logger.info(f"[REC] Top recommendation similarity scores: {[rec['similarity_score'] for rec in top_recommendations]}")
        
        # Cache the results
        recommendations_cache[cache_key] = (top_recommendations, current_time)
        
        # Clean up cache if it gets too large (keep only 100 most recent entries)
        if len(recommendations_cache) > 100:
            # Remove oldest entries
            sorted_cache = sorted(recommendations_cache.items(), key=lambda x: x[1][1])
            for old_key, _ in sorted_cache[:50]:  # Remove 50 oldest entries
                del recommendations_cache[old_key]
        
        return top_recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return []

async def get_question_recommendations_async(user_question: str, user_state: str = None, limit: int = 2) -> List[Dict]:
    """
    Async version of get_question_recommendations for parallel processing
    """
    loop = asyncio.get_event_loop()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = loop.run_in_executor(
            executor, 
            get_question_recommendations, 
            user_question, 
            user_state, 
            limit
        )
        result = await future
        return result

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] App initialized.")
    yield
    print("[Shutdown] App shutting down...")

app = FastAPI(lifespan=lifespan)

origins = [
    "https://agri-annam.vercel.app",
    "https://f3fc768cab21.ngrok-free.app",
    "https://localhost:3000",
    "https://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.middleware("http")
async def add_ngrok_headers(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method == "OPTIONS":
        response = JSONResponse(content={})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Max-Age"] = "86400"
        response.headers["ngrok-skip-browser-warning"] = "true"
        return response
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

MONGO_URI = os.getenv("MONGO_URI")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
client = MongoClient(MONGO_URI) if ENVIRONMENT == "development" else MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client["agrichat"]
sessions_collection = db["sessions"]

# Optimize database indexes for better query performance
def ensure_database_indexes():
    """Ensure proper indexes exist for optimized queries"""
    try:
        # Index for session queries
        sessions_collection.create_index([("session_id", 1)])
        sessions_collection.create_index([("device_id", 1)])
        sessions_collection.create_index([("state", 1)])
        
        # Compound index for recommendations queries
        sessions_collection.create_index([("state", 1), ("messages.question", 1)])
        sessions_collection.create_index([("timestamp", -1)])
        
        logger.info("[PERF] Database indexes ensured for optimal performance")
    except Exception as e:
        logger.error(f"[PERF] Failed to create indexes: {e}")

# Initialize indexes
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
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400"
        }
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
    
    # Start both answer processing and recommendations in parallel if enabled
    async def process_answer():
        try:
            answer_processing_start = time.time()
            # Run answer processing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                raw_answer = await loop.run_in_executor(
                    executor, 
                    get_answer, 
                    request.question, 
                    None,  # conversation_history
                    request.state,  # user_state
                    session_id,
                    db_config
                )
            answer_processing_time = time.time() - answer_processing_start
            logger.info(f"[TIMING] Answer processing took: {answer_processing_time:.3f}s")
            
            # Reduced debug logging for performance
            if answer_processing_time > 10:  # Only log if slow
                logger.info(f"[DEBUG] Slow answer processing detected: {answer_processing_time:.3f}s")
            
            return str(raw_answer).strip()
            
        except Exception as e:
            logger.error(f"[DEBUG] Error in get_answer: {e}")
            raise e

@app.post("/api/query-form")
async def new_session_form(
    question: str = Form(...), 
    device_id: str = Form(...), 
    state: str = Form(""), 
    language: str = Form("en"),
    file: UploadFile = File(None),
    golden_db: bool = Form(False),
    rag_db: bool = Form(True), 
    pops_db: bool = Form(False),
    llm_fallback: bool = Form(False)
):
    """Endpoint for FormData requests with optional file uploads"""
    api_start_time = time.time()
    session_id = str(uuid4())
    logger.info(f"[TIMING] API FormData endpoint started for question: {question[:50]}...")
    
    # Handle audio file if provided
    if file and file.size > 0:
        try:
            logger.info(f"[AUDIO] Processing audio file: {file.filename}")
            # Save the uploaded file temporarily
            temp_file_path = f"/tmp/{session_id}_{file.filename}"
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Transcribe audio
            transcript = local_whisper(temp_file_path, language)
            if transcript and transcript.strip():
                question = transcript.strip()
                logger.info(f"[AUDIO] Transcription successful: {question[:50]}...")
            
            # Clean up temp file
            os.remove(temp_file_path)
            
        except Exception as e:
            logger.error(f"[AUDIO] Error processing audio: {e}")
    
    # Create database configuration
    db_config = DatabaseConfig(
        golden_db_enabled=golden_db,
        rag_db_enabled=rag_db,
        pops_db_enabled=pops_db,
        llm_fallback_enabled=llm_fallback
    )
    
    # Create QueryRequest object
    request = QueryRequest(
        question=question,
        device_id=device_id,
        state=state,
        language=language,
        database_config=db_config.to_dict()
    )
    
    # Process the request using the same logic as the main query endpoint
    async def process_answer():
        try:
            answer_processing_start = time.time()
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                raw_answer = await loop.run_in_executor(
                    executor, 
                    get_answer, 
                    request.question, 
                    None,
                    request.state,
                    session_id,
                    db_config
                )
            answer_processing_time = time.time() - answer_processing_start
            logger.info(f"[TIMING] Answer processing took: {answer_processing_time:.3f}s")
            return str(raw_answer).strip()
        except Exception as e:
            logger.error(f"[DEBUG] Error in get_answer: {e}")
            raise e

    async def process_recommendations():
        if DISABLE_RECOMMENDATIONS:
            return []
        try:
            recommendations_start = time.time()
            recommendations = get_question_recommendations(request.question, request.state, 2)
            recommendations_time = time.time() - recommendations_start
            logger.info(f"[TIMING] Recommendations took: {recommendations_time:.3f}s")
            return recommendations[:2]
        except Exception as e:
            logger.error(f"[DEBUG] Error getting recommendations: {e}")
            return []

    try:
        if PARALLEL_RECOMMENDATIONS:
            answer_task = asyncio.create_task(process_answer())
            recommendations_task = asyncio.create_task(process_recommendations())
            answer, recommendations = await asyncio.gather(answer_task, recommendations_task)
        else:
            answer = await process_answer()
            recommendations = await process_recommendations()

        # Create session record
        session = {
            "session_id": session_id,
            "device_id": request.device_id,
            "question": request.question,
            "answer": answer,
            "state": request.state,
            "language": request.language,
            "timestamp": datetime.now(IST).isoformat(),
            "status": "active",
            "recommendations": recommendations[:2],
            "database_config": {
                "golden_db_enabled": golden_db,
                "rag_db_enabled": rag_db,
                "pops_db_enabled": pops_db,
                "llm_fallback_enabled": llm_fallback,
                "mode": "traditional" if db_config.is_traditional_mode() else "selective",
                "enabled_databases": db_config.get_enabled_databases()
            },
            "total_processing_time": time.time() - api_start_time
        }

        sessions_collection.insert_one(session.copy())

        api_total_time = time.time() - api_start_time
        logger.info(f"[TIMING] Total API request took: {api_total_time:.3f}s")

        return {"session": session}

    except Exception as e:
        logger.error(f"[DEBUG] Error in new_session_form: {e}")
        error_session = {
            "session_id": session_id,
            "device_id": request.device_id,
            "question": request.question,
            "answer": "Sorry, I encountered an error. Please try again.",
            "state": request.state,
            "language": request.language,
            "timestamp": datetime.now(IST).isoformat(),
            "status": "active",
            "recommendations": [],
            "error": str(e)
        }
        return {"session": error_session}

@app.post("/api/query-legacy")
async def new_session_legacy(question: str = Form(...), device_id: str = Form(...), state: str = Form(...), language: str = Form(...)):
    """Legacy endpoint for backward compatibility"""
    request = QueryRequest(
        question=question,
        device_id=device_id,
        state=state,
        language=language,
        database_config=None
    )
    return await new_session(request)
    
    # Create tasks for parallel execution
    tasks = [process_answer()]
    
    # Add recommendations task if enabled and parallel processing is on
    if not DISABLE_RECOMMENDATIONS and PARALLEL_RECOMMENDATIONS:
        tasks.append(get_question_recommendations_async(request.question, request.state, 2))
    
    try:
        # Execute tasks in parallel
        if len(tasks) > 1:
            parallel_start = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            parallel_time = time.time() - parallel_start
            logger.info(f"[TIMING] Parallel processing took: {parallel_time:.3f}s")
            
            # Handle results
            answer_only = results[0]
            if isinstance(answer_only, Exception):
                raise answer_only
                
            recommendations = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else []
            if isinstance(results[1], Exception):
                logger.error(f"Parallel recommendations failed: {results[1]}")
                recommendations = []
        else:
            # Sequential processing if parallel is disabled
            answer_only = await tasks[0]
            recommendations = []
    except Exception as e:
        logger.error(f"[DEBUG] Error in parallel processing: {e}")
        return JSONResponse(status_code=500, content={"error": "LLM processing failed."})

    # Optimized markdown processing - reduced logging
    markdown_processing_start = time.time()
    html_answer = markdown.markdown(answer_only, extensions=["extra", "nl2br"])
    markdown_processing_time = time.time() - markdown_processing_start
    logger.info(f"[TIMING] Markdown processing took: {markdown_processing_time:.3f}s")

    session_creation_start = time.time()
    session = {
        "session_id": session_id,
        "timestamp": datetime.now(IST).isoformat(),
        "messages": [{"question": request.question, "answer": html_answer, "rating": None}],
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
    
    # Handle recommendations - either from parallel processing or sequential
    recommendations_start = time.time()
    if not DISABLE_RECOMMENDATIONS:
        if not PARALLEL_RECOMMENDATIONS:
            # Sequential recommendations if parallel is disabled
            try:
                recommendations = get_question_recommendations(
                    user_question=request.question,
                    user_state=request.state,
                    limit=2
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

    # Start both answer processing and recommendations in parallel if enabled
    async def process_session_answer():
        try:
            # Enhanced conversation history with context resolution
            conversation_history = []
            messages = session.get("messages", [])
            
            current_state = request.state or session.get("state", "unknown")
            
            # Run answer processing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                raw_answer = await loop.run_in_executor(
                    executor, 
                    get_answer, 
                    request.question, 
                    [],  # conversation_history
                    current_state,
                    session_id,  # Pass session_id for context resolution
                    db_config
                )
            return str(raw_answer).strip()
            
        except Exception as e:
            logger.error(f"[DEBUG] Error in get_answer: {e}")
            raise e
    
    # Create tasks for parallel execution
    tasks = [process_session_answer()]
    
    # Add recommendations task if enabled and parallel processing is on
    current_state = request.state or session.get("state", "unknown")
    if not DISABLE_RECOMMENDATIONS and PARALLEL_RECOMMENDATIONS:
        logger.info("[PARALLEL] Starting session recommendations in parallel with answer processing")
        tasks.append(get_question_recommendations_async(request.question, current_state, 2))
    
    try:
        # Execute tasks in parallel
        if len(tasks) > 1:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results
            answer_only = results[0]
            if isinstance(answer_only, Exception):
                raise answer_only
                
            recommendations = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else []
            if isinstance(results[1], Exception):
                logger.error(f"Parallel session recommendations failed: {results[1]}")
                recommendations = []
        else:
            # Sequential processing if parallel is disabled
            answer_only = await tasks[0]
            recommendations = []
            
    except Exception as e:
        logger.error(f"[DEBUG] Error in parallel session processing: {e}")
        return JSONResponse(status_code=500, content={"error": "LLM processing failed."})

    html_answer = markdown.markdown(answer_only, extensions=["extra", "nl2br"])

    crop = session.get("crop", "unknown")
    state = state or session.get("state", "unknown")

    sessions_collection.update_one(
        {"session_id": session_id},
        {
            "$push": {"messages": {"question": request.question, "answer": html_answer, "rating": None}},
            "$set": {
                "has_unread": True,
                "crop": crop,
                "state": state,
                "timestamp": datetime.now(IST).isoformat()
            },
        },
    )

    updated = sessions_collection.find_one({"session_id": session_id})
    if updated:
        updated.pop("_id", None)
        
        # Handle recommendations - either from parallel processing or sequential
        if not DISABLE_RECOMMENDATIONS:
            if not PARALLEL_RECOMMENDATIONS:
                # Sequential recommendations if parallel is disabled
                try:
                    recommendations = get_question_recommendations(
                        user_question=request.question,
                        user_state=current_state,
                        limit=2
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

@app.post("/api/session/{session_id}/query-form")
async def continue_session_form(
    session_id: str,
    question: str = Form(...),
    device_id: str = Form(...),
    state: str = Form(""),
    golden_db: bool = Form(False),
    rag_db: bool = Form(True), 
    pops_db: bool = Form(False),
    llm_fallback: bool = Form(False)
):
    """Form-based session continuation endpoint with database toggle support"""
    session = sessions_collection.find_one({"session_id": session_id})
    if not session or session.get("status") == "archived" or session.get("device_id") != device_id:
        return JSONResponse(status_code=403, content={"error": "Session is archived, missing or unauthorized"})

    # Create database configuration
    db_config = DatabaseConfig(
        golden_db_enabled=golden_db,
        rag_db_enabled=rag_db,
        pops_db_enabled=pops_db,
        llm_fallback_enabled=llm_fallback
    )
    
    logger.info(f"[DB_CONFIG] Session continuation with database config: {db_config.get_enabled_databases()}")

    # Start both answer processing and recommendations in parallel if enabled
    async def process_session_answer():
        try:
            current_state = state or session.get("state", "unknown")
            
            # Run answer processing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                raw_answer = await loop.run_in_executor(
                    executor, 
                    get_answer, 
                    question, 
                    [],  # conversation_history
                    current_state,
                    session_id,  # Pass session_id for context resolution
                    db_config
                )
            return str(raw_answer).strip()
            
        except Exception as e:
            logger.error(f"[DEBUG] Error in get_answer: {e}")
            raise e
    
    # Create tasks for parallel execution
    tasks = [process_session_answer()]
    
    # Add recommendations task if enabled and parallel processing is on
    current_state = state or session.get("state", "unknown")
    if not DISABLE_RECOMMENDATIONS and PARALLEL_RECOMMENDATIONS:
        logger.info("[PARALLEL] Starting session recommendations in parallel with answer processing")
        tasks.append(get_question_recommendations_async(question, current_state, 2))
    
    try:
        # Execute answer processing and optionally recommendations in parallel
        if len(tasks) > 1:
            answer_only, recommendations = await asyncio.gather(*tasks)
        else:
            answer_only = await tasks[0]
            recommendations = []
    except Exception as e:
        logger.error(f"[DEBUG] Error in parallel session processing: {e}")
        return JSONResponse(status_code=500, content={"error": "LLM processing failed."})

    html_answer = markdown.markdown(answer_only, extensions=["extra", "nl2br"])

    crop = session.get("crop", "unknown")
    current_state = state or session.get("state", "unknown")

    sessions_collection.update_one(
        {"session_id": session_id},
        {
            "$push": {"messages": {"question": question, "answer": html_answer, "rating": None}},
            "$set": {
                "has_unread": True,
                "crop": crop,
                "state": current_state,
                "timestamp": datetime.now(IST).isoformat(),
                "database_config": {
                    "golden_db_enabled": golden_db,
                    "rag_db_enabled": rag_db,
                    "pops_db_enabled": pops_db,
                    "llm_fallback_enabled": llm_fallback,
                    "mode": "traditional" if db_config.is_traditional_mode() else "selective",
                    "enabled_databases": db_config.get_enabled_databases()
                }
            },
        },
    )

    updated = sessions_collection.find_one({"session_id": session_id})
    if updated:
        updated.pop("_id", None)
        
        # Handle recommendations - either from parallel processing or sequential
        if not DISABLE_RECOMMENDATIONS:
            if not PARALLEL_RECOMMENDATIONS:
                # Sequential recommendations if parallel is disabled
                try:
                    recommendations = get_question_recommendations(
                        user_question=question,
                        user_state=current_state,
                        limit=2
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
            raw_answer = get_answer(question, conversation_history=None, user_state=state)
            answer_only = str(raw_answer).strip()
            html_answer = markdown.markdown(answer_only, extensions=["extra", "nl2br"])
            
            session_id = str(uuid4())
            session = {
                "session_id": session_id,
                "timestamp": datetime.now(IST).isoformat(),
                "messages": [{"question": question, "answer": html_answer, "rating": None}],
                "crop": "unknown",
                "state": state,
                "status": "active",
                "language": language,
                "has_unread": True,
                "device_id": device_id
            }
            
            sessions_collection.insert_one(session)
            session.pop("_id", None)
            
            yield f"data: {json.dumps({'type': 'complete', 'session': session})}\n\n"
            
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

        transcript = local_whisper.transcribe_audio(contents, file.filename)

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

# OLD TRANSCRIBE-AUDIO
# HF_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3" # using this whisper model
# HF_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-tiny"
# HF_API_TOKEN = os.getenv("HF_API_TOKEN")  

# @app.post("/api/transcribe-audio")
# async def transcribe_audio(file: UploadFile = File(...)):
#     try:
#         logger.info(f"Received file: {file.filename}")

#         contents = await file.read()
#         logger.info(f"File size: {len(contents)} bytes")

#         content_type = file.content_type
#         if file.filename:
#             if file.filename.lower().endswith('.wav'):
#                 content_type = "audio/wav"
#             elif file.filename.lower().endswith('.mp3'):
#                 content_type = "audio/mpeg"
#             elif file.filename.lower().endswith('.flac'):
#                 content_type = "audio/flac"
#             elif file.filename.lower().endswith('.ogg'):
#                 content_type = "audio/ogg"
#             elif file.filename.lower().endswith('.m4a'):
#                 content_type = "audio/m4a"
#             elif file.filename.lower().endswith('.webm'):
#                 content_type = "audio/webm"

#         logger.info(f"Using content type: {content_type}")

#         headers = {
#             "Content-Type": content_type,
#         }

#         if HF_API_TOKEN and HF_API_TOKEN.strip():
#             headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
#             logger.info("Using authenticated request")
#         else:
#             logger.info("Trying without authentication (public model)")

#         logger.info(f"Transcribing with whisper-tiny model")

#         response = requests.post(HF_API_URL, headers=headers, data=contents, timeout=60)

#         logger.info(f"HF Status: {response.status_code}")
#         logger.info(f"HF Response: {response.text[:500]}...")

#         if response.status_code == 200:
#             try:
#                 result = response.json()
#                 transcript = result.get("text") or result.get("generated_text")

#                 if transcript and transcript.strip():
#                     logger.info(f"Transcription successful: {transcript[:100]}...")
#                     return {"transcript": transcript.strip()}
#                 else:
#                     logger.warning(f"No transcript in response: {result}")
#                     return JSONResponse(
#                         status_code=500,
#                         content={"error": "No transcript found in response", "raw": str(result)}
#                     )
#             except Exception as json_error:
#                 logger.error(f"Failed to parse JSON response: {json_error}")
#                 return JSONResponse(
#                     status_code=500,
#                     content={"error": "Invalid JSON response from Hugging Face", "raw_response": response.text[:500]}
#                 )

#         elif response.status_code == 503:
#             return JSONResponse(
#                 status_code=200,
#                 content={
#                     "transcript": f"Audio Processing in Progress! Your audio file '{file.filename}' has been received. Our speech recognition service is currently starting up. Please wait 30-60 seconds and try uploading your audio again. Thank you for your patience!",
#                     "demo_mode": True,
#                     "api_status": "working",
#                     "model_status": "loading_503",
#                     "file_info": {
#                         "filename": file.filename,
#                         "size_bytes": len(contents),
#                         "content_type": content_type
#                     },
#                     "retry_suggestion": "Please wait 30-60 seconds and try again"
#                 }
#             )

#         elif response.status_code == 404:
#             logger.warning("Whisper model returned 404 - not available")
#             return JSONResponse(
#                 status_code=200,
#                 content={
#                     "transcript": f"Audio Successfully Received! Your audio file '{file.filename}' has been uploaded and processed. However, our speech-to-text service is temporarily unavailable. Please try again in a few minutes, or contact support if the issue persists. We apologize for the inconvenience!",
#                     "demo_mode": True,
#                     "api_status": "working",
#                     "model_status": "unavailable_404", 
#                     "file_info": {
#                         "filename": file.filename,
#                         "size_bytes": len(contents),
#                         "content_type": content_type
#                     },
#                     "debug_info": {
#                         "hf_status_code": response.status_code,
#                         "hf_response": response.text[:100],
#                         "model_url": HF_API_URL,
#                         "token_provided": bool(HF_API_TOKEN and HF_API_TOKEN.strip())
#                     },
#                     "next_steps": "The speech recognition service is temporarily unavailable. Please try again in a few minutes."
#                 }
#             )

#         elif response.status_code == 429:
#             return JSONResponse(
#                 status_code=200,
#                 content={
#                     "transcript": f"Processing Limit Reached! Your audio file '{file.filename}' has been received successfully. However, we're currently processing many requests. Please wait a few minutes and try again. We appreciate your patience!",
#                     "demo_mode": True,
#                     "api_status": "working",
#                     "model_status": "rate_limited_429",
#                     "file_info": {
#                         "filename": file.filename,
#                         "size_bytes": len(contents),
#                         "content_type": content_type
#                     },
#                     "retry_suggestion": "Please wait 3-5 minutes before trying again"
#                 }
#             )

#         else:
#             logger.error(f"Unexpected status code: {response.status_code}")
#             logger.error(f"Response content: {response.text}")
#             return JSONResponse(
#                 status_code=502,
#                 content={
#                     "error": f"Hugging Face API error: {response.status_code}",
#                     "details": response.text[:300],
#                     "debug_info": f"Headers sent: {dict(headers)}"
#                 }
#             )

#     except requests.exceptions.Timeout:
#         logger.error("Request timed out")
#         return JSONResponse(
#             status_code=200,
#             content={
#                 "transcript": f"Processing Timeout! Your audio file '{file.filename}' was received, but the transcription service is taking longer than expected. This might be due to high server load. Please try again in a few minutes.",
#                 "demo_mode": True,
#                 "api_status": "timeout",
#                 "model_status": "timeout_504",
#                 "file_info": {
#                     "filename": file.filename,
#                     "size_bytes": len(contents),
#                     "content_type": content_type if 'content_type' in locals() else "unknown"
#                 },
#                 "retry_suggestion": "Please try again in 2-3 minutes"
#             }
#         )
#     except requests.exceptions.ConnectionError:
#         logger.error("Connection error")
#         return JSONResponse(
#             status_code=200,
#             content={
#                 "transcript": f"Connection Issue! Your audio file '{file.filename}' was received, but we're having trouble connecting to our speech recognition service. Please check your internet connection and try again.",
#                 "demo_mode": True,
#                 "api_status": "connection_error",
#                 "model_status": "connection_failed_503",
#                 "file_info": {
#                     "filename": file.filename,
#                     "size_bytes": len(contents),
#                     "content_type": content_type if 'content_type' in locals() else "unknown"
#                 },
#                 "retry_suggestion": "Please check your internet connection and try again"
#             }
#         )
#     except Exception as e:
#         logger.exception("Transcription failed with unexpected error")
#         return JSONResponse(
#             status_code=200, 
#             content={
#                 "transcript": f"Unexpected Error! Your audio file '{file.filename}' was received, but we encountered an unexpected issue during processing. Our technical team has been notified. Please try again later.",
#                 "demo_mode": True,
#                 "api_status": "error",
#                 "model_status": "internal_error_500",
#                 "file_info": {
#                     "filename": file.filename if 'file' in locals() and hasattr(file, 'filename') else "unknown",
#                     "size_bytes": len(contents) if 'contents' in locals() else 0,
#                     "content_type": content_type if 'content_type' in locals() else "unknown"
#                 },
#                 "error_details": str(e),
#                 "retry_suggestion": "Please try again later or contact support if the issue persists"
#             }
#         )

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
        if db_config.is_traditional_mode():
            response = get_fast_answer(question, None, "unknown", None)
        else:
            response = get_fast_answer(question, None, "unknown", db_config)
        
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
