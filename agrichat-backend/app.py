from fastapi import FastAPI, Request, Form, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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
from dotenv import load_dotenv
import logging
from typing import Optional, List, Dict
import requests
from local_whisper_interface import local_whisper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.schema import HumanMessage, AIMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")
current_dir = os.path.dirname(os.path.abspath(__file__))
agentic_rag_path = os.path.join(current_dir, "Agentic_RAG")
sys.path.insert(0, agentic_rag_path)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

USE_FAST_MODE = os.getenv("USE_FAST_MODE", "false").lower() == "true"
logger.info(f"[CONFIG] USE_FAST_MODE environment variable: {os.getenv('USE_FAST_MODE', 'not set')}")
logger.info(f"[CONFIG] Fast Mode Enabled: {USE_FAST_MODE}")

fast_handler = None
if USE_FAST_MODE:
    try:
        from fast_response_handler import FastResponseHandler
        fast_handler = FastResponseHandler()
        logger.info("[CONFIG] Fast response handler loaded successfully")
    except ImportError as e:
        logger.warning(f"[CONFIG] Fast response handler not available: {e}")
        logger.info("[CONFIG] Falling back to CrewAI mode")
        USE_FAST_MODE = False

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

chroma_db_path = os.path.join(current_dir, "chromaDb")
logger.info(f"[DEBUG] ChromaDB path: {chroma_db_path}")
logger.info(f"[DEBUG] ChromaDB path exists: {os.path.exists(chroma_db_path)}")

query_handler = ChromaQueryHandler(chroma_path=chroma_db_path)

# Session-based memory store using LangChain
session_memories = {}

def get_session_memory(session_id: str) -> ConversationBufferWindowMemory:
    """Get or create conversation memory for a session"""
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferWindowMemory(
            k=5,  # Keep last 5 exchanges
            return_messages=True
        )
    return session_memories[session_id]

def format_conversation_context(memory: ConversationBufferWindowMemory) -> str:
    """Format conversation history for RAG context"""
    if not memory.chat_memory.messages:
        return "This is the start of the conversation."
    
    context_parts = []
    messages = memory.chat_memory.messages[-10:]  # Last 10 messages (5 exchanges)
    
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            human_msg = messages[i]
            ai_msg = messages[i + 1]
            context_parts.append(f"Previous Q: {human_msg.content}")
            context_parts.append(f"Previous A: {ai_msg.content[:200]}...")  # Truncate long responses
    
    return "\n".join(context_parts)

def get_answer(question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = None, session_id: str = None) -> str:
    """
    Main answer function that routes to fast mode or CrewAI based on configuration.
    
    Args:
        question: Current user question
        conversation_history: List of previous Q&A pairs for context (legacy)
        user_state: User's state/region detected from frontend location
        session_id: Session ID for memory management
    """
    # Get session memory for context
    if session_id:
        memory = get_session_memory(session_id)
        context_str = format_conversation_context(memory)
        
        # Check if this is a follow-up question that needs context resolution
        resolved_question = resolve_context_question(question, context_str)
        if resolved_question != question:
            logger.info(f"[CONTEXT] Resolved '{question}' to '{resolved_question}'")
            question = resolved_question
    
    if USE_FAST_MODE and fast_handler:
        response = get_fast_answer(question, conversation_history, user_state)
    else:
        response = get_crewai_answer(question, conversation_history, user_state)
    
    # Store in memory if session_id provided
    if session_id:
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(response)
    
    return response

def resolve_context_question(question: str, context: str) -> str:
    """
    Resolve ambiguous questions using conversation context
    """
    question_lower = question.lower().strip()
    
    # Common follow-up patterns that need context resolution
    context_patterns = [
        "how do i cure it", "how to cure it", "cure it", "treat it", "how to treat it",
        "what should i do", "how to fix it", "fix it", "prevent it", "how to prevent it",
        "what medicine", "which medicine", "what treatment", "what chemical", "what spray",
        "how much", "when to apply", "when should i", "how often", "what dosage",
        "side effects", "precautions", "what about", "tell me more", "more details"
    ]
    
    if any(pattern in question_lower for pattern in context_patterns):
        # Extract the most recent topic from context
        if "late blight" in context.lower():
            if "cure" in question_lower or "treat" in question_lower:
                return "How to cure late blight in potato?"
            elif "prevent" in question_lower:
                return "How to prevent late blight in potato?"
            elif "medicine" in question_lower or "chemical" in question_lower:
                return "What medicines or chemicals to use for late blight in potato?"
        
        # Add more context resolution patterns as needed
        recent_topics = extract_topics_from_context(context)
        if recent_topics:
            return f"{question} for {recent_topics[0]}"
    
    return question

def extract_topics_from_context(context: str) -> List[str]:
    """Extract agricultural topics from conversation context"""
    topics = []
    context_lower = context.lower()
    
    # Common crop diseases and pests
    disease_patterns = [
        "late blight", "early blight", "powdery mildew", "downy mildew", 
        "bacterial wilt", "fungal infection", "leaf spot", "root rot"
    ]
    
    crop_patterns = [
        "potato", "tomato", "wheat", "rice", "cotton", "sugarcane", "maize", "corn"
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
    
    return topics

def get_fast_answer(question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = None) -> str:
    """
    Fast mode using FastResponseHandler for 10-15x faster responses.
    """
    logger.info(f"[FAST] Processing question with fast handler: {question}")
    try:
        return fast_handler.get_answer(question, conversation_history, user_state)
    except Exception as e:
        logger.error(f"[FAST] Error in fast mode: {e}")
        logger.info("[FAST] Falling back to CrewAI")
        return get_crewai_answer(question, conversation_history, user_state)

def get_crewai_answer(question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = None) -> str:
    """
    CrewAI mode - original multi-agent approach for complex reasoning.
    """
    logger.info(f"[CREWAI] Processing question with CrewAI approach: {question}")
    
    question_lower = question.lower().strip()
    simple_greetings = [
        'hi', 'hello', 'hey', 'namaste', 'namaskaram', 'vanakkam', 
        'good morning', 'good afternoon', 'good evening', 'good day',
        'howdy', 'greetings', 'salaam', 'adaab', 'hi there', 'hello there'
    ]
    
    if len(question_lower) < 20 and any(greeting in question_lower for greeting in simple_greetings):
        logger.info(f"[GREETING] Detected simple greeting: {question}")
        logger.info(f"[SOURCE] Fast pattern matching used for greeting: {question}")
        state_context = f" in {user_state}" if user_state and user_state.lower() != "unknown" else " in India"
        if 'namaste' in question_lower:
            return f"Namaste! Welcome to AgriChat, your trusted agricultural assistant for Indian farming{state_context}. I specialize in crop management, soil health, weather patterns, and farming practices specific to Indian conditions. What agricultural challenge can I help you with today?"
        elif 'namaskaram' in question_lower:
            return f"Namaskaram! I'm your specialized agricultural assistant for Indian farmers{state_context}. Feel free to ask me about Indian crop varieties, monsoon farming, soil management, or any farming techniques suited to Indian climate and conditions."
        elif 'vanakkam' in question_lower:
            return f"Vanakkam! I'm here to assist you with Indian farming and agriculture{state_context}. What regional agricultural topic would you like to discuss - from rice cultivation to spice farming, I'm here to help with India-specific guidance!"
        elif any(time in question_lower for time in ['morning', 'afternoon', 'evening']):
            time_word = next(time for time in ['morning', 'afternoon', 'evening'] if time in question_lower)
            return f"Good {time_word}! I'm your agricultural assistant specializing in Indian farming practices{state_context}. How can I help you with your crop management, seasonal farming, or any agriculture-related questions specific to Indian conditions today?"
        else:
            return f"Hello! I'm your agricultural assistant specializing in Indian farming and crop management{state_context}. I'm here to help with crops, farming techniques, and agricultural practices tailored to Indian soil, climate, and regional conditions. What would you like to know?"
    
    if conversation_history:
        logger.info(f"[DEBUG] Using conversation context with {len(conversation_history)} previous interactions")
    if user_state:
        logger.info(f"[DEBUG] Using frontend-detected state: {user_state}")
    
    try:
        rag_crew = Crew(
            agents=[
                Retriever_Agent
            ],
            tasks=[
                retriever_task
            ],
            verbose=True,
        )
        
        inputs = {
            "question": question,
            "conversation_history": conversation_history or []
        }
        
        result = rag_crew.kickoff(inputs=inputs)
        logger.info(f"[DEBUG] CrewAI result: {result}")
        
        result_str = str(result).strip()
        if "Source: RAG Database" in result_str:
            logger.info(f"[SOURCE] RAG Database used for question: {question[:50]}...")
            result_str = result_str.replace("Source: RAG Database", "").strip()
        if "Source: Local LLM" in result_str:
            logger.info(f"[SOURCE] Local LLM used for question: {question[:50]}...")
            result_str = result_str.replace("Source: Local LLM", "").strip()
        
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
    
    Args:
        user_question: The user's current question
        user_state: User's state for state-specific recommendations
        limit: Number of recommendations to return (default 2)
    
    Returns:
        List of recommended questions with their details (empty if not agriculture-related)
    """
    try:
        question_category = query_handler.classify_query(user_question)
        if question_category != "AGRICULTURE":
            logger.info(f"Question classified as {question_category}, skipping recommendations: {user_question}")
            return []
        
        pipeline = [
            {"$unwind": "$messages"},
            {"$group": {
                "_id": {
                    "question": "$messages.question",
                    "state": "$state"
                },
                "count": {"$sum": 1},
                "sample_answer": {"$first": "$messages.answer"}
            }},
            {"$match": {"count": {"$gte": 1}}},
            {"$project": {
                "question": "$_id.question",
                "state": "$_id.state",
                "count": 1,
                "sample_answer": 1
            }},
            {"$limit": 200}
        ]
        
        questions_data = list(sessions_collection.aggregate(pipeline))
        
        if not questions_data:
            logger.info("No questions found in database for recommendations")
            return []
        
        processed_user_question = preprocess_question(user_question)
        
        questions_list = []
        metadata_list = []
        
        for item in questions_data:
            question = item['question']
            question_category = query_handler.classify_query(question)
            if question_category != "AGRICULTURE":
                continue
                
            processed_question = preprocess_question(question)
            
            if processed_question == processed_user_question:
                continue
                
            questions_list.append(processed_question)
            metadata_list.append({
                'original_question': question,
                'state': item.get('state', 'unknown'),
                'count': item['count'],
                'sample_answer': item['sample_answer']
            })
        
        if not questions_list:
            logger.info("No suitable questions found for recommendations")
            return []
        
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        all_questions = [processed_user_question] + questions_list
        tfidf_matrix = vectorizer.fit_transform(all_questions)
        
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        recommendations = []
        for i, similarity_score in enumerate(similarities):
            if similarity_score > 0.1:
                rec = {
                    'question': metadata_list[i]['original_question'],
                    'state': metadata_list[i]['state'],
                    'similarity_score': float(similarity_score),
                    'popularity': metadata_list[i]['count'],
                    'sample_answer': metadata_list[i]['sample_answer'][:200] + "..." 
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
        
        logger.info(f"Found {len(top_recommendations)} recommendations for question: {user_question[:50]}...")
        
        return top_recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return []

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] App initialized.")
    yield
    print("[Shutdown] App shutting down...")

app = FastAPI(lifespan=lifespan)

origins = [
    "https://agri-annam.vercel.app",
    "https://f7387713c09d.ngrok-free.app",
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
async def new_session(question: str = Form(...), device_id: str = Form(...), state: str = Form(...), language: str = Form(...)):
    session_id = str(uuid4())
    try:
        raw_answer = get_answer(question, conversation_history=None, user_state=state, session_id=session_id)
        logger.info(f"[DEBUG] Raw answer: {raw_answer}")
        logger.info(f"[DEBUG] Raw answer type: {type(raw_answer)}")
        
        answer_only = str(raw_answer).strip()
        
    except Exception as e:
        logger.error(f"[DEBUG] Error in get_answer: {e}")
        return JSONResponse(status_code=500, content={"error": "LLM processing failed."})

    logger.info(f"[DEBUG] Answer after processing: {answer_only}")
    html_answer = markdown.markdown(answer_only, extensions=["extra", "nl2br"])
    logger.info(f"[DEBUG] HTML answer: {html_answer}")

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
    
    # Get recommendations for this question
    try:
        recommendations = get_question_recommendations(
            user_question=question,
            user_state=state,
            limit=2
        )
        session["recommendations"] = recommendations
        logger.info(f"Added {len(recommendations)} recommendations to session response")
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        session["recommendations"] = []
    
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
async def continue_session(session_id: str, question: str = Form(...), device_id: str = Form(...), state: str = Form("")):
    session = sessions_collection.find_one({"session_id": session_id})
    if not session or session.get("status") == "archived" or session.get("device_id") != device_id:
        return JSONResponse(status_code=403, content={"error": "Session is archived, missing or unauthorized"})

    try:
        conversation_history = []
        messages = session.get("messages", [])
        
        for msg in messages[-5:]:
            if "question" in msg and "answer" in msg:
                from bs4 import BeautifulSoup
                clean_answer = BeautifulSoup(msg["answer"], "html.parser").get_text()
                conversation_history.append({
                    "question": msg["question"],
                    "answer": clean_answer
                })
        
        # Initialize session memory with existing conversation history if not already done
        if session_id not in session_memories:
            memory = get_session_memory(session_id)
            
            # Load existing conversation into memory
            for msg in messages:
                if "question" in msg and "answer" in msg:
                    clean_answer = BeautifulSoup(msg["answer"], "html.parser").get_text()
                    memory.chat_memory.add_user_message(msg["question"])
                    memory.chat_memory.add_ai_message(clean_answer)
        
        logger.info(f"[DEBUG] Using conversation context: {len(conversation_history)} previous interactions")
        
        current_state = state or session.get("state", "unknown")
        
        # Use the new session memory system
        raw_answer = get_answer(question, conversation_history=conversation_history, user_state=current_state, session_id=session_id)
    except Exception as e:
        logger.error(f"[DEBUG] Error in get_answer: {e}")
        return JSONResponse(status_code=500, content={"error": "LLM processing failed."})

    answer_only = str(raw_answer).strip()
    html_answer = markdown.markdown(answer_only, extensions=["extra", "nl2br"])

    crop = session.get("crop", "unknown")
    state = state or session.get("state", "unknown")

    sessions_collection.update_one(
        {"session_id": session_id},
        {
            "$push": {"messages": {"question": question, "answer": html_answer, "rating": None}},
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
        
        # Get recommendations for the latest question
        try:
            recommendations = get_question_recommendations(
                user_question=question,
                user_state=state,
                limit=2
            )
            updated["recommendations"] = recommendations
            logger.info(f"Added {len(recommendations)} recommendations to continue session response")
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            updated["recommendations"] = []
    
    return {"session": updated}

# Streaming endpoint for better UX
@app.post("/api/query/stream")
async def stream_query(question: str = Form(...), device_id: str = Form(...), state: str = Form(...), language: str = Form(...)):
    """
    Streaming endpoint that shows thinking progress
    """
    async def generate_thinking_stream():
        import json
        import asyncio
        
        # Send thinking states
        thinking_states = [
            "Understanding your question...",
            "Searching agricultural database...", 
            "Processing with AI...",
            "Generating response..."
        ]
        
        for i, state in enumerate(thinking_states):
            yield f"data: {json.dumps({'type': 'thinking', 'message': state, 'progress': (i + 1) * 25})}\n\n"
            await asyncio.sleep(0.5)  # Small delay for better UX
        
        try:
            # Get the actual answer
            raw_answer = get_answer(question, conversation_history=None, user_state=state)
            answer_only = str(raw_answer).strip()
            html_answer = markdown.markdown(answer_only, extensions=["extra", "nl2br"])
            
            # Send final response
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

        # Transcribe using Local Whisper (GPU-accelerated)
        transcript = local_whisper.transcribe_audio(contents, file.filename)

        # Check if transcription failed
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
        
        # Get recommendations
        recommendations = get_question_recommendations(
            user_question=user_question,
            user_state=user_state,
            limit=min(limit, 5)  # Max 5 recommendations
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
