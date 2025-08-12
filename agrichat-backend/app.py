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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

current_dir = os.path.dirname(os.path.abspath(__file__))
agentic_rag_path = os.path.join(current_dir, "Agentic_RAG")
sys.path.insert(0, agentic_rag_path)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from Agentic_RAG.crew_agents import retriever_response

chroma_db_path = os.path.join(current_dir, "Agentic_RAG", "chromaDb")
logger.info(f"[DEBUG] ChromaDB path: {chroma_db_path}")
logger.info(f"[DEBUG] ChromaDB path exists: {os.path.exists(chroma_db_path)}")

def get_answer(question: str, conversation_history: Optional[List[Dict]] = None) -> str:
    """
    Use the same logic as main.py - direct retriever_response function:
    1. Try RAG tool first with optional conversation context
    2. If RAG returns __FALLBACK__, use fallback tool
    
    Args:
        question: Current user question
        conversation_history: List of previous Q&A pairs for context
    """
    logger.info(f"[DEBUG] Processing question with retriever_response approach: {question}")
    if conversation_history:
        logger.info(f"[DEBUG] Using conversation context with {len(conversation_history)} previous interactions")
    
    try:
        response = retriever_response(question, conversation_history)
        logger.info(f"[DEBUG] Retriever response: {response}")
        return response
            
    except Exception as e:
        logger.error(f"[DEBUG] Error in retriever_response: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please try again later."

logger.info(f"[DEBUG] Using retriever_response function same as main.py approach")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] App initialized.")
    yield
    print("[Shutdown] App shutting down...")

app = FastAPI(lifespan=lifespan)

origins = [
    "https://agri-annam.vercel.app",
    "https://68ebe24fbd01.ngrok-free.app",
    "https://localhost:3000",
    "https://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.middleware("http")
async def add_ngrok_headers(request: Request, call_next):
    if request.method == "OPTIONS":
        response = JSONResponse(content={})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Max-Age"] = "86400"
        response.headers["ngrok-skip-browser-warning"] = "true"
        return response
    
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
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
        # No conversation history for new sessions
        raw_answer = get_answer(question, conversation_history=None)
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
        # Extract conversation history for context-aware responses
        conversation_history = []
        messages = session.get("messages", [])
        
        # Convert stored messages to context format (limit to last 5 for efficiency)
        for msg in messages[-5:]:  # Keep last 5 interactions
            if "question" in msg and "answer" in msg:
                # Remove HTML tags from answer for cleaner context
                from bs4 import BeautifulSoup
                clean_answer = BeautifulSoup(msg["answer"], "html.parser").get_text()
                conversation_history.append({
                    "question": msg["question"],
                    "answer": clean_answer
                })
        
        logger.info(f"[DEBUG] Using conversation context: {len(conversation_history)} previous interactions")
        
        raw_answer = get_answer(question, conversation_history=conversation_history)
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
    return {"session": updated}

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


HF_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-tiny"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  

@app.post("/api/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")

        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")

        content_type = file.content_type
        if file.filename:
            if file.filename.lower().endswith('.wav'):
                content_type = "audio/wav"
            elif file.filename.lower().endswith('.mp3'):
                content_type = "audio/mpeg"
            elif file.filename.lower().endswith('.flac'):
                content_type = "audio/flac"
            elif file.filename.lower().endswith('.ogg'):
                content_type = "audio/ogg"
            elif file.filename.lower().endswith('.m4a'):
                content_type = "audio/m4a"
            elif file.filename.lower().endswith('.webm'):
                content_type = "audio/webm"
        
        logger.info(f"Using content type: {content_type}")

        headers = {
            "Content-Type": content_type,
        }

        if HF_API_TOKEN and HF_API_TOKEN.strip():
            headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
            logger.info("Using authenticated request")
        else:
            logger.info("Trying without authentication (public model)")

        logger.info(f"Transcribing with whisper-tiny model")
        
        response = requests.post(HF_API_URL, headers=headers, data=contents, timeout=60)
        
        logger.info(f"HF Status: {response.status_code}")
        logger.info(f"HF Response: {response.text[:500]}...")
        
        if response.status_code == 200:
            try:
                result = response.json()
                transcript = result.get("text") or result.get("generated_text")
                
                if transcript and transcript.strip():
                    logger.info(f"Transcription successful: {transcript[:100]}...")
                    return {"transcript": transcript.strip()}
                else:
                    logger.warning(f"No transcript in response: {result}")
                    return JSONResponse(
                        status_code=500,
                        content={"error": "No transcript found in response", "raw": str(result)}
                    )
            except Exception as json_error:
                logger.error(f"Failed to parse JSON response: {json_error}")
                return JSONResponse(
                    status_code=500,
                    content={"error": "Invalid JSON response from Hugging Face", "raw_response": response.text[:500]}
                )
        
        elif response.status_code == 503:
            return JSONResponse(
                status_code=200,
                content={
                    "transcript": f"Audio Processing in Progress! Your audio file '{file.filename}' has been received. Our speech recognition service is currently starting up. Please wait 30-60 seconds and try uploading your audio again. Thank you for your patience!",
                    "demo_mode": True,
                    "api_status": "working",
                    "model_status": "loading_503",
                    "file_info": {
                        "filename": file.filename,
                        "size_bytes": len(contents),
                        "content_type": content_type
                    },
                    "retry_suggestion": "Please wait 30-60 seconds and try again"
                }
            )
        
        elif response.status_code == 404:
            logger.warning("Whisper model returned 404 - not available")
            return JSONResponse(
                status_code=200,
                content={
                    "transcript": f"Audio Successfully Received! Your audio file '{file.filename}' has been uploaded and processed. However, our speech-to-text service is temporarily unavailable. Please try again in a few minutes, or contact support if the issue persists. We apologize for the inconvenience!",
                    "demo_mode": True,
                    "api_status": "working",
                    "model_status": "unavailable_404", 
                    "file_info": {
                        "filename": file.filename,
                        "size_bytes": len(contents),
                        "content_type": content_type
                    },
                    "debug_info": {
                        "hf_status_code": response.status_code,
                        "hf_response": response.text[:100],
                        "model_url": HF_API_URL,
                        "token_provided": bool(HF_API_TOKEN and HF_API_TOKEN.strip())
                    },
                    "next_steps": "The speech recognition service is temporarily unavailable. Please try again in a few minutes."
                }
            )
        
        elif response.status_code == 429:
            return JSONResponse(
                status_code=200,
                content={
                    "transcript": f"Processing Limit Reached! Your audio file '{file.filename}' has been received successfully. However, we're currently processing many requests. Please wait a few minutes and try again. We appreciate your patience!",
                    "demo_mode": True,
                    "api_status": "working",
                    "model_status": "rate_limited_429",
                    "file_info": {
                        "filename": file.filename,
                        "size_bytes": len(contents),
                        "content_type": content_type
                    },
                    "retry_suggestion": "Please wait 3-5 minutes before trying again"
                }
            )
        
        else:
            logger.error(f"Unexpected status code: {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return JSONResponse(
                status_code=502,
                content={
                    "error": f"Hugging Face API error: {response.status_code}",
                    "details": response.text[:300],
                    "debug_info": f"Headers sent: {dict(headers)}"
                }
            )

    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        return JSONResponse(
            status_code=200,
            content={
                "transcript": f"Processing Timeout! Your audio file '{file.filename}' was received, but the transcription service is taking longer than expected. This might be due to high server load. Please try again in a few minutes.",
                "demo_mode": True,
                "api_status": "timeout",
                "model_status": "timeout_504",
                "file_info": {
                    "filename": file.filename,
                    "size_bytes": len(contents),
                    "content_type": content_type if 'content_type' in locals() else "unknown"
                },
                "retry_suggestion": "Please try again in 2-3 minutes"
            }
        )
    except requests.exceptions.ConnectionError:
        logger.error("Connection error")
        return JSONResponse(
            status_code=200,
            content={
                "transcript": f"Connection Issue! Your audio file '{file.filename}' was received, but we're having trouble connecting to our speech recognition service. Please check your internet connection and try again.",
                "demo_mode": True,
                "api_status": "connection_error",
                "model_status": "connection_failed_503",
                "file_info": {
                    "filename": file.filename,
                    "size_bytes": len(contents),
                    "content_type": content_type if 'content_type' in locals() else "unknown"
                },
                "retry_suggestion": "Please check your internet connection and try again"
            }
        )
    except Exception as e:
        logger.exception("Transcription failed with unexpected error")
        return JSONResponse(
            status_code=200, 
            content={
                "transcript": f"Unexpected Error! Your audio file '{file.filename}' was received, but we encountered an unexpected issue during processing. Our technical team has been notified. Please try again later.",
                "demo_mode": True,
                "api_status": "error",
                "model_status": "internal_error_500",
                "file_info": {
                    "filename": file.filename if 'file' in locals() and hasattr(file, 'filename') else "unknown",
                    "size_bytes": len(contents) if 'contents' in locals() else 0,
                    "content_type": content_type if 'content_type' in locals() else "unknown"
                },
                "error_details": str(e),
                "retry_suggestion": "Please try again later or contact support if the issue persists"
            }
        )
