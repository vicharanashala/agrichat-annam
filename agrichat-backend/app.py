from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pymongo import MongoClient
from uuid import uuid4
from datetime import datetime
from bs4 import BeautifulSoup
#from {// main.py path} import ChromaQueryHandler
# from Agentic_RAG.chroma_query_handler import ChromaQueryHandler
from Agentic_RAG.main import initialize_handler
import markdown
import csv
from io import StringIO
import os
import certifi
import pytz
from dateutil import parser
IST = pytz.timezone("Asia/Kolkata")
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
from fastapi import Body
import requests
from fastapi import UploadFile, File
import logging
from sarvamai import SarvamAI


query_handler = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global query_handler
    query_handler = initialize_handler()
    print("[Startup] QueryHandler initialized.")
    yield
    print("[Shutdown] Cleanup complete.")
    
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global query_handler
#     chroma_path = "./RAGpipelinev3/Gemini_based_processing/chromaDb"

#     if not os.path.exists(chroma_path):
#         print(f"[Warning] chroma_path '{chroma_path}' does not exist!")

#     query_handler = ChromaQueryHandler(
#         chroma_path=chroma_path,
#         gemini_api_key=os.getenv("GEMINI_API_KEY")
#     )
#     print("[Startup] QueryHandler initialized.")

#     yield

#     print("[Shutdown] App shutting down...")
app = FastAPI(lifespan=lifespan)

origins = ["https://agrichat-annam.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# client = MongoClient("mongodb://localhost:27017/")
MONGO_URI = os.getenv("MONGO_URI")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
client = MongoClient(MONGO_URI) if ENVIRONMENT == "development" else MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client["agrichat"]
sessions_collection = db["sessions"]

# query_handler = ChromaQueryHandler(
#     chroma_path="./chroma_db",
#     model_name="gemma:2b",
#     base_url="http://localhost:11434/v1",
# )

@app.get("/")
async def root():
    return {"message": "AgriChat backend is running."}

def clean_session(s):
    s["_id"] = str(s["_id"])
    return s

MAX_SESSIONS = 25

@app.get("/api/sessions")
async def list_sessions(request: Request):
    device_id = request.query_params.get("device_id")
    if not device_id:
        return JSONResponse(status_code=400, content={"error": "Device ID is required"})
    sessions = list(
    sessions_collection.find({"device_id": device_id}, {"session_id": 1, "timestamp": 1, "crop": 1, "state": 1, "status": 1, "has_unread": 1, "messages": {"$slice": 1}})
    .sort("timestamp", -1)
    .limit(20)
    )
    # active_sessions = list(sessions_collection.find({"device_id": device_id, "status": "active"}))
    # isFull = len(active_sessions) >= MAX_SESSIONS
    return {"sessions": [clean_session(s) for s in sessions]}

@app.post("/api/query")
async def new_session(
    question: str = Form(None), 
    device_id: str = Form(...), 
    state: str = Form(...), 
    language: str = Form(...),
    audio_file: UploadFile = File(None)
):
    active_count = sessions_collection.count_documents({"device_id":device_id, "status":"active"})
    if active_count >= MAX_SESSIONS:
        return JSONResponse(
            status_code=400,
            content={"error": "Maximum active sessions reached. Please archive or delete an old session to start a new one."}
        )
    
    # Handle audio transcription if audio file is provided
    if audio_file:
        try:
            logger.info(f"Processing audio file: {audio_file.filename}")
            
            contents = await audio_file.read()
            logger.info(f"Audio file size: {len(contents)} bytes")

            # Get language code for Sarvam AI
            selected_language_code = LANGUAGE_CODE_MAP.get(language, "en-IN")
            logger.info(f"Using language: {language} (Code: {selected_language_code})")

            # Transcribe using Sarvam AI
            response = SARVAM_CLIENT.speech_to_text.transcribe(
                file=(audio_file.filename, contents),
                model="saarika:v2.5",
                language_code=selected_language_code
            )

            logger.info(f"Sarvam AI Response: {response}")

            if not response or not hasattr(response, 'transcript'):
                return JSONResponse(
                    status_code=500,
                    content={"error": "Transcription failed - no transcript in response"}
                )
            
            question = response.transcript.strip()
            logger.info(f"Transcribed text: {question}")
            
        except Exception as e:
            logger.exception("Sarvam AI transcription failed")
            return JSONResponse(status_code=500, content={"error": f"Sarvam AI transcription failed: {str(e)}"})
    
    # Validate that we have a question (either from text or audio)
    if not question:
        return JSONResponse(
            status_code=400,
            content={"error": "Either question text or audio file must be provided"}
        )
    
    session_id = str(uuid4())
    try:
        raw_answer = query_handler(question)
        # raw_answer = query_handler.get_answer(question)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "LLM processing failed."})

    answer_only = raw_answer.split("</think>")[-1].strip() if "</think>" in raw_answer else raw_answer.strip()
    html_answer = markdown.markdown(answer_only, extensions=["extra", "nl2br"])
    
    session = {
        "session_id": session_id,
        "timestamp": datetime.now(IST).isoformat(),
        "messages": [{"question": question, "answer": html_answer, "rating":None}],
        "crop": "unknown",
        "state": state,
        "status": "active",
        "language": language,
        "has_unread": True,
        "device_id": device_id, 
    }

    sessions_collection.insert_one(session)
    session.pop("_id", None)
    return {"session": session}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    session = sessions_collection.find_one({"session_id": session_id})
    if session:
        session["_id"] = str(session["_id"])
        sessions_collection.update_one({"session_id": session_id},{"$set":{"has_unread":False}})
        session["has_unread"]=False
    return {"session": session}


@app.post("/api/session/{session_id}/query")
async def continue_session(
    session_id: str, 
    question: str = Form(None), 
    device_id: str = Form(...), 
    state: str = Form(""),
    language: str = Form("English"),
    audio_file: UploadFile = File(None)
):
    session = sessions_collection.find_one({"session_id": session_id})
    if not session or session.get("status") == "archived" or session.get("device_id") != device_id:
        return JSONResponse(status_code=403, content={"error": "Session is archived, missing or unauthorized"})
    
    # Handle audio transcription if audio file is provided
    if audio_file:
        try:
            logger.info(f"Processing audio file for session {session_id}: {audio_file.filename}")
            
            contents = await audio_file.read()
            logger.info(f"Audio file size: {len(contents)} bytes")

            # Get language code for Sarvam AI
            selected_language_code = LANGUAGE_CODE_MAP.get(language, "en-IN")
            logger.info(f"Using language: {language} (Code: {selected_language_code})")

            # Transcribe using Sarvam AI
            response = SARVAM_CLIENT.speech_to_text.transcribe(
                file=(audio_file.filename, contents),
                model="saarika:v2.5",
                language_code=selected_language_code
            )

            logger.info(f"Sarvam AI Response: {response}")

            if not response or not hasattr(response, 'transcript'):
                return JSONResponse(
                    status_code=500,
                    content={"error": "Transcription failed - no transcript in response"}
                )
            
            question = response.transcript.strip()
            logger.info(f"Transcribed text for session {session_id}: {question}")
            
        except Exception as e:
            logger.exception("Sarvam AI transcription failed")
            return JSONResponse(status_code=500, content={"error": f"Sarvam AI transcription failed: {str(e)}"})
    
    # Validate that we have a question (either from text or audio)
    if not question:
        return JSONResponse(
            status_code=400,
            content={"error": "Either question text or audio file must be provided"}
        )
    
    raw_answer = query_handler(question)
    # raw_answer = query_handler.get_answer(question)
    answer_only = raw_answer.split("</think>")[-1].strip()
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
                "language": language,
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
        r = msg.get("rating","")
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
    result = sessions_collection.delete_one({"session_id":session_id})
    if result.deleted_count == 0:
        return JSONResponse(status_code=404, content={"error":"Session not found"})
    return {"message": "Session deleted successfully"}

@app.post("/api/session/{session_id}/rate")
async def rate_answer(session_id:str, question_index: int=Form(...), rating: str = Form(...)):
    if rating not in ["up","down"]:
        return JSONResponse(status_code=400,content={"error": "Invalid rating"})
    
    session = sessions_collection.find_one({"session_id":session_id})
    if not session:
        return JSONResponse(status_code=404,content={"error": "Session not found"})
    if question_index<0 or question_index>=len(session["messages"]):
        return JSONResponse(status_code=400,content={"error": "Invalid message index"})
    
    update_field = f"messages.{question_index}.rating"
    sessions_collection.update_one({"session_id":session_id},{"$set":{update_field:rating}})
    return {"message":"Rating Updated"}
    

@app.post("/api/update-language")
async def update_language(data: dict = Body(...)):
    device_id = data.get("device_id")
    state = data.get("state")
    language = data.get("language")
    print(state,language)
    if not device_id:
        return JSONResponse(status_code=400, content={"error": "Device ID is required"})

    result = sessions_collection.update_many(
        {"device_id": device_id},
        {"$set": {"state": state, "language": language}}
    )
    return {"status": "success", "matched": result.matched_count, "updated": result.modified_count}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


HF_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  

# Sarvam AI Configuration
SARVAM_API_KEY = "sk_0plgwxhb_stFNbHUQ5KzUjHaU0PQ7AsEH"
SARVAM_CLIENT = SarvamAI(api_subscription_key=SARVAM_API_KEY)

# Language code mapping for Sarvam AI
LANGUAGE_CODE_MAP = {
    "English": "en-IN", "Hindi": "hi-IN", "Tamil": "ta-IN", "Telugu": "te-IN",
    "Kannada": "kn-IN", "Gujarati": "gu-IN", "Marathi": "mr-IN", "Bengali": "bn-IN",
    "Punjabi": "pa-IN", "Malayalam": "ml-IN", "Odia": "or-IN", "Assamese": "as-IN",
    "Urdu": "ur-IN"
}

@app.post("/api/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")

        # Determine content type based on file extension
        content_type = file.content_type
        if not content_type or content_type == "application/octet-stream":
            if file.filename.lower().endswith('.mp3'):
                content_type = "audio/mpeg"
            elif file.filename.lower().endswith('.wav'):
                content_type = "audio/wav"
            elif file.filename.lower().endswith('.flac'):
                content_type = "audio/flac"
            elif file.filename.lower().endswith('.m4a'):
                content_type = "audio/m4a"
            else:
                content_type = "audio/mpeg"  # default to mp3
        
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": content_type,
        }

        response = requests.post(HF_API_URL, headers=headers, data=contents)

        logger.info(f"HF Status: {response.status_code}")
        logger.info(f"HF Raw response: {response.text}")

        if response.status_code != 200:
            return JSONResponse(
                status_code=502,
                content={"error": f"Hugging Face API error: {response.text}"}
            )

        result = response.json()

        transcript = result.get("text") or result.get("generated_text")
        if not transcript:
            return JSONResponse(
                status_code=500,
                content={"error": "Transcription not found in response", "raw": result}
            )

        return {"transcript": transcript}

    except Exception as e:
        logger.exception("Transcription failed")
        return JSONResponse(status_code=500, content={"error": str(e)})