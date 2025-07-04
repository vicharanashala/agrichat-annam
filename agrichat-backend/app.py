from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pymongo import MongoClient
from uuid import uuid4
from datetime import datetime
from bs4 import BeautifulSoup
#from {// main.py path} import ChromaQueryHandler
from RAGpipelinev3.Gemini_based_processing.main import ChromaQueryHandler
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global query_handler
    chroma_path = "./RAGpipelinev3/Gemini_based_processing/chromaDb"

    if not os.path.exists(chroma_path):
        print(f"[Warning] chroma_path '{chroma_path}' does not exist!")

    query_handler = ChromaQueryHandler(
        chroma_path=chroma_path,
        gemini_api_key=os.getenv("GEMINI_API_KEY")
    )
    print("[Startup] QueryHandler initialized.")

    yield

    print("[Shutdown] App shutting down...")
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
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
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
    return {"sessions": [clean_session(s) for s in sessions]}

@app.post("/api/query")
async def new_session(question: str = Form(...), device_id: str = Form(...)):
    session_id = str(uuid4())
    try:
        raw_answer = query_handler.get_answer(question)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "LLM processing failed."})

    answer_only = raw_answer.split("</think>")[-1].strip() if "</think>" in raw_answer else raw_answer.strip()
    html_answer = markdown.markdown(answer_only, extensions=["extra", "nl2br"])
    
    session = {
        "session_id": session_id,
        "timestamp": datetime.now(IST).isoformat(),
        "messages": [{"question": question, "answer": html_answer, "rating":None}],
        "crop": "unknown",
        "state": "unknown",
        "status": "active",
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
async def continue_session(session_id: str, question: str = Form(...), device_id: str = Form(...)):
    session = sessions_collection.find_one({"session_id": session_id})
    if not session or session.get("status") == "archived" or session.get("device_id") != device_id:
        return JSONResponse(status_code=403, content={"error": "Session is archived, missing or unauthorized"})

    raw_answer = query_handler.get_answer(question)
    answer_only = raw_answer.split("</think>")[-1].strip()
    html_answer = markdown.markdown(answer_only, extensions=["extra", "nl2br"])

    crop = session.get("crop", "unknown")
    state = session.get("state", "unknown")

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
    