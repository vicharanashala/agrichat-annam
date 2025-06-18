import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import markdown
from main import ChromaQueryHandler

from pymongo import MongoClient
from uuid import uuid4
from datetime import datetime
import csv
from io import StringIO
from fastapi.responses import StreamingResponse
from bs4 import BeautifulSoup

client = MongoClient("mongodb://localhost:27017/")
db = client["agrichat"]
sessions_collection = db["sessions"]

app = FastAPI()

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory="templates")

query_handler = ChromaQueryHandler(
    chroma_path=r"./chroma_db",
    # chroma_path=r"/Users/madhurthareja/itachicmd/agrichat-annam/RAG pipeline v2/chroma_db",
    # model_name="gemma3:1b",
    model_name="gemma:2b",
    base_url="http://localhost:11434/v1",
)

def get_session(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id:
        return sessions_collection.find_one({"session_id": session_id}), session_id
    return None, None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    response = templates.TemplateResponse("index.html", {
        "request": request,
        "session": None,
        "messages": [],
        "recent_sessions": list(
            sessions_collection.find({}, {"session_id": 1, "timestamp": 1, "crop": 1, "state": 1, "status": 1, "has_unread": 1})
            .sort("timestamp", -1).limit(10)
        )
    })
    response.delete_cookie("session_id")
    return response

@app.post("/query", response_class=HTMLResponse)
async def query(request: Request, question: str = Form(...)):
    session_id = str(uuid4())
    raw_answer = query_handler.get_answer(question)
    answer_only = raw_answer.split("</think>")[-1].strip() if "</think>" in raw_answer else raw_answer.strip()
    html_answer = markdown.markdown(answer_only, extensions=["extra", "nl2br"])

    session_data = {
        "session_id": session_id,
        "timestamp": datetime.now(),
        "messages": [{"question": question, "answer": html_answer}],
        "crop": "unknown",
        "state": "unknown",
        "status": "active",
        "has_unread": True
    }
    sessions_collection.insert_one(session_data)

    response = RedirectResponse(url=f"/resume/{session_id}", status_code=303)
    response.set_cookie(key="session_id", value=session_id)
    return response

@app.get("/resume/{session_id}", response_class=HTMLResponse)
async def resume_session(session_id: str, request: Request):
    session = sessions_collection.find_one({"session_id": session_id})
    if not session:
        return RedirectResponse("/")

    sessions_collection.update_one(
        {"session_id": session_id},
        {"$set": {"has_unread": False}}
    )
    messages = session.get("messages", [])
    response = templates.TemplateResponse("index.html", {
        "request": request,
        "session": session,
        "messages": messages,
        "recent_sessions": list(
            sessions_collection.find({}, {"session_id": 1, "timestamp": 1, "crop": 1, "state": 1, "status": 1, "has_unread": 1})
            .sort("timestamp", -1).limit(10)
        )
    })
    response.set_cookie(key="session_id", value=session_id)
    return response

@app.post("/resume/{session_id}/query", response_class=HTMLResponse)
async def continue_session(session_id: str, request: Request, question: str = Form(...)):
    session = sessions_collection.find_one({"session_id": session_id})
    if not session:
        return RedirectResponse("/")

    raw_answer = query_handler.get_answer(question)
    answer_only = raw_answer.split("</think>")[-1].strip() if "</think>" in raw_answer else raw_answer.strip()
    html_answer = markdown.markdown(answer_only, extensions=["extra", "nl2br"])

    sessions_collection.update_one(
        {"session_id": session_id},
        {"$push": {"messages": {"question": question, "answer": html_answer}},"$set": {"has_unread": True}}
    )

    return RedirectResponse(url=f"/resume/{session_id}", status_code=303)

@app.post("/new-session", response_class=HTMLResponse)
async def new_session(request: Request):
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie("session_id")
    return response

@app.get("/export/csv/{session_id}")
async def export_csv(session_id: str):
    session = sessions_collection.find_one({"session_id": session_id})
    if not session:
        return {"error": "Session not found"}
    
    output=StringIO()
    writer = csv.writer(output)
    writer.writerow(["Question","Answer"])
    
    for msg in session.get("messages",[]):
        question = msg["question"]
        soup = BeautifulSoup(msg["answer"], "html.parser")
        plain_answer = soup.get_text()
        writer.writerow([question, plain_answer])
        
    output.seek(0)
    return StreamingResponse(output, media_type="text/csv", headers={
        "Content-Disposition": f"attachment; filename=agrichat_session_{session_id}.csv"
    })