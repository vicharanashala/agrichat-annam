from __future__ import annotations

import logging
import csv
import requests
from io import StringIO, BytesIO
from typing import Any, Dict

from fastapi import APIRouter, Body, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from dateutil import parser

from bs4 import BeautifulSoup

from ..db import (
    database_unavailable_response,
    missing_device_response,
    session_store,
    sessions_db_available,
    unauthorized_device_response,
)
from ..models import DatabaseToggleConfig, QueryRequest, SessionQueryRequest
from ..pipeline_service import (
    build_pipeline_overrides_from_config,
    handle_new_session,
    handle_session_query,
    thinking_stream_response,
)
from ..utils import clean_session, get_request_device_id, format_iso
from ..config import IST

logger = logging.getLogger("agrichat.app.routes.chat")

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/query")
async def new_session(request: QueryRequest):
    logger.info("[API] new_session invoked")
    return await handle_new_session(request)


@router.post("/session/{session_id}/query")
async def continue_session(session_id: str, request: SessionQueryRequest):
    logger.info("[API] continue_session invoked")
    return await handle_session_query(session_id, request)


@router.get("/sessions")
async def list_sessions(request: Request):
    if not sessions_db_available():
        return database_unavailable_response()

    device_id = get_request_device_id(request)
    if not device_id:
        return missing_device_response()

    sessions = list(
        session_store.collection.find({"device_id": device_id}).sort("timestamp", -1).limit(20)
    )
    return {"sessions": [clean_session(s) for s in sessions]}


@router.get("/session/{session_id}")
async def get_session(session_id: str, request: Request):
    if not sessions_db_available():
        return database_unavailable_response()

    device_id = get_request_device_id(request)
    if not device_id:
        return missing_device_response()

    session = session_store.collection.find_one({"session_id": session_id})
    if not session:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    if session.get("device_id") != device_id:
        return unauthorized_device_response()

    session["_id"] = str(session["_id"])
    session_store.collection.update_one({"session_id": session_id}, {"$set": {"has_unread": False}})
    session["has_unread"] = False
    return {"session": session}


@router.post("/query/thinking-stream")
async def thinking_stream_query(request: QueryRequest):
    return await thinking_stream_response(request)


@router.post("/toggle-status/{session_id}/{status}")
async def toggle_status(session_id: str, status: str, request: Request):
    if not sessions_db_available():
        return database_unavailable_response()

    device_id = get_request_device_id(request)
    if not device_id:
        return missing_device_response()

    session = session_store.collection.find_one({"session_id": session_id})
    if not session:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    if session.get("device_id") != device_id:
        return unauthorized_device_response()

    new_status = "archived" if status == "active" else "active"
    session_store.collection.update_one(
        {"session_id": session_id},
        {"$set": {"status": new_status}},
    )
    return {"status": new_status}


@router.get("/export/csv/{session_id}")
async def export_csv(session_id: str, request: Request):
    if not sessions_db_available():
        return database_unavailable_response()

    device_id = get_request_device_id(request)
    if not device_id:
        return missing_device_response()

    session = session_store.collection.find_one({"session_id": session_id})
    if not session:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    if session.get("device_id") != device_id:
        return unauthorized_device_response()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Question", "Answer", "Rating", "Timestamp"])
    for index, message in enumerate(session.get("messages", [])):
        question = message.get("question")
        answer = BeautifulSoup(message.get("answer", ""), "html.parser").get_text()
        rating = message.get("rating", "")
        timestamp = (
            parser.isoparse(session["timestamp"]).astimezone(IST).strftime("%Y-%m-%d %H:%M:%S")
            if index == 0
            else ""
        )
        writer.writerow([question, answer, rating, timestamp])

    output.seek(0)
    filename = f"session_{session_id}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.delete("/delete-session/{session_id}")
async def delete_session(session_id: str, request: Request):
    if not sessions_db_available():
        return database_unavailable_response()

    device_id = get_request_device_id(request)
    if not device_id:
        return missing_device_response()

    session = session_store.collection.find_one({"session_id": session_id})
    if not session:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    if session.get("device_id") != device_id:
        return unauthorized_device_response()

    result = session_store.collection.delete_one({"session_id": session_id})
    if result.deleted_count == 0:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    return {"message": "Session deleted successfully"}


@router.post("/session/{session_id}/rate")
async def rate_answer(
    session_id: str,
    request: Request,
    question_index: int = Form(...),
    rating: str = Form(...),
    device_id: str = Form(...),
):
    if not sessions_db_available():
        return database_unavailable_response()

    if rating not in {"up", "down"}:
        return JSONResponse(status_code=400, content={"error": "Invalid rating value"})

    if not device_id or not device_id.strip():
        return missing_device_response()

    session = session_store.collection.find_one({"session_id": session_id})
    if not session:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    if session.get("device_id") != device_id.strip():
        return unauthorized_device_response()

    if question_index < 0 or question_index >= len(session.get("messages", [])):
        return JSONResponse(status_code=400, content={"error": "Question index out of range"})

    update_field = f"messages.{question_index}.rating"
    session_store.collection.update_one({"session_id": session_id}, {"$set": {update_field: rating}})
    return {"message": "Rating Updated"}


@router.post("/update-language")
async def update_language(data: Dict[str, Any] = Body(...)):
    if not sessions_db_available():
        return database_unavailable_response()

    device_id = data.get("device_id")
    state = data.get("state")
    language = data.get("language")

    if not device_id or not isinstance(device_id, str) or not device_id.strip():
        return missing_device_response()
    device_id = device_id.strip()

    if session_store.collection.count_documents({"device_id": device_id}) == 0:
        return JSONResponse(status_code=404, content={"error": "Device not found"})

    session_store.collection.update_many(
        {"device_id": device_id},
        {"$set": {"state": state, "language": language}},
    )
    return {"message": "Language updated"}


@router.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile, language: str = Form("English")):
    try:
        # Use the custom transcription API instead of Whisper
        import requests
        from io import BytesIO

        # Read the uploaded file
        audio_data = await file.read()

        # Create a BytesIO object for the requests library
        audio_buffer = BytesIO(audio_data)
        audio_buffer.seek(0)

        # Call the custom transcription API
        url = "https://hyperlogical-soppiest-krystin.ngrok-free.dev/api/transcribe"
        files = {"audio": ("audio.wav", audio_buffer, "audio/wav")}
        data = {"translate": "false"}  # Default to no translation

        response = requests.post(url, files=files, data=data)
        response.raise_for_status()

        result = response.json()

        if result.get("success"):
            # Return the punctuated text if available, otherwise original text
            punctuated_text = result.get("punctuation", {}).get("punctuated_text")
            if punctuated_text:
                transcript = punctuated_text
            else:
                transcript = result.get("transcription", {}).get("original_text", "")
        else:
            raise Exception("Transcription API returned failure")

        return {"transcript": transcript}

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Transcription failed: {str(e)}"}
        )


@router.post("/test-database-toggle")
async def test_database_toggle(
    question: str = Form(...),
    golden_db: bool = Form(False),
    pops_db: bool = Form(False),
    llm_fallback: bool = Form(False),
):
    config = DatabaseToggleConfig(
        golden_enabled=golden_db,
        pops_enabled=pops_db,
        llm_enabled=llm_fallback,
    )
    overrides = build_pipeline_overrides_from_config(config)
    return {"question": question, "overrides": overrides}
