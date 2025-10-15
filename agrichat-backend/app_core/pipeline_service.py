from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import markdown
from bs4 import BeautifulSoup
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain.memory import ConversationBufferWindowMemory

from pipeline import classify_question_intent, run_pipeline
from pipeline.types import PipelineResult

from .context import convert_langchain_memory_to_history, enhance_answer_with_context_questions
from .db import (
    database_unavailable_response,
    missing_device_response,
    session_store,
    sessions_db_available,
    unauthorized_device_response,
)
from .models import DatabaseToggleConfig, QueryRequest, SessionQueryRequest
from .config import iso_now
from .utils import (
    build_answer_message,
    clean_session,
    conversation_memory_for_session,
    extract_answer_content,
    extract_golden_database_metadata,
    get_request_device_id,
    normalize_source_name,
    pipeline_result_to_answer_dict,
)

logger = logging.getLogger("agrichat.app.pipeline")


GREETINGS = {
    "hi",
    "hello",
    "hey",
    "namaste",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
}


def build_pipeline_overrides_from_config(db_config: DatabaseToggleConfig) -> Dict[str, Any]:
    config_dict = db_config.dict()
    overrides: Dict[str, Any] = {
        "enable_golden": db_config.golden_enabled,
        "enable_pops": db_config.pops_enabled,
        "enable_llm": db_config.llm_enabled,
        "golden_min_cosine": db_config.similarity_threshold,
        "pops_min_cosine": db_config.pops_similarity_threshold,
        "adaptive_thresholds": db_config.enable_adaptive_thresholds,
        "strict_validation": db_config.strict_validation,
        "raw_database_config": config_dict,
    }
    return overrides


def extract_thinking_process(text: str) -> Tuple[str, str]:
    if not text:
        return "", text

    think_pattern = r"<think>(.*?)</think>"
    matches = re.findall(think_pattern, text, re.DOTALL | re.IGNORECASE)
    thinking = ""
    if matches:
        thinking = matches[0].strip()
        clean_answer = re.sub(think_pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    else:
        clean_answer = text

    return thinking, clean_answer


def _greeting_response(question: str) -> Optional[Dict[str, Any]]:
    try:
        if question.lower().strip() in GREETINGS:
            return {
                "answer": (
                    "Hello! I'm your agricultural assistant specializing in Indian farming. "
                    "I can help with crop management, soil health, pest control, fertilizers, "
                    "irrigation, farming techniques, and agricultural practices. What would you like to know?"
                ),
                "source": "Greeting",
                "confidence": 1.0,
                "thinking": "",
            }
    except Exception:  # pragma: no cover - defensive
        logger.debug("[Greeting] Greeting detection failed; continuing with pipeline")
    return None


session_memories: Dict[str, ConversationBufferWindowMemory] = {}


def _intent_failure_payload(
    raw_db_config: Optional[Dict[str, Any]], intent_metadata: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    return {
        "answer": (
            "I'm an agricultural assistant focused on Indian farming. I can only help with agriculture-related questions. "
            "Please ask about crops, farming practices, soil management, pest control, or other agricultural topics."
        ),
        "source": "Non-Agricultural",
        "confidence": 1.0,
        "thinking": "",
        "metadata": {
            "intent_classification": intent_metadata,
            **({"database_config": raw_db_config} if raw_db_config else {}),
        },
    }


async def run_pipeline_answer(
    question: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    user_state: Optional[str] = None,
    db_config: Optional[DatabaseToggleConfig] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw_db_config: Optional[Dict[str, Any]] = None
    if config_overrides and isinstance(config_overrides, dict):
        raw_db_config = config_overrides.get("raw_database_config")
    elif db_config:
        try:
            raw_db_config = db_config.dict()
        except Exception:
            raw_db_config = None

    overrides_payload: Optional[Dict[str, Any]] = None
    if db_config:
        overrides_payload = build_pipeline_overrides_from_config(db_config)
    elif config_overrides:
        overrides_payload = config_overrides

    greeting_payload = _greeting_response(question)
    if greeting_payload:
        return greeting_payload

    classification_text = question
    if conversation_history:
        recent_user_messages = [
            (entry.get("content") or "")
            for entry in conversation_history
            if entry.get("role") == "user"
        ]
        if recent_user_messages:
            recent_context = " \n".join(message.strip() for message in recent_user_messages[-3:] if message)
            if recent_context:
                classification_text = f"{recent_context}\nFollow-up: {question}"

    try:
        intent_metadata = classify_question_intent(classification_text)
    except Exception as exc:  # pragma: no cover
        logger.warning("[Intent] classification failed: %s", exc)
        intent_metadata = None

    if intent_metadata and not intent_metadata.get("final"):
        return _intent_failure_payload(raw_db_config, intent_metadata)

    loop = asyncio.get_event_loop()

    try:
        pipeline_result: PipelineResult = await loop.run_in_executor(
            None,
            lambda: run_pipeline(
                question,
                conversation_history or [],
                user_state,
                intent_metadata=intent_metadata,
                config_overrides=overrides_payload,
            ),
        )
    except Exception as exc:  # pragma: no cover
        logger.error("[Pipeline] run_pipeline failed: %s", exc)
        return {
            "answer": "Service temporarily unavailable: internal error.",
            "thinking": "",
            "source": "Error",
            "confidence": 0.0,
            "research_data": [],
            "reasoning_steps": [],
            "metadata": {},
        }

    response = pipeline_result_to_answer_dict(pipeline_result)
    if raw_db_config:
        response.setdefault("metadata", {})
        response["metadata"].setdefault("database_config", raw_db_config)

    raw_answer = response.get("answer", "")
    thinking, clean_answer = extract_thinking_process(raw_answer)
    response["answer"] = clean_answer
    if thinking:
        response["thinking"] = thinking
    response["answer_markdown"] = clean_answer
    response["answer_plain"] = (
        BeautifulSoup(clean_answer, "html.parser").get_text(separator=" ").strip() if clean_answer else ""
    )
    response["source"] = normalize_source_name(response.get("source"))
    response["confidence"] = response.get("similarity", 0.0) or 0.0
    response.setdefault("research_data", [])
    response.setdefault("reasoning_steps", response.get("reasoning", []))

    return response


async def handle_new_session(request: QueryRequest) -> Dict[str, Any]:
    if not request.device_id or not request.device_id.strip():
        return missing_device_response()

    db_config: Optional[DatabaseToggleConfig] = None
    config_overrides: Optional[Dict[str, Any]] = None
    if request.database_config:
        db_config = DatabaseToggleConfig(**request.database_config)
        config_overrides = build_pipeline_overrides_from_config(db_config)

    session_id = str(uuid4())
    memory = conversation_memory_for_session(session_id, session_memories)

    memory.chat_memory.add_user_message(request.question)
    answer = await run_pipeline_answer(
        request.question,
        conversation_history=[],
        user_state=request.state,
        db_config=db_config,
        config_overrides=config_overrides,
    )

    if isinstance(answer, dict) and answer.get("answer"):
        memory.chat_memory.add_ai_message(answer["answer"])
    else:  # pragma: no cover
        memory.chat_memory.add_ai_message(str(answer))

    answer_only, golden_metadata = extract_answer_content(answer)
    html_answer = markdown.markdown(answer_only, extensions=["extra", "nl2br"])
    message = build_answer_message(request.question, answer, html_answer, golden_metadata)

    session_document = {
        "session_id": session_id,
        "timestamp": iso_now(),
        "messages": [message],
        "crop": "unknown",
        "state": request.state,
        "status": "active",
        "language": request.language,
        "has_unread": True,
        "device_id": request.device_id,
    }

    if sessions_db_available():
        try:
            session_store.collection.insert_one(session_document)
            session_document.pop("_id", None)
        except Exception as exc:  # pragma: no cover
            logger.error("[Mongo] Failed to persist session %s: %s", session_id, exc)
    else:
        logger.warning("[Mongo] Session storage unavailable; skipping persistence for %s", session_id)

    session_document["recommendations"] = []

    return {"session": session_document}


async def handle_session_query(session_id: str, request: SessionQueryRequest) -> JSONResponse:
    if not sessions_db_available():
        return database_unavailable_response()

    if not request.device_id or not request.device_id.strip():
        return missing_device_response()

    session = session_store.collection.find_one({"session_id": session_id})
    if not session or session.get("status") == "archived" or session.get("device_id") != request.device_id:
        return JSONResponse(status_code=403, content={"error": "Session is archived, missing or unauthorized"})

    db_config: Optional[DatabaseToggleConfig] = None
    config_overrides: Optional[Dict[str, Any]] = None
    if request.database_config:
        db_config = DatabaseToggleConfig(**request.database_config)
        config_overrides = build_pipeline_overrides_from_config(db_config)

    memory = conversation_memory_for_session(session_id, session_memories)
    conversation_history = convert_langchain_memory_to_history(memory)
    current_state = request.state or session.get("state", "unknown")

    memory.chat_memory.add_user_message(request.question)
    answer = await run_pipeline_answer(
        request.question,
        conversation_history=conversation_history,
        user_state=current_state,
        db_config=db_config,
        config_overrides=config_overrides,
    )

    if isinstance(answer, dict) and answer.get("answer"):
        memory.chat_memory.add_ai_message(answer["answer"])
    else:  # pragma: no cover
        memory.chat_memory.add_ai_message(str(answer))

    answer_only, golden_metadata = extract_answer_content(answer)
    html_answer = markdown.markdown(answer_only, extensions=["extra", "nl2br"])
    new_message = build_answer_message(request.question, answer, html_answer, golden_metadata)

    crop = session.get("crop", "unknown")
    session_store.collection.update_one(
        {"session_id": session_id},
        {
            "$push": {"messages": new_message},
            "$set": {
                "has_unread": True,
                "crop": crop,
                "state": current_state,
                "timestamp": iso_now(),
            },
        },
    )

    updated = session_store.collection.find_one({"session_id": session_id})
    if updated:
        updated.pop("_id", None)
        updated["recommendations"] = []

    return JSONResponse(status_code=200, content={"session": updated})


async def thinking_stream_response(request: QueryRequest) -> StreamingResponse:
    if not request.device_id or not request.device_id.strip():
        return missing_device_response()

    db_config: Optional[DatabaseToggleConfig] = None
    config_overrides: Optional[Dict[str, Any]] = None
    if request.database_config:
        try:
            db_config = DatabaseToggleConfig(**request.database_config)
            config_overrides = build_pipeline_overrides_from_config(db_config)
        except Exception as exc:
            logger.error("[Stream] Invalid database configuration: %s", exc)
            db_config = None
            config_overrides = None

    can_persist_stream = sessions_db_available()

    async def generate_stream():
        session_id = str(uuid4())
        logger.info("[Stream] Starting stream for question: %s", request.question[:50])

        yield f"data: {json.dumps({'type': 'session_start', 'session_id': session_id})}\n\n"
        yield f"data: {json.dumps({'type': 'thinking_start'})}\n\n"

        try:
            result = await run_pipeline_answer(
                request.question,
                conversation_history=[],
                user_state=request.state,
                db_config=db_config,
                config_overrides=config_overrides,
            )
        except Exception as exc:  # pragma: no cover
            logger.error("[Stream] Failed to generate answer: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to generate answer'})}\n\n"
            yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
            return

        thinking_content = result.get("thinking") or ""
        yield f"data: {json.dumps({'type': 'thinking_complete', 'thinking': thinking_content})}\n\n"

        yield f"data: {json.dumps({'type': 'answer_start'})}\n\n"

        answer_only, golden_metadata = extract_answer_content(result)
        answer_markdown = result.get("answer_markdown") or answer_only
        response_data = {
            "type": "answer",
            "answer": answer_markdown,
            "source": normalize_source_name(result.get("source")),
            "confidence": result.get("confidence", 0.0),
        }

        if thinking_content:
            response_data["thinking"] = thinking_content

        metadata = result.get("metadata") or {}
        if metadata:
            response_data["metadata"] = metadata

        if result.get("database_path") or metadata.get("database_path"):
            response_data["database_path"] = result.get("database_path") or metadata.get("database_path")

        if result.get("confidence_scores") or metadata.get("confidence_scores"):
            response_data["confidence_scores"] = result.get("confidence_scores") or metadata.get("confidence_scores")

        if result.get("reasoning_steps") or metadata.get("reasoning_steps") or result.get("reasoning"):
            response_data["reasoning_steps"] = (
                result.get("reasoning_steps")
                or metadata.get("reasoning_steps")
                or result.get("reasoning")
            )

        yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"

        html_answer = markdown.markdown(answer_only, extensions=["extra", "nl2br"])
        message = build_answer_message(request.question, result, html_answer, golden_metadata)

        session_document = {
            "session_id": session_id,
            "timestamp": iso_now(),
            "messages": [message],
            "crop": "unknown",
            "state": request.state,
            "status": "active",
            "language": request.language,
            "has_unread": True,
            "device_id": request.device_id,
        }

        storage_status = "skipped"
        if can_persist_stream:
            try:
                session_store.collection.insert_one(session_document)
                session_document.pop("_id", None)
                storage_status = "persisted"
            except Exception as exc:
                logger.error("[Stream] Failed to persist streamed session %s: %s", session_id, exc)
        else:
            logger.warning("[Stream] Session storage unavailable; skipping persistence for %s", session_id)

        session_document["recommendations"] = []

        completion_payload = {
            "type": "session_complete",
            "session": session_document,
            "stored": storage_status == "persisted",
        }
        yield f"data: {json.dumps(completion_payload, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )
