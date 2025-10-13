from __future__ import annotations

import csv
import json
import os
import re
from copy import deepcopy
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import markdown
import pytz
from bs4 import BeautifulSoup
from dateutil import parser
from fastapi import Request
from fastapi.responses import JSONResponse
from langchain.memory import ConversationBufferWindowMemory

from pipeline.types import PipelineResult

from .config import IST


def normalize_source_name(source: Optional[str]) -> str:
    if not source:
        return ""

    original = source.strip()
    lowered = original.lower()

    mapping = {
        "rag database": "Golden Database",
        "rag database (golden)": "Golden Database",
        "golden": "Golden Database",
        "golden database": "Golden Database",
        "fallback llm": "AI Reasoning Engine (gpt-oss pipeline)",
        "fallback_llm": "AI Reasoning Engine (gpt-oss pipeline)",
        "llm_fallback": "AI Reasoning Engine (gpt-oss pipeline)",
        "gpt_oss_fallback": "AI Reasoning Engine (gpt-oss pipeline)",
        "llm": "AI Reasoning Engine (gpt-oss pipeline)",
        "ai reasoning engine (gpt-oss pipeline)": "AI Reasoning Engine (gpt-oss pipeline)",
    }

    return mapping.get(lowered, original)


def clean_golden_database_answer(raw_content: str) -> str:
    clean_content = re.sub(r"<[^>]+>", "", raw_content)

    lines = clean_content.strip().split("\n")
    answer_lines = []
    in_answer_section = False

    for line in lines:
        line_clean = line.strip()

        if ("|" in line_clean and any(key in line_clean for key in ["Date:", "State:", "Crop:", "District:", "Season:"])):
            continue

        if (":" in line_clean and any(key in line_clean for key in [
            "Date:",
            "State:",
            "Crop:",
            "District:",
            "Season:",
            "YT Video Link:",
            "Agri Specialist:",
            "Source:",
        ])):
            continue

        if line_clean.startswith("Question:"):
            continue

        if line_clean.startswith("Answer:"):
            answer_content = line_clean.replace("Answer:", "").strip()
            if answer_content:
                answer_lines.append(answer_content)
            in_answer_section = True
            continue

        if in_answer_section and line_clean:
            answer_lines.append(line_clean)

    clean_answer = " ".join(answer_lines).strip()
    if not clean_answer or len(clean_answer) < 20:
        return raw_content

    return clean_answer


def extract_golden_database_metadata(raw_content: str) -> dict:
    agri_specialist = None
    source = None
    if not raw_content:
        return {"agri_specialist": None, "source": None}
    clean_content = re.sub(r"<[^>]+>", "", raw_content)
    lines = clean_content.strip().split("\n")
    for line in lines:
        line_clean = line.strip()
        if "Agri Specialist:" in line_clean:
            match = re.search(r"Agri Specialist:\s*([^|\n]+)", line_clean)
            if match:
                agri_specialist = match.group(1).strip()
        if "Source:" in line_clean:
            match = re.search(r"Source:\s*([^|\n]+)", line_clean)
            if match:
                source = match.group(1).strip()
    return {"agri_specialist": agri_specialist, "source": source}


def pipeline_result_to_answer_dict(result: PipelineResult) -> Dict[str, Any]:
    metadata = deepcopy(result.metadata or {})
    database_config = metadata.get("database_config") or {}
    show_database_path = database_config.get("show_database_path", True)
    show_confidence_scores = database_config.get("show_confidence_scores", True)

    normalized_source = normalize_source_name(result.source)

    payload: Dict[str, Any] = {
        "answer": result.answer,
        "source": normalized_source,
        "thinking": "\n".join(result.reasoning) if result.reasoning else "",
        "reasoning": result.reasoning or [],
        "clarifying_questions": result.clarifying_questions or [],
        "metadata": metadata,
    }

    if show_confidence_scores and result.similarity is not None:
        payload["similarity"] = result.similarity
    if show_confidence_scores and result.distance is not None:
        payload["distance"] = result.distance

    retrieved_sources = []
    for entry in metadata.get("retrieved_sources") or []:
        if not isinstance(entry, dict):
            continue
        entry_copy = dict(entry)
        if not show_database_path:
            entry_copy.pop("source", None)
        if not show_confidence_scores:
            entry_copy.pop("cosine", None)
            entry_copy.pop("distance", None)
        retrieved_sources.append(entry_copy)
    if retrieved_sources:
        metadata["retrieved_sources"] = retrieved_sources
    else:
        metadata.pop("retrieved_sources", None)

    if not show_confidence_scores and metadata.get("llm_context_sources"):
        sanitized_context: Dict[str, List[Dict[str, Any]]] = {}
        context_sources = metadata.get("llm_context_sources") or {}
        for key, items in context_sources.items():
            sanitized_items: List[Dict[str, Any]] = []
            for item in items:
                if isinstance(item, dict):
                    item_copy = dict(item)
                    item_copy.pop("cosine", None)
                    item_copy.pop("distance", None)
                    sanitized_items.append(item_copy)
                else:
                    sanitized_items.append(item)
            sanitized_context[key] = sanitized_items
        metadata["llm_context_sources"] = sanitized_context

    if not show_confidence_scores and metadata.get("diagnostics"):
        diagnostics_section = metadata.get("diagnostics")
        if isinstance(diagnostics_section, dict):
            for section in diagnostics_section.values():
                if not isinstance(section, dict):
                    continue
                for hit in section.get("top_hits", []) or []:
                    if isinstance(hit, dict):
                        hit.pop("cosine", None)
                        hit.pop("distance", None)

    research_data: List[Dict[str, Any]] = []
    diagnostics = metadata.get("diagnostics") or {}
    label_map = {
        "golden": "Golden Database",
        "pops": "PoPs Database",
    }
    for key, section in diagnostics.items():
        if not isinstance(section, dict):
            continue
        label = label_map.get(key, key.title())
        source_label = normalize_source_name(label) if show_database_path else "Knowledge Source"
        for hit in section.get("top_hits", [])[:3]:
            if not isinstance(hit, dict):
                continue
            research_data.append(
                {
                    "source": source_label,
                    "content_preview": hit.get("preview", ""),
                    "metadata": {
                        "state": hit.get("state"),
                        **({"cosine": hit.get("cosine"), "distance": hit.get("distance")} if show_confidence_scores else {}),
                    },
                }
            )

    for entry in metadata.get("retrieved_sources", []) or []:
        if not isinstance(entry, dict):
            continue
        source_label = normalize_source_name(entry.get("source")) if show_database_path else "Knowledge Source"
        research_data.append(
            {
                "source": source_label,
                "content_preview": entry.get("preview", ""),
                "metadata": {
                    "state": entry.get("state"),
                    **({"cosine": entry.get("cosine"), "distance": entry.get("distance")} if show_confidence_scores else {}),
                },
            }
        )

    if research_data and show_database_path:
        payload["research_data"] = research_data

    if metadata.get("llm_context_sources"):
        payload["context_sources"] = metadata["llm_context_sources"]

    if metadata.get("intent_classification"):
        payload["intent_classification"] = metadata["intent_classification"]

    if metadata.get("clarifications"):
        payload.setdefault("clarifications", metadata["clarifications"])

    if metadata.get("retrieval_keyword_overlap"):
        payload["retrieval_keyword_overlap"] = metadata["retrieval_keyword_overlap"]

    if metadata.get("context_note"):
        payload["context_note"] = metadata["context_note"]

    return payload


def extract_answer_content(answer_result: Any) -> Tuple[str, Optional[Dict[str, str]]]:
    if answer_result is None:
        return "", None

    if isinstance(answer_result, PipelineResult):
        answer_result = pipeline_result_to_answer_dict(answer_result)

    if isinstance(answer_result, dict):
        answer_text = answer_result.get("answer", "") or ""
        source = normalize_source_name(answer_result.get("source", "") or "")
        doc_metadata = answer_result.get("document_metadata", {}) or {}
    else:
        answer_text = str(answer_result)
        source = ""
        doc_metadata = {}

    golden_metadata = None

    if source == "Golden Database":
        answer_text = clean_golden_database_answer(answer_text)

        if doc_metadata and ("Agri Specialist" in doc_metadata or "Source" in doc_metadata):
            golden_metadata = {
                "agri_specialist": doc_metadata.get("Agri Specialist"),
                "source": doc_metadata.get("Source"),
            }

    return answer_text, golden_metadata


def extract_sources(answer_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []

    research_data = answer_result.get("research_data") if isinstance(answer_result, dict) else None
    if research_data:
        for data in research_data:
            if not isinstance(data, dict):
                continue
            source_name = data.get("source") or data.get("collection_type") or "unknown"
            preview = data.get("content_preview") or (data.get("page_content") or "")[:300]
            sources.append(
                {
                    "source": normalize_source_name(source_name),
                    "preview": preview,
                    "metadata": data.get("metadata", {}),
                }
            )
    elif isinstance(answer_result, dict) and answer_result.get("context_sources"):
        for label, entries in answer_result["context_sources"].items():
            if not isinstance(entries, list):
                continue
            for item in entries:
                if not isinstance(item, dict):
                    continue
                sources.append(
                    {
                        "source": normalize_source_name(label),
                        "preview": item.get("preview", ""),
                        "metadata": {
                            "state": item.get("state"),
                            "cosine": item.get("cosine"),
                            "distance": item.get("distance"),
                        },
                    }
                )
    elif isinstance(answer_result, dict) and answer_result.get("source"):
        sources.append({"source": normalize_source_name(answer_result["source"]), "preview": ""})

    return sources


def build_answer_message(question: str, answer_result: Any, html_answer: str, golden_metadata: Optional[Dict[str, str]]) -> Dict[str, Any]:
    message: Dict[str, Any] = {
        "question": question,
        "thinking": answer_result.get("thinking", "") if isinstance(answer_result, dict) else "",
        "final_answer": html_answer,
        "answer": html_answer,
        "rating": None,
    }

    metadata_block: Dict[str, Any] = {}
    if golden_metadata:
        metadata_block.update(golden_metadata)

    if isinstance(answer_result, dict):
        if answer_result.get("metadata"):
            message["pipeline_metadata"] = answer_result["metadata"]
        if answer_result.get("reasoning"):
            message["reasoning_trace"] = answer_result["reasoning"]
        if answer_result.get("clarifying_questions"):
            message["clarifying_questions"] = answer_result["clarifying_questions"]
        if answer_result.get("context_note"):
            message["context_note"] = answer_result["context_note"]
        if answer_result.get("confidence") is not None:
            message["confidence"] = answer_result["confidence"]
        if answer_result.get("retrieval_keyword_overlap"):
            message["retrieval_keyword_overlap"] = answer_result["retrieval_keyword_overlap"]
        if answer_result.get("intent_classification"):
            metadata_block.setdefault("intent_classification", answer_result["intent_classification"])
        if "research_data" in answer_result:
            message["research_data"] = answer_result["research_data"]
        if "source" in answer_result:
            message["source"] = normalize_source_name(answer_result["source"])
        if "ragas_score" in answer_result:
            message["ragas_score"] = answer_result["ragas_score"]
        if answer_result.get("similarity") is not None:
            message["similarity"] = answer_result["similarity"]
        if answer_result.get("distance") is not None:
            message["distance"] = answer_result["distance"]

        sources = extract_sources(answer_result)
        if sources:
            message["sources"] = sources

    if metadata_block:
        message["metadata"] = metadata_block

    return message


def get_request_device_id(request: Request) -> Optional[str]:
    header_device = request.headers.get("X-Device-Id") if hasattr(request, "headers") else None
    if header_device and header_device.strip():
        return header_device.strip()
    query_device = request.query_params.get("device_id") if hasattr(request, "query_params") else None
    if query_device and isinstance(query_device, str) and query_device.strip():
        return query_device.strip()
    return None


def clean_session(document: Dict[str, Any]) -> Dict[str, Any]:
    document["_id"] = str(document["_id"])
    return document


def format_iso(ts: datetime) -> str:
    return ts.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S")


def conversation_memory_for_session(session_id: str, store: Dict[str, ConversationBufferWindowMemory]) -> ConversationBufferWindowMemory:
    if session_id not in store:
        store[session_id] = ConversationBufferWindowMemory(k=8, return_messages=True)
    return store[session_id]
