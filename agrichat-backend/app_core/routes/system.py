from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..config import CORS_ORIGINS, iso_now
from ..health import check_chroma_health, check_mongo_health, check_ollama_health

logger = logging.getLogger("agrichat.app.routes.system")

router = APIRouter(tags=["system"])


@router.get("/")
async def root():
    return {"message": "AgriChat backend is running."}


@router.get("/health")
async def health():
    checks = {
        "mongo": check_mongo_health(),
        "chroma": check_chroma_health(),
        "ollama": check_ollama_health(),
    }

    statuses = [check.get("status") for check in checks.values()]
    if any(status == "down" for status in statuses):
        overall = "unhealthy"
    elif any(status == "warn" for status in statuses):
        overall = "degraded"
    else:
        overall = "healthy"

    return {
        "status": overall,
        "timestamp": iso_now(),
        "checks": checks,
    }


@router.options("/{full_path:path}")
async def options_handler(request: Request):
    origin = request.headers.get("origin")
    allowed_origin = origin if origin in CORS_ORIGINS else CORS_ORIGINS[0]

    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": allowed_origin,
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "86400",
        },
    )
