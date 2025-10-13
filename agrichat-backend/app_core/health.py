import logging
import os
from typing import Any, Dict

import requests

from .config import CHROMA_DB_PATH
from .db import session_store

logger = logging.getLogger("agrichat.app.health")


def check_mongo_health() -> Dict[str, Any]:
    return session_store.health()


def check_chroma_health() -> Dict[str, Any]:
    exists = os.path.exists(CHROMA_DB_PATH)
    status = "ok" if exists else "warn"
    detail = "available" if exists else "path not found"
    return {"status": status, "detail": detail, "path": CHROMA_DB_PATH}


def check_ollama_health() -> Dict[str, Any]:
    host = os.getenv("OLLAMA_HOST", "localhost:11434")
    if host.startswith("http://") or host.startswith("https://"):
        base_url = host
    else:
        base_url = f"http://{host}"
    url = f"{base_url}/api/tags"
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            return {"status": "ok", "detail": "reachable", "endpoint": base_url}
        return {"status": "warn", "detail": f"status {response.status_code}", "endpoint": base_url}
    except Exception as exc:  # pragma: no cover
        return {"status": "down", "detail": str(exc), "endpoint": base_url}
