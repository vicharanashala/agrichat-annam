from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi.responses import JSONResponse
from pymongo import MongoClient

from .config import MONGO_URI

logger = logging.getLogger("agrichat.app.db")


class SessionStore:
    def __init__(self) -> None:
        self._client: Optional[MongoClient] = None
        self._collection = None
        self._connect()

    def _connect(self) -> None:
        if not MONGO_URI:
            logger.warning("[Mongo] MONGO_URI not configured; session storage disabled")
            return
        try:
            self._client = MongoClient(MONGO_URI)
            db = self._client.get_database("agrichat")
            self._collection = db["sessions"]
            logger.info("[Mongo] Session store connected")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("[Mongo] Failed to initialize session client: %s", exc)
            self._client = None
            self._collection = None

    @property
    def collection(self):  # type: ignore[override]
        return self._collection

    def available(self) -> bool:
        return self._collection is not None

    def ensure_indexes(self) -> None:
        if not self.available():
            logger.warning("[Mongo] Skipping index creation: session collection unavailable")
            return
        try:
            self._collection.create_index([("state", 1)])
            self._collection.create_index([("state", 1), ("messages.question", 1)])
            self._collection.create_index([("timestamp", -1)])
            logger.info("[Mongo] Session indexes ensured")
        except Exception as exc:  # pragma: no cover
            logger.error("[Mongo] Failed to create indexes: %s", exc)

    def health(self) -> Dict[str, Any]:
        if self._client is None:
            return {"status": "down", "detail": "Mongo client not initialized"}
        try:
            self._client.admin.command("ping")
            status = "ok" if self.available() else "warn"
            detail = "sessions collection missing" if status == "warn" else "connected"
            return {"status": status, "detail": detail}
        except Exception as exc:  # pragma: no cover
            return {"status": "down", "detail": str(exc)}


session_store = SessionStore()
session_store.ensure_indexes()


def sessions_db_available() -> bool:
    return session_store.available()


def database_unavailable_response() -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": "Session storage temporarily unavailable"})


def missing_device_response() -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": "Device ID is required"})


def unauthorized_device_response() -> JSONResponse:
    return JSONResponse(status_code=403, content={"error": "Device authorization failed"})
