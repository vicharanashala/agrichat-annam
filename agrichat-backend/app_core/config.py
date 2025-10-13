import json
import logging
import os
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional

import pytz


# Configure root logger once for the backend package
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agrichat.app")


IST = pytz.timezone("Asia/Kolkata")

BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_ROOT)
CURRENT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "agrichat-backend"))

if os.path.exists("/app"):
    CHROMA_DB_PATH = "/app/chromaDb"
else:
    CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "chromaDb")

MONGO_URI = os.getenv("MONGO_URI")


CORS_ORIGINS = [
    "https://agri-annam.vercel.app",
    "https://agrichat.annam.ai",
    "https://agrichat.serveo.net",
    "https://localhost:3000",
    "https://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*",
]


def iso_now() -> str:
    return datetime.now(IST).isoformat()
