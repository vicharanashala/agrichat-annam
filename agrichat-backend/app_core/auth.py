from __future__ import annotations

import csv
import logging
import os
from typing import Dict

from fastapi import APIRouter

from .models import AuthResponse, LoginRequest

logger = logging.getLogger("agrichat.app.auth")

router = APIRouter(prefix="/api", tags=["auth"])


def _users_csv_path() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "users.csv")


def load_users_from_csv() -> Dict[str, Dict[str, str]]:
    users: Dict[str, Dict[str, str]] = {}
    csv_path = os.path.abspath(_users_csv_path())

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                users[row["username"]] = {
                    "password": row["password"],
                    "role": row["role"],
                    "full_name": row["full_name"],
                }
        logger.info("[AUTH] Loaded %d users from CSV", len(users))
    except FileNotFoundError:
        logger.error("[AUTH] Users CSV file not found at %s", csv_path)
    except Exception as exc:  # pragma: no cover
        logger.error("[AUTH] Error loading users CSV: %s", exc)

    return users


def authenticate_user(username: str, password: str) -> Dict[str, str]:
    users = load_users_from_csv()

    if username in users and users[username]["password"] == password:
        user = users[username]
        return {
            "authenticated": True,
            "username": username,
            "role": user["role"],
            "full_name": user["full_name"],
        }

    return {"authenticated": False}


@router.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest) -> AuthResponse:
    auth_result = authenticate_user(request.username, request.password)

    if auth_result.get("authenticated"):
        return AuthResponse(
            authenticated=True,
            username=auth_result.get("username"),
            role=auth_result.get("role"),
            full_name=auth_result.get("full_name"),
            message="Login successful",
        )

    return AuthResponse(
        authenticated=False,
        message="Invalid username or password",
    )
