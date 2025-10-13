from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DatabaseToggleConfig(BaseModel):
    """Toggle and scoring configuration for knowledge sources used by the lightweight pipeline."""

    golden_enabled: bool = True
    pops_enabled: bool = True
    llm_enabled: bool = True

    similarity_threshold: float = 0.7
    pops_similarity_threshold: float = 0.35
    enable_adaptive_thresholds: bool = True
    strict_validation: bool = False
    show_database_path: bool = True
    show_confidence_scores: bool = True

    def is_any_enabled(self) -> bool:
        return any([
            self.golden_enabled,
            self.pops_enabled,
            self.llm_enabled,
        ])

    def is_traditional_mode(self) -> bool:
        return self.golden_enabled and self.pops_enabled and self.llm_enabled

    def get_enabled_databases(self) -> List[str]:
        enabled: List[str] = []
        if self.golden_enabled:
            enabled.append("golden")
        if self.pops_enabled:
            enabled.append("pops")
        if self.llm_enabled:
            enabled.append("llm")
        return enabled

    def dict(self, *args, **kwargs):  # type: ignore[override]
        data = super().dict(*args, **kwargs)
        return data


class QueryRequest(BaseModel):
    question: str
    device_id: str
    state: str = ""
    language: str = "en"
    database_config: Optional[Dict[str, Any]] = None


class SessionQueryRequest(BaseModel):
    question: str
    device_id: str
    state: str = ""
    database_config: Optional[Dict[str, Any]] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    authenticated: bool
    username: Optional[str] = None
    role: Optional[str] = None
    full_name: Optional[str] = None
    message: str = ""
