"""
Core package initialization for AgriChat Agentic RAG system.
"""

from .config import (
    ConfigManager, 
    EnvironmentConfig, 
    get_config, 
    get_chroma_path, 
    get_ollama_config,
    is_fast_mode_enabled,
    validate_setup,
    get_config_summary
)

from .base import (
    ResponseMode,
    AgentRole,
    QueryContext,
    AgriResponse,
    BaseAgriTool,
    BaseAgriAgent,
    BaseResponseHandler,
    AgriSystemInterface
)

__all__ = [
    # Configuration
    "ConfigManager",
    "EnvironmentConfig", 
    "get_config",
    "get_chroma_path",
    "get_ollama_config",
    "is_fast_mode_enabled",
    "validate_setup",
    "get_config_summary",
    
    # Base classes and interfaces
    "ResponseMode",
    "AgentRole", 
    "QueryContext",
    "AgriResponse",
    "BaseAgriTool",
    "BaseAgriAgent",
    "BaseResponseHandler",
    "AgriSystemInterface"
]
