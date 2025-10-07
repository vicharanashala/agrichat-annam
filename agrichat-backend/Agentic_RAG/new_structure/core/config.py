"""
Core configuration and environment management for AgriChat Agentic RAG system.

This module handles environment detection, path configuration, and global settings
for the agricultural AI system.
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EnvironmentConfig:
    """
    Configuration class for managing environment-specific settings.
    
    Attributes:
        environment: Deployment environment (Docker or Local)
        chroma_db_path: Path to ChromaDB storage
        use_fast_mode: Enable fast response handler
        ollama_model: Default Ollama model name
        ollama_host: Ollama service host
        debug_mode: Enable debug logging
    """
    environment: str
    chroma_db_path: str
    use_fast_mode: bool
    ollama_model: str
    ollama_host: str
    debug_mode: bool
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not os.path.exists(self.chroma_db_path):
            logging.warning(f"ChromaDB path does not exist: {self.chroma_db_path}")


class ConfigManager:
    """
    Centralized configuration manager for the AgriChat system.
    
    This class provides a single point of configuration management,
    environment detection, and settings validation.
    """
    
    def __init__(self):
        """Initialize configuration manager with environment detection."""
        self._config = self._detect_environment()
        self._setup_logging()
        
    def _detect_environment(self) -> EnvironmentConfig:
        """
        Detect deployment environment and configure paths accordingly.
        
        Returns:
            EnvironmentConfig: Environment-specific configuration
        """
        # Environment detection based on Docker container presence
        is_docker = os.path.exists("/app")
        environment = "Docker" if is_docker else "Local"
        
        # Configure ChromaDB path based on environment
        if is_docker:
            chroma_db_path = "/app/chromaDb"
        else:
            chroma_db_path = "/home/ubuntu/agrichat-annam/agrichat-backend/chromaDb"
        
        # Load environment variables with defaults
        use_fast_mode = os.getenv("USE_FAST_MODE", "true").lower() == "true"
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
        ollama_host = os.getenv("OLLAMA_HOST", "localhost:11434")
        debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        
        return EnvironmentConfig(
            environment=environment,
            chroma_db_path=chroma_db_path,
            use_fast_mode=use_fast_mode,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            debug_mode=debug_mode
        )
    
    def _setup_logging(self) -> None:
        """Configure logging based on environment settings."""
        log_level = logging.DEBUG if self._config.debug_mode else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @property
    def config(self) -> EnvironmentConfig:
        """Get current configuration."""
        return self._config
    
    def get_chroma_path(self) -> str:
        """Get ChromaDB path for current environment."""
        return self._config.chroma_db_path
    
    def get_ollama_config(self) -> Dict[str, str]:
        """
        Get Ollama configuration for LLM initialization.
        
        Returns:
            Dict containing model and base_url for Ollama
        """
        return {
            "model": f"ollama/{self._config.ollama_model}",
            "base_url": f"http://{self._config.ollama_host}",
            "api_key": "not-needed",
            "temperature": 0.0
        }
    
    def is_fast_mode_enabled(self) -> bool:
        """Check if fast response mode is enabled."""
        return self._config.use_fast_mode
    
    def validate_setup(self) -> bool:
        """
        Validate that all required components are available.
        
        Returns:
            bool: True if setup is valid, False otherwise
        """
        issues = []
        
        # Check ChromaDB availability
        if not os.path.exists(self._config.chroma_db_path):
            issues.append(f"ChromaDB not found at {self._config.chroma_db_path}")
        
        # Log configuration status
        logging.info(f"[CONFIG] Environment: {self._config.environment}")
        logging.info(f"[CONFIG] ChromaDB path: {self._config.chroma_db_path}")
        logging.info(f"[CONFIG] ChromaDB exists: {os.path.exists(self._config.chroma_db_path)}")
        logging.info(f"[CONFIG] Fast Mode: {self._config.use_fast_mode}")
        logging.info(f"[CONFIG] Ollama Model: {self._config.ollama_model}")
        logging.info(f"[CONFIG] Ollama Host: {self._config.ollama_host}")
        
        if issues:
            for issue in issues:
                logging.warning(f"[CONFIG] {issue}")
            return False
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary for debugging and monitoring.
        
        Returns:
            Dict containing current configuration state
        """
        return {
            "environment": self._config.environment,
            "chroma_path": self._config.chroma_db_path,
            "chroma_exists": os.path.exists(self._config.chroma_db_path),
            "fast_mode": self._config.use_fast_mode,
            "ollama_model": self._config.ollama_model,
            "ollama_host": self._config.ollama_host,
            "debug_mode": self._config.debug_mode
        }


# Global configuration instance
config_manager = ConfigManager()

def get_config() -> EnvironmentConfig:
    """Get global configuration instance."""
    return config_manager.config

def get_chroma_path() -> str:
    """Get ChromaDB path for current environment."""
    return config_manager.get_chroma_path()

def get_ollama_config() -> Dict[str, str]:
    """Get Ollama configuration for LLM initialization."""
    return config_manager.get_ollama_config()

def is_fast_mode_enabled() -> bool:
    """Check if fast response mode is enabled."""
    return config_manager.is_fast_mode_enabled()

def validate_setup() -> bool:
    """Validate that all required components are available."""
    return config_manager.validate_setup()

def get_config_summary() -> Dict[str, Any]:
    """Get configuration summary for debugging."""
    return config_manager.get_summary()
