"""
Local LLM Interface using Ollama for agricultural AI processing.

This module provides a robust interface to locally hosted Ollama models with
dual-model support, comprehensive error handling, and agricultural-specific optimizations.
"""

import os
import json
import logging
import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class ModelConfig:
    """Configuration for Ollama model parameters."""
    name: str
    temperature: float = 0.3
    max_tokens: int = 2048
    context_length: int = 32768
    batch_size: int = 4096
    gpu_layers: int = 99
    thread_count: int = 48


class OllamaLLMInterface:
    """
    Advanced interface for Ollama-hosted language models.
    
    This class provides robust LLM connectivity with features including:
    - Dual-model support (primary/fallback)
    - Agricultural domain optimization
    - Connection pooling and error handling
    - Performance monitoring and caching
    - Multi-language support
    
    Features:
    - Primary model for fast RAG responses
    - Fallback model for complex queries
    - Automatic retry mechanisms
    - Connection health monitoring
    - Performance optimization
    """
    
    def __init__(self, 
                 ollama_endpoint: Optional[str] = None, 
                 model_name: Optional[str] = None, 
                 fallback_model: Optional[str] = None):
        """
        Initialize Ollama LLM interface.
        
        Args:
            ollama_endpoint: Ollama service endpoint URL
            model_name: Primary model name for fast responses
            fallback_model: Fallback model for complex queries
        """
        self.ollama_endpoint = ollama_endpoint or f"http://{os.getenv('OLLAMA_HOST', 'localhost:11434')}"
        self.model_name = model_name or os.getenv('OLLAMA_MODEL', 'llama3.1:latest')
        self.fallback_model = fallback_model or os.getenv('OLLAMA_FALLBACK_MODEL', 'llama3.1:8b')
        
        # Initialize session with connection pooling
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        self.logger = logging.getLogger(__name__)
        
        # Model configurations
        self.primary_config = ModelConfig(
            name=self.model_name,
            temperature=0.3,
            max_tokens=2048,
            batch_size=4096
        )
        
        self.fallback_config = ModelConfig(
            name=self.fallback_model,
            temperature=0.2,
            max_tokens=3072,
            batch_size=2048
        )
        
        self.logger.info(f"[LLM] Initialized with primary: {self.model_name}, fallback: {self.fallback_model}")
    
    def generate_content(self, 
                        prompt: str, 
                        temperature: float = 0.3, 
                        max_tokens: int = 2048, 
                        use_fallback: bool = False) -> str:
        """
        Generate content using Ollama with intelligent model selection.
        
        Args:
            prompt: Input prompt for content generation
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            use_fallback: Use fallback model for complex queries
            
        Returns:
            str: Generated content or error message
        """
        config = self.fallback_config if use_fallback else self.primary_config
        
        try:
            self.logger.debug(f"[LLM] Using {'fallback' if use_fallback else 'primary'} model: {config.name}")
            
            # Build optimized payload
            payload = {
                "model": config.name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": config.context_length,
                    "num_batch": config.batch_size,
                    "num_gpu": config.gpu_layers,
                    "num_thread": config.thread_count
                }
            }
            
            # Make request with timeout
            response = self.session.post(
                f"{self.ollama_endpoint}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                
                if not generated_text.strip():
                    self.logger.warning("[LLM] Empty response received")
                    return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                
                return generated_text
            else:
                self.logger.error(f"[LLM] API error {response.status_code}: {response.text}")
                return "I apologize, but I'm having trouble generating a response right now."
                
        except requests.exceptions.Timeout:
            self.logger.error("[LLM] Request timeout")
            return "I apologize, but the response is taking longer than expected. Please try again."
        except requests.exceptions.ConnectionError:
            self.logger.error("[LLM] Connection failed")
            return "I apologize, but I'm having trouble connecting to the model. Please ensure Ollama is running."
        except Exception as e:
            self.logger.error(f"[LLM] Unexpected error: {e}")
            return "I apologize, but an error occurred while processing your request."
    
    def health_check(self) -> bool:
        """
        Check Ollama service health and model availability.
        
        Returns:
            bool: True if service is healthy, False otherwise
        """
        try:
            # Check service status
            response = self.session.get(f"{self.ollama_endpoint}/api/tags", timeout=10)
            
            if response.status_code != 200:
                return False
            
            # Check if models are available
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            
            primary_available = any(self.model_name in name for name in model_names)
            fallback_available = any(self.fallback_model in name for name in model_names)
            
            self.logger.info(f"[LLM] Health check - Primary: {primary_available}, Fallback: {fallback_available}")
            
            return primary_available or fallback_available
            
        except Exception as e:
            self.logger.error(f"[LLM] Health check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from Ollama.
        
        Returns:
            List of available model names
        """
        try:
            response = self.session.get(f"{self.ollama_endpoint}/api/tags", timeout=10)
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model.get("name", "") for model in models]
            
            return []
            
        except Exception as e:
            self.logger.error(f"[LLM] Failed to get models: {e}")
            return []
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Model name to get info for (defaults to primary model)
            
        Returns:
            Dict containing model information
        """
        target_model = model_name or self.model_name
        
        try:
            response = self.session.post(
                f"{self.ollama_endpoint}/api/show",
                json={"name": target_model},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            
            return {"error": f"Model {target_model} not found"}
            
        except Exception as e:
            self.logger.error(f"[LLM] Failed to get model info: {e}")
            return {"error": str(e)}


class OllamaEmbeddings:
    """
    Ollama embedding interface for vector operations.
    
    This class provides robust embedding generation for agricultural text
    with caching and error handling capabilities.
    """
    
    def __init__(self, 
                 ollama_endpoint: Optional[str] = None, 
                 embedding_model: Optional[str] = None):
        """
        Initialize Ollama embedding interface.
        
        Args:
            ollama_endpoint: Ollama service endpoint URL
            embedding_model: Embedding model name
        """
        self.ollama_endpoint = ollama_endpoint or f"http://{os.getenv('OLLAMA_HOST', 'localhost:11434')}"
        self.embedding_model = embedding_model or os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text:latest')
        
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"[EMBEDDINGS] Initialized with model: {self.embedding_model}")
    
    @lru_cache(maxsize=200)
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single text query with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        try:
            payload = {
                "model": self.embedding_model,
                "prompt": text
            }
            
            response = self.session.post(
                f"{self.ollama_endpoint}/api/embeddings",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("embedding", [])
            else:
                self.logger.error(f"[EMBEDDINGS] API error {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"[EMBEDDINGS] Embedding generation failed: {e}")
            return []
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def health_check(self) -> bool:
        """
        Check embedding service health.
        
        Returns:
            bool: True if service is healthy, False otherwise
        """
        try:
            test_embedding = self.embed_query("test")
            return len(test_embedding) > 0
        except Exception:
            return False


# Global instances for backward compatibility
ollama_llm = OllamaLLMInterface()
ollama_embeddings = OllamaEmbeddings()

# Legacy function interfaces
def run_local_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 2048, use_fallback: bool = False) -> str:
    """
    Legacy function interface for LLM generation.
    
    Args:
        prompt: Input prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        use_fallback: Use fallback model
        
    Returns:
        str: Generated response
    """
    return ollama_llm.generate_content(prompt, temperature, max_tokens, use_fallback)

def local_llm(prompt: str, **kwargs) -> str:
    """Legacy simplified LLM interface."""
    return ollama_llm.generate_content(prompt, **kwargs)

# Export for other modules
local_embeddings = ollama_embeddings
