"""
Local LLM Interface using Ollama
Connects to your locally installed Ollama models
"""

import requests
import json
import os
from typing import List, Dict, Optional

class OllamaLLMInterface:
    """Interface for Ollama-hosted models with dual-model support"""
    
    def __init__(self, ollama_endpoint: str = None, model_name: str = None, fallback_model: str = None):
        self.ollama_endpoint = ollama_endpoint or f"http://{os.getenv('OLLAMA_HOST', 'localhost:11434')}"
        self.model_name = model_name or os.getenv('OLLAMA_MODEL', 'llama3.1:latest')
        self.fallback_model = fallback_model or os.getenv('OLLAMA_FALLBACK_MODEL', 'llama3.1:8b')
        self.session = requests.Session()

    def generate_content(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048, use_fallback: bool = False) -> str:
        """
        Generate content using Ollama with dual-model support
        - Primary model (fast): Llama 3.1 8B for RAG pipeline
        - Fallback model (powerful): Gemma 3 27B for complex queries
        """
        try:
            selected_model = self.fallback_model if use_fallback else self.model_name
            
            print(f"[LLM_DEBUG] ==========================================")
            print(f"[LLM_DEBUG] Model Selection:")
            print(f"[LLM_DEBUG] - Primary model: {self.model_name}")
            print(f"[LLM_DEBUG] - Fallback model: {self.fallback_model}")
            print(f"[LLM_DEBUG] - Use fallback: {use_fallback}")
            print(f"[LLM_DEBUG] - Selected model: {selected_model}")
            print(f"[LLM_DEBUG] - Ollama endpoint: {self.ollama_endpoint}")
            print(f"[LLM_DEBUG] - Temperature: {temperature}")
            print(f"[LLM_DEBUG] - Max tokens: {max_tokens}")
            print(f"[LLM_DEBUG] ==========================================")
            
            if use_fallback:
                print(f"[LLM] Using fallback model: {selected_model}")
            else:
                print(f"[LLM] Using primary model: {selected_model}")
            
            payload = {
                "model": selected_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": 32768,
                    "num_batch": 4096 if not use_fallback else 2048,
                    "num_gpu": 99,
                    "num_thread": 48
                }
            }
            
            print(f"[LLM_DEBUG] Payload options: {payload['options']}")
            print(f"[LLM_DEBUG] Prompt preview: {prompt[:200]}...")
            
            response = self.session.post(
                f"{self.ollama_endpoint}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            
            print(f"[LLM_DEBUG] API response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                generated_response = result.get("response", "No response generated")
                print(f"[LLM_DEBUG] Generated response length: {len(generated_response)} characters")
                print(f"[LLM_DEBUG] Response preview: {generated_response[:200]}...")
                return generated_response
            else:
                print(f"[ERROR] Ollama API returned status {response.status_code}: {response.text}")
                return "I apologize, but I'm having trouble generating a response right now."
                
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to connect to Ollama: {e}")
            return "I apologize, but I'm having trouble connecting to the model right now. Make sure Ollama is running."
        except Exception as e:
            print(f"[ERROR] Unexpected error during inference: {e}")
            return "I apologize, but an error occurred while processing your request."

class OllamaEmbeddings:
    """Ollama embedding interface"""
    
    def __init__(self, ollama_endpoint: str = None, embedding_model: str = None):
        self.ollama_endpoint = ollama_endpoint or f"http://{os.getenv('OLLAMA_HOST', 'localhost:11434')}"
        self.embedding_model = embedding_model or os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text:latest')
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents using Ollama"""
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query using Ollama"""
        try:
            payload = {
                "model": self.embedding_model,
                "prompt": text
            }
            
            response = requests.post(
                f"{self.ollama_endpoint}/api/embeddings",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("embedding", [0.0] * 768)
            else:
                print(f"[ERROR] Ollama embedding API returned status {response.status_code}")
                return [0.0] * 768
                
        except Exception as e:
            print(f"[ERROR] Embedding failed: {e}")
            return [0.0] * 768


local_llm = OllamaLLMInterface(model_name="llama3.1:latest")
local_embeddings = OllamaEmbeddings(embedding_model="nomic-embed-text")

def run_local_llm(prompt: str, temperature: float = 0.1, max_tokens: int = 1024, use_fallback: bool = False) -> str:
    """
    Convenience function for LLM inference with dual-model support
    - use_fallback=False: Uses fast Llama 3.1 8B for RAG pipeline
    - use_fallback=True: Uses powerful Gemma 3 27B for complex fallback queries
    """
    print(f"[RUN_LLM_DEBUG] ==========================================")
    print(f"[RUN_LLM_DEBUG] Function called with:")
    print(f"[RUN_LLM_DEBUG] - use_fallback: {use_fallback}")
    print(f"[RUN_LLM_DEBUG] - temperature: {temperature}")
    print(f"[RUN_LLM_DEBUG] - max_tokens: {max_tokens}")
    print(f"[RUN_LLM_DEBUG] ==========================================")
    
    result = local_llm.generate_content(prompt, temperature, max_tokens, use_fallback)
    
    print(f"[RUN_LLM_DEBUG] Final result length: {len(result)} characters")
    return result
