"""
Local LLM Interface using Ollama
Connects to your locally installed Ollama models
"""

import requests
import json
from typing import List, Dict, Optional

class OllamaLLMInterface:
    """Interface for Ollama-hosted models"""
    
    def __init__(self, ollama_endpoint: str = "http://localhost:11434", model_name: str = "your-model-name"):
        self.ollama_endpoint = ollama_endpoint
        self.model_name = model_name
    
    def generate_content(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> str:
        """
        Generate content using Ollama
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.ollama_endpoint}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60  # Ollama can be slower for large models
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated")
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
    
    def __init__(self, ollama_endpoint: str = "http://localhost:11434", embedding_model: str = "nomic-embed-text"):
        self.ollama_endpoint = ollama_endpoint
        self.embedding_model = embedding_model
    
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


local_llm = OllamaLLMInterface(model_name="your-model-name")
local_embeddings = OllamaEmbeddings(embedding_model="nomic-embed-text")

def run_local_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """Convenience function for LLM inference"""
    return local_llm.generate_content(prompt, temperature, max_tokens)
