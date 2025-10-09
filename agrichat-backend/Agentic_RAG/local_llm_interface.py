"""
Local LLM Interface using Ollama
Connects to your locally installed Ollama models
"""

import requests
import json
import os
from typing import List, Dict, Optional

class OllamaLLMInterface:
    """Optimized interface for Ollama-hosted models"""
    
    def __init__(self, ollama_endpoint: str = None, model_name: str = None):
        self.ollama_endpoint = ollama_endpoint or f"http://{os.getenv('OLLAMA_HOST', 'localhost:11434')}"
        self.model_name = model_name or os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
        self.session = requests.Session()

    def stream_generate(self, prompt: str, model: str = None, temperature: float = 0.3):
        """Stream generation from Ollama (line/chunk-level). Yields dict events.

        Events format examples:
        - {'type': 'token', 'text': '...'}
        - {'type': 'done'}
        - {'type': 'error', 'message': '...'}
        - {'type': 'raw', 'data': <raw chunk>}  # raw model thoughts for research
        """
        selected_model = model or self.model_name
        source = str(selected_model).split(':')[0] if selected_model else ''
        yield {'type': 'model', 'model': selected_model, 'source': source}
        try:
            payload = {
                "model": selected_model,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": temperature}
            }

            with self.session.post(f"{self.ollama_endpoint}/api/generate", json=payload, stream=True, timeout=300) as resp:
                if resp.status_code != 200:
                    yield {'type': 'error', 'message': f'Ollama returned status {resp.status_code}'}
                    return

                buffer = ''
                for chunk in resp.iter_lines(decode_unicode=True):
                    if chunk is None:
                        continue
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    try:
                        parsed = json.loads(chunk)
                        yield {'type': 'raw', 'data': parsed, 'model': selected_model, 'source': source}
                        if isinstance(parsed, dict):
                            if 'token' in parsed:
                                buffer += parsed.get('token', '')
                            if 'response' in parsed and parsed.get('response'):
                                buffer += parsed.get('response', '')

                            if '\n' in buffer or len(buffer) > 60:
                                out = buffer
                                buffer = ''
                                yield {'type': 'token', 'text': out, 'model': selected_model, 'source': source}
                    except Exception:
                        buffer += chunk + '\n'
                        if '\n' in buffer or len(buffer) > 60:
                            out = buffer
                            buffer = ''
                            yield {'type': 'token', 'text': out, 'model': selected_model, 'source': source}

                if buffer:
                    yield {'type': 'token', 'text': buffer, 'model': selected_model, 'source': source}
                yield {'type': 'done'}

        except requests.exceptions.RequestException as e:
            yield {'type': 'error', 'message': f'Connection error: {e}'}
        except Exception as e:
            yield {'type': 'error', 'message': f'Unexpected error: {e}'}

    def generate_content(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048) -> str:
        """
        Generate content using optimized single model approach
        """
        try:
            print(f"[LLM_DEBUG] ==========================================")
            print(f"[LLM_DEBUG] Optimized Model Generation:")
            print(f"[LLM_DEBUG] - Model: {self.model_name}")
            print(f"[LLM_DEBUG] - Ollama endpoint: {self.ollama_endpoint}")
            print(f"[LLM_DEBUG] - Temperature: {temperature}")
            print(f"[LLM_DEBUG] - Max tokens: {max_tokens}")
            print(f"[LLM_DEBUG] ==========================================")
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": 32768,
                    "num_batch": 4096,
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
                return "Unable to generate response."
                
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to connect to Ollama: {e}")
            return "Connection error. Check if Ollama is running."
        except Exception as e:
            print(f"[ERROR] Unexpected error during inference: {e}")
            return "Processing error occurred."

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


reasoner_llm = OllamaLLMInterface(model_name=os.getenv('OLLAMA_MODEL_REASONER', 'qwen3:1.7b'))
structurer_llm = OllamaLLMInterface(model_name=os.getenv('OLLAMA_MODEL_STRUCTURER', 'llama3.1:8b'))
fallback_llm = OllamaLLMInterface(model_name=os.getenv('OLLAMA_MODEL_FALLBACK', 'gpt-oss:latest'))

local_embeddings = OllamaEmbeddings(embedding_model=os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text'))


def run_local_llm(prompt: str, temperature: float = 0.1, max_tokens: int = 1024, use_fallback: bool = False, model: Optional[str] = None) -> str:
    """Deterministic role->model routing for Ollama.

    Routing policy (strict):
      - If use_fallback is True -> use the configured fallback model (llama3.1:latest)
      - If model explicitly requests 'reasoner' or matches reasoner model -> use reasoner (qwen3:1.7b)
      - If model explicitly requests 'structurer' or matches structurer model -> use structurer (gemma:latest)
      - If model explicitly requests 'fallback' or matches fallback model -> use fallback (llama3.1:latest)
      - Default: use reasoner (qwen3:1.7b)

    This removes the old "local_llm" backward-compat behavior and prevents automatic multi-model fallback loops.
    """
    print(f"[RUN_LLM_DEBUG] ==========================================")
    print(f"[RUN_LLM_DEBUG] Function called with:")
    print(f"[RUN_LLM_DEBUG] - use_fallback: {use_fallback}")
    print(f"[RUN_LLM_DEBUG] - temperature: {temperature}")
    print(f"[RUN_LLM_DEBUG] - max_tokens: {max_tokens}")
    print(f"[RUN_LLM_DEBUG] - requested model: {model}")
    print(f"[RUN_LLM_DEBUG] ==========================================")

    try:
        if use_fallback:
            return fallback_llm.generate_content(prompt, temperature, max_tokens, use_fallback=True)

        if model:
            m = str(model).strip().lower()
            short = m.split(':')[0]

            if short in ('reasoner',) or short == str(reasoner_llm.model_name).split(':')[0]:
                return reasoner_llm.generate_content(prompt, temperature, max_tokens, use_fallback=False)

            if short in ('structurer',) or short == str(structurer_llm.model_name).split(':')[0]:
                return structurer_llm.generate_content(prompt, temperature, max_tokens, use_fallback=False)

            if short in ('fallback',) or short == str(fallback_llm.model_name).split(':')[0]:
                return fallback_llm.generate_content(prompt, temperature, max_tokens, use_fallback=True)

            print(f"[RUN_LLM_DEBUG] Requested model '{model}' not recognized as a role; defaulting to reasoner")
            return reasoner_llm.generate_content(prompt, temperature, max_tokens, use_fallback=False)

        return reasoner_llm.generate_content(prompt, temperature, max_tokens, use_fallback=False)

    except Exception as e:
        print(f"[RUN_LLM_ERROR] {e}")
        return "Unable to generate response."
    
class LocalLLMCompatibility:
    """Compatibility proxy exposing stream_generate and generate_content
    to preserve existing imports of `local_llm` in the codebase.

    Delegates to reasoner/structurer/fallback named interfaces based on
    the provided `model` argument or `use_fallback` flag. Defaults to
    the reasoner model when unspecified.
    """
    def stream_generate(self, prompt: str, model: str = None, temperature: float = 0.3):
        if model:
            short = str(model).split(':')[0].lower()
            if short == str(reasoner_llm.model_name).split(':')[0] or short == 'reasoner':
                yield from reasoner_llm.stream_generate(prompt, model=reasoner_llm.model_name, temperature=temperature)
                return
            if short == str(structurer_llm.model_name).split(':')[0] or short == 'structurer':
                yield from structurer_llm.stream_generate(prompt, model=structurer_llm.model_name, temperature=temperature)
                return
            if short == str(fallback_llm.model_name).split(':')[0] or short == 'fallback':
                yield from fallback_llm.stream_generate(prompt, model=fallback_llm.model_name, temperature=temperature)
                return

        yield from reasoner_llm.stream_generate(prompt, model=reasoner_llm.model_name, temperature=temperature)

    def generate_content(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048, use_fallback: bool = False, model: str = None) -> str:
        
        if use_fallback:
            return fallback_llm.generate_content(prompt, temperature=temperature, max_tokens=max_tokens, use_fallback=True)

        if model:
            short = str(model).split(':')[0].lower()
            if short == str(reasoner_llm.model_name).split(':')[0] or short == 'reasoner':
                return reasoner_llm.generate_content(prompt, temperature=temperature, max_tokens=max_tokens, use_fallback=False)
            if short == str(structurer_llm.model_name).split(':')[0] or short == 'structurer':
                return structurer_llm.generate_content(prompt, temperature=temperature, max_tokens=max_tokens, use_fallback=False)
            if short == str(fallback_llm.model_name).split(':')[0] or short == 'fallback':
                return fallback_llm.generate_content(prompt, temperature=temperature, max_tokens=max_tokens, use_fallback=True)

        return reasoner_llm.generate_content(prompt, temperature=temperature, max_tokens=max_tokens, use_fallback=False)


local_llm = LocalLLMCompatibility()

