"""Ollama interfaces with graceful fallback when golden_pipeline is absent."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, Iterator, List, Optional

import requests

logger = logging.getLogger("agrichat.pipeline.llm_adapter")

_CACHE: Optional[ModuleType] = None


def _load_module() -> Optional[ModuleType]:
    """Try to load the legacy golden_pipeline local LLM adapter.

    Returns None when the optional package isn't present so callers can
    gracefully fall back to the direct Ollama implementation without
    surfacing noisy import errors in the logs.
    """

    global _CACHE
    if _CACHE is not None:
        return _CACHE

    base_path = Path(__file__).resolve().parent.parent
    target = base_path / "golden_pipeline" / "local_llm_interface.py"

    if not target.exists():
        return None

    spec = importlib.util.spec_from_file_location("golden_lo_llm", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load local_llm_interface from {target}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _CACHE = module
    return module


def get_attr(name: str) -> Any:
    module = _load_module()
    if module is None:
        raise ImportError("golden_pipeline local_llm_interface.py not available")
    try:
        return getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ImportError(f"Attribute {name} not found in local_llm_interface") from exc


def _ollama_base_url() -> str:
    host = os.getenv("OLLAMA_HOST", "localhost:11434")
    if host.startswith("http://") or host.startswith("https://"):
        return host
    return f"http://{host}"


def _ollama_timeout() -> int:
    try:
        return int(os.getenv("OLLAMA_TIMEOUT", "180"))
    except ValueError:
        return 180


class _FallbackOllamaEmbeddings:
    """Minimal embeddings client that talks directly to Ollama."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    def _embed(self, text: str) -> List[float]:
        payload = {"model": self.model, "prompt": text}
        url = f"{_ollama_base_url()}/api/embeddings"
        try:
            response = requests.post(url, json=payload, timeout=_ollama_timeout())
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding")
            if isinstance(embedding, list):
                return embedding
            raise ValueError("Embedding response missing 'embedding' list")
        except Exception as exc:  # pragma: no cover - network failure
            logger.error("Ollama embedding request failed: %s", exc)
            raise

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class _FallbackOllamaLLMInterface:
    """Lightweight Ollama text generation client."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("PIPELINE_LLM_MODEL", "gpt-oss:latest")

    def _generate_payload(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        options: Dict[str, Any] = {"temperature": float(temperature)}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        return {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": options,
        }

    def generate_content(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        use_fallback: bool = False,  # Reserved for API compatibility
    ) -> str:
        payload = self._generate_payload(prompt, temperature=temperature, max_tokens=max_tokens, stream=False)
        url = f"{_ollama_base_url()}/api/generate"
        try:
            response = requests.post(url, json=payload, timeout=_ollama_timeout())
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as exc:  # pragma: no cover - network failure
            logger.error("Ollama generate request failed: %s", exc)
            raise

    def stream_generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        payload = self._generate_payload(prompt, temperature=temperature, max_tokens=max_tokens, stream=True)
        url = f"{_ollama_base_url()}/api/generate"
        try:
            with requests.post(url, json=payload, timeout=_ollama_timeout(), stream=True) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        data = json.loads(raw_line)
                    except json.JSONDecodeError:
                        yield {"type": "raw", "data": raw_line.decode("utf-8", errors="ignore")}
                        continue

                    if data.get("done"):
                        yield {"type": "raw", "data": data}
                        break

                    if "error" in data:
                        yield {"type": "error", "message": data.get("error", "Unknown error")}
                        continue

                    token = data.get("response")
                    if token:
                        yield {"type": "token", "text": token}
        except Exception as exc:  # pragma: no cover - network failure
            yield {"type": "error", "message": str(exc)}


def _fallback_run_local_llm(
    prompt: str,
    *,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
) -> str:
    interface = _FallbackOllamaLLMInterface(model_name=model)
    return interface.generate_content(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


try:  # Prefer the dedicated golden_pipeline implementation when available.
    OllamaLLMInterface = get_attr("OllamaLLMInterface")
    local_embeddings = get_attr("local_embeddings")
    run_local_llm = get_attr("run_local_llm")
except (ImportError, FileNotFoundError):
    logger.warning("golden_pipeline local_llm_interface not found; using direct Ollama fallback")
    OllamaLLMInterface = _FallbackOllamaLLMInterface
    local_embeddings = _FallbackOllamaEmbeddings()
    run_local_llm = _fallback_run_local_llm