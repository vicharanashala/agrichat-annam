from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from .gemini import GeminiClient
from .ollama import OllamaClient


class LLMWrapper(LLM):
    client: Any = None
    client_type: str = "gemini"

    def __init__(self, client_type: str = "gemini"):
        super().__init__()
        if client_type == "gemini":
            self.client = GeminiClient()
        elif client_type == "ollama":
            self.client = OllamaClient()
        else:
            raise ValueError(f"Unknown client type: {client_type}")

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self.client.get_completion(prompt)

    @property
    def _llm_type(self) -> str:
        return "custom"
