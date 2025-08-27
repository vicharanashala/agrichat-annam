from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def get_completion(self, prompt: str, **kwargs) -> str:
        pass
