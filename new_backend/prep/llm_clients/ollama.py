import ollama
from .base import LLMClient
from .config import OLLAMA_HOST

class OllamaClient(LLMClient):
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_HOST)

    def get_completion(self, prompt: str, **kwargs) -> str:
        response = self.client.chat(model="llama3.1", messages=[{'role': 'user', 'content': prompt}], **kwargs)
        return response['message']['content']
