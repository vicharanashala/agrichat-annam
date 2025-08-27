import google.generativeai as genai
from .base import LLMClient
from .config import GEMINI_API_KEY

class GeminiClient(LLMClient):
    def __init__(self):
        if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
            raise ValueError("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
        genai.configure(api_key=GEMINI_API_KEY)

    def get_completion(self, prompt: str, **kwargs) -> str:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, **kwargs)
        return response.text
