
from pydantic import PrivateAttr
from crewai.tools import BaseTool
from chroma_query_handler import ChromaQueryHandler
from typing import ClassVar, List, Dict, Optional
from local_llm_interface import run_local_llm
import os
import csv
from datetime import datetime


def log_fallback_to_csv(question: str, answer: str, csv_file: str = "fallback_queries.csv"):
    """Log fallback queries and answers to a CSV file."""
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(['timestamp', 'question', 'answer', 'fallback_reason'])
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, question, answer, 'FallbackAgriTool_called'])


class FireCrawlWebSearchTool(BaseTool):
    pass


class RAGTool(BaseTool):
    _handler: any = PrivateAttr()
    _classifier: any = PrivateAttr()

    def __init__(self, chroma_path, gemini_api_key, **kwargs):
        super().__init__(
            name="rag_tool",
            description="Retrieval-Augmented Generation tool using ChromaDB and Gemini API.",
            **kwargs
        )
        self._handler = ChromaQueryHandler(chroma_path, gemini_api_key)

    def _run(self, question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = None) -> str:
        """
        Run RAG tool with improved fallback detection
        
        Args:
            question: Current user question
            conversation_history: List of previous Q&A pairs for context
            user_state: User's state/region detected from frontend
            
        Returns:
            Generated response with appropriate source attribution or __FALLBACK__ indicator
        """
        answer = self._handler.get_answer(question, conversation_history, user_state)
        
        # Check for fallback indicators from the new database-first approach
        if answer.startswith("__FALLBACK__"):
            # Log fallback for analysis
            log_fallback_to_csv(question, "Database search failed - using LLM fallback", "fallback_queries.csv")
            return "__FALLBACK__"
        
        if answer.strip() == "I don't have enough information to answer that.":
            return "__FALLBACK__"
        
        if answer.startswith("__NO_SOURCE__"):
            return answer.replace("__NO_SOURCE__", "")
        
        # For successful database responses, remove any metadata exposure
        return answer
    
    def run_with_context(self, question: str, conversation_history: List[Dict]) -> str:
        """
        Helper method to explicitly run with conversation context
        """
        return self._run(question, conversation_history)

class FallbackAgriTool(BaseTool):
    _classifier: any = PrivateAttr()

    FALLBACK_PROMPT: ClassVar[str] = """
You are an expert agricultural assistant. Use your own expert knowledge to answer the user's agricultural question. If the question is a normal greeting or salutation (e.g., “hello,” “how are you?”, “good morning”), respond gently and politely—don’t refuse, but give a soft, appropriate answer. Do NOT answer non-agricultural queries except for such greetings.

- Give detailed, step-by-step advice and structure your answer with bullet points, headings, or tables when appropriate.
- Stick strictly to the user's topic; do not introduce unrelated information.
- Always provide sources for your information (LLM knowledge or database).
- Keep the response concise, focused, and do not provide any preamble or explanations except for the final answer.

### Detailed Explanation
- Provide a comprehensive, step-by-step explanation using both the context and your own agricultural knowledge, but only as it directly relates to the user's question.
- Use bullet points, sub-headings, or tables to clarify complex information.
- Reference and explain all relevant data points from the context.
- Briefly define technical terms inline if needed.
- Avoid detailed botanical or scientific explanations not relevant to farmers unless explicitly asked.

### Special Instructions for Disease, Fertilizer, Fungicide, or Tonic Queries
- Whenever a question relates to disease, pest attacks, fertilizers, fungicides, plant tonics, or similar agricultural inputs—even if not explicitly stated—include:

    -Standard recommendations (chemical fertilizers, fungicides, plant protection chemicals).
    -Quick, low-cost household/natural solutions suitable for farmers seeking alternatives.

- For each method, explain when/why it may be preferable, with any relevant precautions.
- Always offer both professional and practical (DIY) solutions unless the question strictly forbids one or the other.

### Additional Guidance for General Crop Management Questions (e.g., maximizing yield, disease prevention, precautions)
- If a general question is asked about growing a specific crop and the database contains information for that crop, analyze the context of the user’s question (such as disease prevention, yield maximization, or best practices).
- Retrieve and provide all relevant guidance from existing sources about that crop, including:

    - Disease management
    - Best agronomic practices for yield
    - Important precautions and crop requirements
    - Fertilizer and input recommendations
    - Risks/general crop care tips

### To keep responses concise and focused, the answer should:

    - Only address the specific question asked, using clear bullet points, tables, or short sub-headings as needed.
    - Make sure explanations are actionable, practical, and relevant for farmers—avoiding lengthy background or scientific context unless requested.
    - For questions about diseases, fertilizers, fungicides, or tonics:
    - Briefly provide both standard (chemical) and quick low-cost/natural solutions, each with a very short explanation and clear usage or caution notes.
    - For broader crop management questions, summarize key data points (disease management, input use, care tips, risks) in a succinct, easy-to-use manner—only including what's relevant to the query.
    - Never add unrelated information, avoid detailed paragraphs unless multiple issues are asked, and always keep the response direct and farmer-friendly.

- Even if the information does not directly match the question, use context and reasoning to include database data points that could help answer the user’s general query about that crop.
- Synthesize relevant knowledge and sources into a complete, actionable answer.

### User Question
{question}

---
### Your Answer:
"""

    def __init__(self, **kwargs):
        super().__init__(
            name="fallback_agri_tool",
            description="Fallback LLM tool for general agricultural answering.",
            **kwargs
        )

    def _run(self, question: str) -> str:
        print(f"[DEBUG] FallbackAgriTool called with question: {question}")
        prompt = self.FALLBACK_PROMPT.format(question=question)
        response_text = run_local_llm(prompt)
        final_answer = "Source: Local LLM\n\n" + response_text.strip()
        log_fallback_to_csv(question, final_answer)
        print(f"[DEBUG] Logged fallback call to CSV")
        return final_answer


