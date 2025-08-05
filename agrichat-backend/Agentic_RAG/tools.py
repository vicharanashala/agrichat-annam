from firecrawl import FirecrawlApp, ScrapeOptions
from pydantic import PrivateAttr
from crewai.tools import BaseTool
import google.generativeai as genai
from chroma_query_handler import ChromaQueryHandler
from typing import ClassVar
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
    def __init__(self, api_key: str):
        super().__init__(name="FireCrawlWebSearchTool", description="Performs web search using Firecrawl API and returns markdown results.")
        object.__setattr__(self, 'app', FirecrawlApp(api_key=api_key))
        object.__setattr__(self, 'ScrapeOptions', ScrapeOptions)

    def _run(self, query: str, limit: int = 2) -> str:
        scrape_options = self.ScrapeOptions(formats=["markdown"])
        search_result = self.app.search(query, limit=limit, scrape_options=scrape_options)
        combined_markdown = "\n\n".join([item.get("markdown", "") for item in search_result.data])
        return combined_markdown if combined_markdown else "No relevant web search results found."


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

    def _run(self, question: str) -> str:
        answer = self._handler.get_answer(question)
        if answer.strip() == "I don't have enough information to answer that.":
            return "__FALLBACK__"
        
        if answer.startswith("__NO_SOURCE__"):
            return answer.replace("__NO_SOURCE__", "")
        
        return "Source: RAG Database\n\n" + answer

class FallbackAgriTool(BaseTool):
    _llm_model: any = PrivateAttr()
    _websearch_tool: any = PrivateAttr()
    _classifier: any = PrivateAttr()

    FALLBACK_PROMPT: ClassVar[str] = """
You are an expert agricultural assistant. Use your own expert knowledge and review any web search information provided to answer the user's agricultural question. If the question is a normal greeting or salutation (e.g., “hello,” “how are you?”, “good morning”), respond gently and politely—don’t refuse, but give a soft, appropriate answer. Do NOT answer non-agricultural queries except for such greetings.

- Give detailed, step-by-step advice and structure your answer with bullet points, headings, or tables when appropriate.
- Stick strictly to the user's topic; do not introduce unrelated information.
- If the web search results are not relevant, do not use them.
- If the web search results are relevant, incorporate them into your answer.
- Always provide sources for your information (LLM knowledge and/or web-based crawl/search).
- Keep the response concise, focused, and do not provide any preamble or explanations except for the final answer.

### Detailed Explanation
- Provide a comprehensive, step-by-step explanation using both the web/context and your own agricultural knowledge, but only as it directly relates to the user's question.
- Use bullet points, sub-headings, or tables to clarify complex information.
- Reference and explain all relevant data points from the context or web.
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

- Even if the information does not directly match the question, use context and reasoning to include database/web data points that could help answer the user’s general query about that crop.
- Synthesize relevant knowledge and sources into a complete, actionable answer.

### User Question
{question}

{web_results}

---
### Your Answer:
"""

    def __init__(self, google_api_key, model: str, websearch_tool, **kwargs):
        super().__init__(
            name="fallback_agri_tool",
            description="Fallback LLM+websearch tool for general agricultural answering.",
            **kwargs
        )
        genai.configure(api_key=google_api_key)
        self._llm_model = genai.GenerativeModel(model)
        self._websearch_tool = websearch_tool

    def _run(self, question: str) -> str:
        print(f"[DEBUG] FallbackAgriTool called with question: {question}")
        
        web_results = self._websearch_tool._run(question, limit=2)
        web_results_str = f"\nWeb search results:\n{web_results}\n" if web_results else ""
        prompt = self.FALLBACK_PROMPT.format(question=question, web_results=web_results_str)
        response = self._llm_model.generate_content(
            contents=prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.4,
                max_output_tokens=4096,
            )
        )
        
        final_answer = "Source: LLM knowledge & Web Search\n\n" + response.text.strip()
        
        log_fallback_to_csv(question, final_answer)
        print(f"[DEBUG] Logged fallback call to CSV")
        
        return final_answer


