from firecrawl import FirecrawlApp, ScrapeOptions
from pydantic import PrivateAttr
from crewai.tools import BaseTool
import google.generativeai as genai
from chroma_query_handler import ChromaQueryHandler
from typing import ClassVar


# 1. Firecrawl web search tool
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

# 2. Chroma RAG Tool

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
        return "Source: RAG Database\n\n" + answer

# 3. Fallback Tool (LLM+web search), using classifier
class FallbackAgriTool(BaseTool):
    _llm_model: any = PrivateAttr()
    _websearch_tool: any = PrivateAttr()
    _classifier: any = PrivateAttr()

    FALLBACK_PROMPT: ClassVar[str] = """
You are an expert agricultural assistant. Use your own expert knowledge and also review any web search information provided to answer the user's agricultural question. Do NOT answer non-agricultural queries; if detected, politely decline.

- Give detailed, step-by-step advice and structure your answer with bullet points, headings, or tables when appropriate.
- Do not introduce irrelevant information; stick to the user's topic.
- If the web search results are not relevant, do not use them.
- If the web search results are relevant, incorporate them into your answer.
- Always provide sources for your information.
- If you cannot find a valid response, respond with "Sorry! unable to find a valid response".
- If the user question is not about agriculture, respond with "This tool only answers agricultural queries. Your question does not seem to be about agriculture."
- Do not provide any preamble or explanations except for the final answer.
- Keep the response concise and focused on the question.

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
        # self._classifier = llm_classifier

    def _run(self, question: str) -> str:
        # if not self._classifier.is_agriculture_query(question):
        #     return "This tool only answers agricultural queries. Your question does not seem to be about agriculture."
        web_results = self._websearch_tool._run(question, limit=2)
        web_results_str = f"\nWeb search results:\n{web_results}\n" if web_results else ""
        prompt = self.FALLBACK_PROMPT.format(question=question, web_results=web_results_str)
        response = self._llm_model.generate_content(
            contents=prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.4,
                max_output_tokens=1024,
            )
        )
        return "Source: LLM knowledge & Web Search\n\n" + response.text.strip()


