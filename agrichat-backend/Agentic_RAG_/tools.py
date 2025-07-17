from firecrawl import FirecrawlApp, ScrapeOptions
import numpy as np
from pydantic import PrivateAttr
from crewai.tools import BaseTool
from langchain_chroma import Chroma
from .chroma_query_handler import ChromaQueryHandler

class FireCrawlWebSearchTool(BaseTool):
    def __init__(self, api_key:str):
        super().__init__(name="FireCrawlWebSearchTool", description="Performs web search using Firecrawl API and returns markdown results.")
        object.__setattr__(self, 'app', FirecrawlApp(api_key=api_key))
        object.__setattr__(self, 'ScrapeOptions', ScrapeOptions)

    def _run(self, query: str, limit: int = 2) -> str:
        """
        Perform a web search using Firecrawl's search API and return combined markdown content.
        Args:
            query (str): The search query.
            limit (int): Number of search results to retrieve.
        Returns:
            str: Combined markdown content from search results.
        """
        scrape_options = self.ScrapeOptions(formats=["markdown"])
        search_result = self.app.search(query, limit=limit, scrape_options=scrape_options)
        combined_markdown = "\n\n".join([item.get("markdown", "") for item in search_result.data])
        return combined_markdown if combined_markdown else "No relevant web search results found."

_chroma_handler = None  

def inject_chroma_handler(handler: ChromaQueryHandler):
    global _chroma_handler
    _chroma_handler = handler

def get_rag_tool():
    if _chroma_handler is None:
        raise RuntimeError("ChromaQueryHandler not yet injected. Call inject_chroma_handler first.")
    return RAGTool(handler=_chroma_handler)

class RAGTool(BaseTool):
    _handler: any = PrivateAttr()

    def __init__(self, handler: ChromaQueryHandler, **kwargs):
    # def __init__(self, chroma_path, gemini_api_key, **kwargs):
        super().__init__(
            name="rag_tool",
            description="Retrieval-Augmented Generation tool using ChromaDB and Gemini API.",
            **kwargs
        )
        # self._handler = ChromaQueryHandler(chroma_path, gemini_api_key)
        self._handler = handler

    def _run(self, question: str) -> str:
        return self._handler.get_answer(question)
