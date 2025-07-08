from firecrawl import FirecrawlApp, ScrapeOptions
from chroma_query_handler import ChromaQueryHandler

def firecrawl_web_search_tool(query, limit=3):
    app = FirecrawlApp(api_key="fc-3042e1475cda4e51b0ce4fdd6ea58578")
    try:
        scrape_options = ScrapeOptions(formats=["markdown"])
        search_result = app.search(
            query,
            limit=limit,
            scrape_options=scrape_options
        )
        combined_markdown = "\n\n".join(
            [item.get("markdown", "") for item in search_result.data]
        )
        return combined_markdown if combined_markdown else "No relevant web search results found."
    except Exception as e:
        return f"Web search error: {e}"

firecrawl_tool_dict = {
    "name": "FireCrawlWebSearchTool",
    "description": "Performs web search using Firecrawl API and returns markdown results.",
    "function": firecrawl_web_search_tool
}

def rag_tool(question):
    chroma_path = r"C:\Users\amank\Gemini_based_processing\chromaDb"
    gemini_api_key = "AIzaSyDZ2ZOEd9bIwOAHmk4wjVuKrpAP4x56EPI"
    handler = ChromaQueryHandler(chroma_path, gemini_api_key)
    return handler.get_answer(question)

rag_tool_dict = {
    "name": "RAGTool",
    "description": "Retrieves answers from the vectorstore using ChromaQueryHandler.",
    "function": rag_tool
}

if __name__ == "__main__":
    query = "best irrigation practices for brinjal"
    result = firecrawl_tool_dict["function"](query, limit=3)
    print("Firecrawl Web Search Results:\n", result)

    # rag_result = rag_tool("How to cure leaf blight in Potato?")
    # print("RAG Tool Result:\n", rag_result)
