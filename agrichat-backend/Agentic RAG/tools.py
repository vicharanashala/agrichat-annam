from firecrawl import FirecrawlApp, ScrapeOptions
import numpy as np
from numpy.linalg import norm
from crewai.tools import BaseTool
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class FireCrawlWebSearchTool(BaseTool):
    def __init__(self, api_key:str):
        super().__init__(name="FireCrawlWebSearchTool", description="Performs web search using Firecrawl API and returns markdown results.")
        object.__setattr__(self, 'app', FirecrawlApp(api_key=api_key))
        object.__setattr__(self, 'ScrapeOptions', ScrapeOptions)

    def _run(self, query: str, limit: int = 1) -> str:
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


class RAGTool(BaseTool):
    def __init__(
        self,
        chroma_path: str,
        gemini_api_key: str,
        embedding_model: str = "models/text-embedding-004"
    ):
        super().__init__(
            name="RAGTool",
            description="Retrieves the best reranked document from Chroma DB based on the query only."
        )
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=gemini_api_key
        )
        self.db = Chroma(
            persist_directory=chroma_path,
            embedding_function=self.embedding_function,
        )
        col = self.db._collection.get()["metadatas"]
        self.meta_index = {
            field: {m[field] for m in col if field in m and m[field]}
            for field in [
                "Year", "Month", "Day",
                "Crop", "DistrictName", "Season", "Sector", "StateName"
            ]
        }

    def _create_metadata_filter(self, question):
        q = question.lower()
        filt = {}
        for field, vals in self.meta_index.items():
            for val in vals:
                if str(val).lower() in q:
                    filt[field] = val
                    break
        return filt or None

    def cosine_sim(self, a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    def rerank_documents(self, question: str, results, top_k: int = 5):
        query_embedding = self.embedding_function.embed_query(question)
        scored = []
        for doc, _ in results:
            d_emb = self.embedding_function.embed_query(doc.page_content)
            score = self.cosine_sim(query_embedding, d_emb)
            scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored[:top_k] if doc.page_content.strip()]

    def _run(self, question: str) -> str:
        results = self.db.similarity_search(question, k=5)
        best_docs = self.rerank_documents(question, results, top_k=3)
        if best_docs:
            return best_docs[0].page_content
        else:
            return "No relevant documents found."
        


if __name__ == "__main__":
    
    rag_tool = RAGTool(
        chroma_path=r"C:\Users\amank\Gemini_based_processing\chromaDb",
        gemini_api_key="AIzaSyCzS2rkrIU-qed90akvU4sjT43W8UANA5A"
    )
    print(rag_tool._run("What is the best crop for summer in India?"))


