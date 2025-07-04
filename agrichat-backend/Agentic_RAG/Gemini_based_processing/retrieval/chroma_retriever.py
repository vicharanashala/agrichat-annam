"""
Module: retrieval.chroma_retriever
Purpose: Encapsulate Chroma vector store and expose a retriever tool for agent use.
"""

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
import os

class ChromaRetriever:
    """
    Wraps Chroma vector DB and exposes a retriever interface for agentic use.
    """
    def __init__(self, persist_dir, embedding_model="models/text-embedding-004", api_key=None):
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=api_key or os.getenv("GOOGLE_API_KEY")
        )
        self.db = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding_function,
        )

    def retrieve(self, query, k=5, filter=None):
        """
        Retrieves top-k relevant documents for a given query.
        """
        return self.db.similarity_search_with_score(query, k=k, filter=filter)

def get_retriever_tool(chroma_retriever):
    """
    Wraps the retriever as a LangChain tool for agent orchestration.
    """
    return create_retriever_tool(
        retriever=chroma_retriever.db.as_retriever(),
        name="chroma_retriever",
        description="Retrieves agricultural Q/A pairs from the Chroma vector database."
    )

def main():
    """
    Main entry point for testing the retriever.
    """
    persist_dir = r"C:\Users\amank\agrichat-annam\Agentic_RAG\Gemini_based_processing\chromaDb"
    api_key = "AIzaSyDZ2ZOEd9bIwOAHmk4wjVuKrpAP4x56EPI"
    retriever = ChromaRetriever(persist_dir, api_key=api_key)
    results = retriever.retrieve("wheat disease", k=3)
    print(f"[INFO] Retrieved {len(results)} results")
