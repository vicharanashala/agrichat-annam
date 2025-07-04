"""
Module: data_ingestion.chroma_db_builder
Purpose: Load CSV data, convert to LangChain Documents, and store in Chroma vector DB.
"""

import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class ChromaDBBuilder:
    """
    Handles loading CSV data, converting to documents, and storing in Chroma vector DB.
    """
    def __init__(self, csv_path, persist_dir, embedding_model="models/text-embedding-004", api_key=None):
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model=self.embedding_model,
            google_api_key=self.api_key
        )
        self.documents = []

    def load_csv_to_documents(self):
        """
        Loads CSV and converts each row to a LangChain Document with metadata.
        """
        df = pd.read_csv(self.csv_path)
        docs = []
        for _, row in df.iterrows():
            metadata = {
                "Year":         row.get("Year"),
                "Month":        row.get("Month"),
                "Day":          row.get("Day"),
                "Crop":         row.get("Crop", "Others"),
                "DistrictName": row.get("DistrictName", "Others"),
                "Season":       row.get("Season", "Others"),
                "Sector":       row.get("Sector", "Others"),
                "StateName":    row.get("StateName", "Others"),
            }
            meta_str = " | ".join(f"{k}: {v}" for k, v in metadata.items() if v)
            content = (
                f"{meta_str}\n"
                f"Question: {row.get('QueryText', '')}\n"
                f"Answer: {row.get('KccAns', '')}"
            )
            docs.append(Document(page_content=content, metadata=metadata))
        self.documents = docs
        print(f"[INFO] Prepared {len(docs)} documents")

    def store_documents_to_chroma(self):
        """
        Stores the prepared documents in a persistent Chroma vector database.
        """
        if not self.documents:
            raise ValueError("No documents to store. Run load_csv_to_documents() first.")
        os.makedirs(self.persist_dir, exist_ok=True)
        db = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embedding_function,
            persist_directory=self.persist_dir,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"[SUCCESS] Stored {len(self.documents)} documents in ChromaDB at: {self.persist_dir}")

def main():
    """
    Main entry point for data ingestion and Chroma DB building.
    """
    csv_file = r"C:\Users\amank\agrichat-annam\agrichat-backend\RAGpipelinev3\Data\sample_data.csv"
    storage_dir = r"C:\Users\amank\agrichat-annam\Agentic_RAG\Gemini_based_processing\chromaDb"
    api_key = "AIzaSyDZ2ZOEd9bIwOAHmk4wjVuKrpAP4x56EPI"
    builder = ChromaDBBuilder(csv_path=csv_file, persist_dir=storage_dir, api_key=api_key)
    builder.load_csv_to_documents()
    builder.store_documents_to_chroma()
