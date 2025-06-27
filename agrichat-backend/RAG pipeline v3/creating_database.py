from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from openai import OpenAI
import pandas as pd
import os
from ollama_embedding import OllamaEmbeddings 

class ChromaDBBuilder:
    def __init__(self, csv_path, persist_dir, base_url='http://localhost:11434/v1'):
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.client = OpenAI(base_url=base_url)
        self.embedding_function = OllamaEmbeddings(self.client)
        self.documents = []

    def load_csv_to_documents(self):
        df = pd.read_csv(self.csv_path)
        docs = []
        for _, row in df.iterrows():
            metadata = {
                "Year":         row["Year"],
                "Month":        row["Month"],
                "Day":          row["Day"],
                "Crop":         row.get("Crop", "Others"),
                "DistrictName": row.get("DistrictName", "Others"),
                "Season":       row.get("Season", "Others"),
                "Sector":       row.get("Sector", "Others"),
                "StateName":    row.get("StateName", "Others"),
            }
            meta_str = " | ".join(f"{k}: {v}" for k, v in metadata.items() if v)
            content = (
                f"{meta_str}\n"
                f"Question: {row['QueryText']}\n"
                f"Answer: {row['KccAns']}"
            )
            docs.append(Document(page_content=content, metadata=metadata))
        self.documents = docs
        print(f"[INFO] Prepared {len(docs)} documents") 

    def store_documents_to_chroma(self):
        if not self.documents:
            raise ValueError("No documents to store. Run load_csv_to_documents() first.")

        os.makedirs(self.persist_dir, exist_ok=True)

        db = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embedding_function,
            persist_directory=self.persist_dir,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"[SUCCESS] Stored {len(self.documents)} agricultural Q/A pairs in ChromaDB at: {self.persist_dir}")

if __name__ == "__main__":
    csv_file = r"agrichat-backend\RAG pipeline v3\Data\sample_data.csv"
    storage_dir = r"agrichat-backend\RAG pipeline v3\chromaDb"

    builder = ChromaDBBuilder(csv_path=csv_file, persist_dir=storage_dir)
    builder.load_csv_to_documents()
    builder.store_documents_to_chroma()