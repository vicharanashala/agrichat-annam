from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import pandas as pd
import os

class ChromaDBBuilder:
    def __init__(self, csv_path, persist_dir):
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
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
    storage_dir = r"agrichat-backend\RAG pipeline v3\Gemini_based_processing\chromaDb"
    
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDZ2ZOEd9bIwOAHmk4wjVuKrpAP4x56EPI"
    
    builder = ChromaDBBuilder(csv_path=csv_file, persist_dir=storage_dir)
    builder.load_csv_to_documents()
    builder.store_documents_to_chroma()
