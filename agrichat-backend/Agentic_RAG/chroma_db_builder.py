from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import pandas as pd
import os
import shutil

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, continue without it
    # The API key can still be set via environment variables
    pass


class ChromaDBBuilder:
    def __init__(self, csv_path, persist_dir):
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        
        # Get API key from environment variable
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )
        
        # Initialize the embedding function with API key from environment
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )
        self.documents = []
        

    def load_csv_to_documents(self):
        df = pd.read_csv(self.csv_path)
        column_map = {
        "DistrictName": "District",
        "StateName": "State",
        "QueryText": "Question",
        "KccAns": "Answer"
        }
        df.rename(columns=column_map, inplace=True)
        standard_columns = [
        "Year", "Month", "Day", "State", "District", 
        "Crop", "Season", "Sector", "Question", "Answer"
        ]
        for col in standard_columns:
            if col not in df.columns:
                df[col] = "Others"
        docs = []
        for _, row in df.iterrows():
            metadata = {
                "Year":          row.get("Year", "Others"),
                "Month":         row.get("Month", "Others"),
                "Day":           row.get("Day", "Others"),
                "State":         row.get("State", "Others"),
                "Crop":          row.get("Crop", "Others"),
                "District":      row.get("District", "Others"),
                "Season":        row.get("Season", "Others"),
                "Sector":        row.get("Sector", "Others"),
            }
            meta_str = " | ".join(f"{k}: {v}" for k, v in metadata.items() if v)
            content = (
                f"{meta_str}\n"
                f"Question: {row['Question']}\n"
                f"Answer: {row['Answer']}"
            )
            print(content)
            docs.append(Document(page_content=content, metadata=metadata))
        self.documents = docs
        print(f"[INFO] Prepared {len(docs)} docs")

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
    csv_file = r"agrichat-annam/agrichat-backend/Agentic RAG/data/sample_data.csv"
    storage_dir = r"agrichat-annam/agrichat-backend/Agentic RAG/chromaDb"

    try:
        builder = ChromaDBBuilder(csv_path=csv_file, persist_dir=storage_dir)
        print(f"[INFO] Processing {csv_file}")
        builder.load_csv_to_documents()
        builder.store_documents_to_chroma()
    except ValueError as e:
        print(f"[ERROR] {e}")
        print("Please set your API key using one of these methods:")
        print("1. Export in terminal: export GOOGLE_API_KEY='your-api-key-here'")
        print("2. Create a .env file with: GOOGLE_API_KEY=your-api-key-here")
        print("3. Set it in your shell profile (~/.zshrc or ~/.bashrc)")
        exit(1)
    
