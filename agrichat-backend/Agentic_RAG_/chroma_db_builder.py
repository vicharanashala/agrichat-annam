from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import pandas as pd
import os
import shutil

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
    data_folder = r"agrichat-backend\Agentic_RAG_\data\sample_data.csv"
    completed_folder =r"C:\Users\amank\Gemini_based_processing\data_completed"
    storage_dir = r"agrichat-backend\Agentic_RAG_\chromaDb"
    os.makedirs(completed_folder, exist_ok=True)
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCzS2rkrIU-qed90akvU4sjT43W8UANA5A"

    for filename in os.listdir(data_folder):
        if filename.lower().endswith(".csv"):
            csv_file = os.path.join(data_folder, filename)
            builder = ChromaDBBuilder(csv_path=csv_file, persist_dir=storage_dir)
            builder.load_csv_to_documents()
            builder.store_documents_to_chroma()
            shutil.move(csv_file, os.path.join(completed_folder, filename))
            print(f"[INFO] Moved {filename} to {completed_folder}")