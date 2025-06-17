from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from openai import OpenAI
import pandas as pd
import os

class OllamaEmbeddings:
    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts):
        return [self.ollama_embed(text) for text in texts]

    def embed_query(self, text):
        return self.ollama_embed(text)

    def ollama_embed(self, text: str):
        response = self.client.embeddings.create(
            model='nomic-embed-text:latest',
            input=text
        )
        return response.data[0].embedding


class ChromaDBBuilder:
    def __init__(self, csv_path, persist_dir, base_url='http://localhost:11434/v1'):
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.client = OpenAI(base_url=base_url)
        self.embedding_function = OllamaEmbeddings(self.client)
        self.documents = []

    def load_csv_to_documents(self):
        df = pd.read_csv(self.csv_path)
        self.documents = [
            Document(
                page_content=f"Question: {row['questions']}\nAnswer: {row['answers']}",
                metadata={"row": i}
            )
            for i, row in df.iterrows()
        ]
        print(f"[INFO] Prepared {len(self.documents)} documents for embedding and storage.")

    def store_documents_to_chroma(self):
        if not self.documents:
            raise ValueError("No documents to store. Run load_csv_to_documents() first.")

        os.makedirs(self.persist_dir, exist_ok=True)

        db = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embedding_function,
            persist_directory=self.persist_dir
        )
        print(f"[SUCCESS] Stored {len(self.documents)} documents in Chroma at: {self.persist_dir}")



if __name__ == "__main__":
    csv_file = r"C:\Users\amank\agrichat-annam\RAG pipeline v2\data\sample_data.csv"
    storage_dir = r"C:\Users\amank\agrichat-annam\RAG pipeline v2\ChromaDb"

    builder = ChromaDBBuilder(csv_path=csv_file, persist_dir=storage_dir)
    builder.load_csv_to_documents()
    builder.store_documents_to_chroma()
