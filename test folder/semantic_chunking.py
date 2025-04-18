import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import shutil
from openai import OpenAI

CHROMA_PATH = r"C:\Users\amank\Downloads\Agri-chatbot\chroma"
CSV_PATH = r"C:\Users\amank\Downloads\Agri-chatbot\data\sample_data.csv"

class OllamaEmbeddings:
    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts):
        return [self.ollama_embed(text) for text in texts]

    def embed_query(self, text):
        return self.ollama_embed(text)

    def ollama_embed(self, text: str):
        response = self.client.embeddings.create(
            model="nomic-embed-text:latest",
            input=text
        )
        return response.data[0].embedding

def load_csv_documents(csv_path: str):
    df = pd.read_csv(csv_path)
    docs = []
    for idx, row in df.iterrows():
        content = f"Q: {row['questions']}\nA: {row['answers']}"
        docs.append(Document(page_content=content, metadata={"source": f"row_{idx}"}))
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
        length_function=len
    )
    return splitter.split_documents(docs)

def store_chunks(chunks, embedding_func, chroma_path):
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    db = Chroma.from_documents(
        chunks,
        embedding=embedding_func,
        persist_directory=chroma_path
    )
    db.persist()
    print(f"✅ Saved {len(chunks)} semantic chunks to ChromaDB at '{chroma_path}'")

if __name__ == "__main__":
    print("🔧 Loading CSV and generating semantic chunks...")

    client = OpenAI(base_url="http://localhost:11434/v1") 
    embedding_func = OllamaEmbeddings(client)

    raw_docs = load_csv_documents(CSV_PATH)
    chunks = split_documents(raw_docs)
    print(f"✅ Split {len(raw_docs)} documents into {len(chunks)} chunks.")
    store_chunks(chunks, embedding_func, CHROMA_PATH)
