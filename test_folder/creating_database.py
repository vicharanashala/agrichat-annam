from openai import OpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import os
import shutil
import onnxruntime

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "markdown_files"  # Path to the directory containing markdown files


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


class ChromaDataStore:
    def __init__(self, data_path=DATA_PATH, chroma_path=CHROMA_PATH):
        self.data_path = data_path
        self.chroma_path = chroma_path
        self.client = OpenAI(base_url='http://localhost:11434/v1')
        self.embedding_function = OllamaEmbeddings(self.client)

    def generate_data_store(self):
        documents = self.load_documents()
        chunks = self.split_text(documents)
        self.save_to_chroma(chunks)

    def load_documents(self):
        loader = DirectoryLoader(self.data_path, glob="*.md")
        documents = loader.load()
        return documents

    def split_text(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=500,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def save_to_chroma(self, chunks: list[Document]):
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)

        db = Chroma.from_documents(
            chunks, embedding=self.embedding_function, persist_directory=self.chroma_path
        )
        db.persist()
        print(f"Saved {len(chunks)} chunks to {self.chroma_path}.")


# Example usage
if __name__ == "__main__":
    data_store = ChromaDataStore()
    data_store.generate_data_store()
    print("Data store generated successfully.")
