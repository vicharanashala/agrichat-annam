import os
import json
from pprint import pprint
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from openai import OpenAI  # Import OpenAI library for Ollama setup
import shutil
from langchain.schema import Document


# Custom Ollama Embeddings Class (Based on your setup)
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


# # Option 2: Use OpenAI embeddings (if you have OpenAI API key) - UNCHANGED
# def load_openai_model(openai_api_key):
#  os.environ["OPENAI_API_KEY"] = openai_api_key
#  openai_ef = OpenAIEmbeddings() # You can specify model name here if needed
#  return openai_ef


# 1. Load your CSV data - UNCHANGED
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna()
    df = df.drop_duplicates()
    return df


# 2. Simple Chunking: Each Q&A pair is a chunk - UNCHANGED
def create_chunks(df):
    chunks = []
    for index, row in df.iterrows():
        question = row['questions']
        answer = row['answers']
        combined_text = f"Question: {question}\nAnswer: {answer}" # Combine Q&A for context
        chunks.append({
        'question': question, # Store original question for lookup
        'answer': answer, # Store the answer
        'chunk': combined_text # The combined text is the chunk
        })
    return chunks


# 3. Embeddings using Ollama or OpenAI - UPDATED to accept custom Ollama
def create_embeddings(chunks, embedding_model):
    embeddings = embedding_model.embed_documents([chunk['chunk'] for chunk in chunks])
    return embeddings


# 4. Store in ChromaDB with Question-Answer pairs - UNCHANGED
def store_chunks(chunks, embedding_func, chroma_path):
  """Stores chunks into ChromaDB using the provided function."""
  if os.path.exists(chroma_path):
    shutil.rmtree(chroma_path)


  documents = [Document(page_content=chunk['chunk'], metadata={'question': chunk['question'], 'answer': chunk['answer']})for chunk in chunks]
  
  
  client = chromadb.PersistentClient(path=chroma_path)
  collection = client.get_or_create_collection(name="faq_collection")
 

  metadatas = []
  ids = []
  embedding = embedding_func.embed_documents([doc.page_content for doc in documents])
  for doc in documents:
    ids.append(doc.page_content)
    metadatas.append(doc.metadata)
  collection.add(documents = [doc.page_content for doc in documents], embeddings=embedding, metadatas=metadatas, ids=ids)
 

  print(f"✅ Saved {len(chunks)} semantic chunks to ChromaDB at '{chroma_path}'")


# 5. Main Execution
if __name__ == "__main__":
# Replace with your actual file path and API key/model name
    csv_file = r"C:\Users\amank\Downloads\Agri-chatbot\data\sample_data.csv"
    chroma_path = r"C:\Users\amank\Downloads\Agri-chatbot\chroma"
    # Option 1: Ollama - UPDATED
    use_ollama = True # Set to False to use OpenAI

    # Load data
    df = load_data(csv_file)


    # Create chunks
    chunks = create_chunks(df)


    # Load embedding model - UPDATED
    if use_ollama:
    # Initialize Ollama client and embeddings
        ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama") #  "ollama" is a placeholder and doesn't matter
        embedding_model = OllamaEmbeddings(client=ollama_client)
    else:
    # embedding_model = load_openai_model(openai_api_key)
        raise ValueError("OpenAI is not fully implemented, set `use_ollama = True` or complete OpenAI implementation")


    # Store in ChromaDB
    store_chunks(chunks, embedding_model, chroma_path)


    print("Data chunked, embedded, and stored in ChromaDB.")
