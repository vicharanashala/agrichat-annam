from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from pprint import pprint
from typing import List, Dict

loader = CSVLoader(r"C:\Users\amank\Downloads\Agri-chatbot-versions\testCode\data\sample_test_data.csv", encoding="utf-8")
documents = loader.load()
pprint(f"Loaded {len(documents)} documents.")

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

client = OpenAI(base_url='http://localhost:11434/v1')
embedding_function = OllamaEmbeddings(client)


db = Chroma.from_documents(
    documents=documents, 
    embedding=embedding_function,
    persist_directory=r"C:\Users\amank\Downloads\Agri-chatbot-versions\testCode\test_chroma_db"
)

CHROMA_PATH = r"C:\Users\amank\Downloads\Agri-chatbot-versions\testCode\test_chroma_db"
PROMPT_TEMPLATE = """
You are an AI assistant tasked with answering user queries based on the provided context. Use only the information from the context to generate your response. If the context does not contain sufficient information, respond with "I don't have enough information to answer that."

### Guidelines:
1. Provide concise and accurate answers.
2. Include relevant details from the context when necessary.
3. If applicable, summarize key points from the context.

### Context:
{context}

---

### User Question:
{question}

---

### Your Answer:
"""
class ChromaQueryHandler:
    def __init__(self, chroma_path=CHROMA_PATH, model_name="deepseek-r1:1.5b", 
                base_url='http://localhost:11434/v1', relevance_threshold=0.5):
        self.chroma_path = chroma_path
        self.client = OpenAI(base_url=base_url, api_key='ollama')
        self.embedding_function = OllamaEmbeddings(self.client)
        self.db = Chroma(persist_directory=self.chroma_path, 
                        embedding_function=self.embedding_function)
        self.model_name = model_name
        self.relevance_threshold = relevance_threshold

    def query(self, question: str) -> str:
        results = self.db.similarity_search_with_score(question, k=3)
        
        relevant_docs = []
        for doc, score in results:
            if score <= self.relevance_threshold:
                relevant_docs.append(doc.page_content)
        
        if not relevant_docs:
            return "No relevant information found in the knowledge base."
            
        context = "\n\n".join(relevant_docs)
        
        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()

handler = ChromaQueryHandler()
answer = handler.query("regarding fertilizer dose of coconut")
pprint(answer)

