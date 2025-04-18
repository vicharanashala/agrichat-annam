from langchain_community.vectorstores import Chroma
from openai import OpenAI
from pprint import pprint
import numpy as np
from numpy.linalg import norm

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

class ChromaQueryHandler:
    PROMPT_TEMPLATE = """
You are an AI assistant specialized in agricultural advisory. Use only the provided context to answer the user's question. If no relevant data is found, respond with "I don't have enough information to answer that."

### Guidelines:
You are an AI assistant that provides helpful and factual agricultural advice.

Use only the given context to answer the user's question. Do not make up any information. If no relevant answer is found in the context, say: "I don't have enough information to answer that."

Read the context carefully and extract the most relevant insights. Present them in a clean, organized format. Use headings or bullet points **only when necessary** to improve clarity.

Focus on information such as:
- Crop or plant involved
- Nature of the problem or query (e.g., disease, soil, yield, pesticide use, timing)
- Any symptoms, causes, or challenges
- Expert recommendations, treatments, or best practices

Avoid repeating raw lines from context. Instead, synthesize information clearly.

### Format:
Present your answer in a clean format using bullet points or short paragraphs with headings where needed.

### Context:
{context}

---

### User Question:
{question}

---

### Your Answer:
"""

    def __init__(self, chroma_path, model_name="gemma3:1b", base_url='http://192.168.1.67:11434/v1', relevance_threshold=0.5):
        self.chroma_path = chroma_path
        self.model_name = model_name
        self.relevance_threshold = relevance_threshold

        self.client = OpenAI(base_url=base_url, api_key='ollama')
        self.embedding_function = OllamaEmbeddings(self.client)

        self.db = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embedding_function
        )

    def cosine_sim(self, a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    def rerank_documents(self, question, results, top_k=5):
        query_embedding = self.embedding_function.embed_query(question)

        reranked = sorted(
            results,
            key=lambda x: self.cosine_sim(query_embedding,
                                          self.embedding_function.embed_query(x[0].page_content)),
            reverse=True
        )

        return [doc.page_content for doc, _ in reranked[:top_k]]

    def construct_prompt(self, context, question):
        return self.PROMPT_TEMPLATE.format(context=context, question=question)

    def get_answer(self, question: str) -> str:
        raw_results = self.db.similarity_search_with_score(question, k=10)

        relevant_docs = self.rerank_documents(question, raw_results)

        if not relevant_docs:
            return "No relevant information found in the knowledge base."

        context = "\n\n".join(relevant_docs)
        prompt = self.construct_prompt(context, question)

        messages = [
            {"role": "system", "content": "You are a helpful assistant who answers agricultural questions using only the context provided."},
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.3
        )

        return response.choices[0].message.content.strip()


