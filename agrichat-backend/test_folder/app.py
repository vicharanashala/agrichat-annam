from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from creating_database import OllamaEmbeddings
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
import re

# Constants
CHROMA_PATH = "chroma"
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

# Initialize FastAPI app
app = FastAPI()

# Allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html
@app.get("/")
async def serve_home():
    return FileResponse("static/index.html")

# Input model
class QueryRequest(BaseModel):
    query: str


class ChromaQueryHandler:
    def __init__(self, chroma_path=CHROMA_PATH, model_name="deepseek-r1:1.5b", base_url='http://localhost:11434/v1', relevance_threshold=0.7):
        self.chroma_path = chroma_path
        self.client = OpenAI(base_url=base_url, api_key='ollama')  # Initialize OpenAI client
        self.embedding_function = OllamaEmbeddings(self.client)  # Pass the client to OllamaEmbeddings
        self.db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_function)
        self.model_name = model_name
        self.relevance_threshold = relevance_threshold  # Set a relevance threshold

    def query_llm_directly(self, query_text):
        """Fallback method to query LLM when RAG database has no relevant context."""
        prompt = f"""
        You are an AI assistant. Answer the following question based on your general knowledge:

        ### User Question:
        {query_text}

        --- 

        ### Your Answer:
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}]
            )
            response_text = response.choices[0].message.content
            print(f"### LLM Response:\n{response_text}")
            response = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL).group(1).strip()
            answer = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
            # Format the response with <think> sections
            formatted_response = (
                f"<think>\n### Query Context:\n{response}\n</think>\n\n"
                f"<think>\n### Answer:\n{answer}\n\n### Sources:\n</think>"
            )
            return {formatted_response}

        except Exception as e:
            print(f"An error occurred while querying the LLM: {e}")

    def search_query(self, query_text):
        results = self.db.similarity_search_with_relevance_scores(query_text, k=3)
        
        # Filter results based on relevance threshold
        filtered_results = [result for result in results if result[1] >= self.relevance_threshold]
        
        if not filtered_results:
            print("No relevant results found in the RAG database. Querying the LLM directly...\n")
            return self.query_llm_directly(query_text)
        
        # Combine context from filtered results
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results])
        
        # Use the updated prompt template
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}]
            )
            response_text = response.choices[0].message.content
            
            sources = [doc.metadata.get("source", None) for doc, _score in filtered_results]
            print("**Response Text:**")
            print(response_text)
            print("**Response Text completed**")
            response = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL).group(1).strip()
            answer = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
            # Format the response with <think> sections
            formatted_response = (
                f"<think>\n### Query Context:\n{response}\n</think>\n\n"
                f"<think>\n### Answer:\n{answer}\n\n### Sources:\n{', '.join(filter(None, sources))}\n</think>"
            )
            return {formatted_response}
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred during query processing: {e}")

# Initialize handler instance
handler = ChromaQueryHandler()

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        result = handler.search_query(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
