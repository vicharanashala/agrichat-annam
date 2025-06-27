from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from pprint import pprint
import numpy as np
from numpy.linalg import norm
import os

class ChromaQueryHandler:
    PROMPT_TEMPLATE = """
You are an AI assistant specialized in agricultural advisory. Use only the provided metadata and context to answer the user's question—nothing more.

### Record Metadata:
- Date:         {Year}-{Month}-{Day}
- Crop:         {Crop}
- District:     {DistrictName}
- Season:       {Season}
- Sector:       {Sector}
- State:        {StateName}

### Context:
{context}

### Guidelines:
1. **Overview**  
   Write one concise sentence framing the issue, using only metadata fields that are directly relevant to the user’s question.

2. **Answer**  
   - Present a clear, structured response that references and summarizes only the relevant parts of the context.  
   - Use headings (e.g., “Control Measures,” “Recommendations”) or bullet points only if they improve clarity.  
   - When the context provides dosages, methods, or names, include them exactly.  
   - Simplify any technical terms with a brief in-line explanation if needed.

3. **Style & Accuracy**  
   - Do not introduce any new facts beyond what’s in the context or metadata.  
   - Avoid self-referential or extraneous preamble.  
   - If the context contains no relevant information, reply exactly:  
     `I don't have enough information to answer that.`

### User Question:
{question}

---

### Your Answer:
"""

    def __init__(self, chroma_path: str, gemini_api_key: str, embedding_model: str = "models/text-embedding-004", chat_model: str = "gemma-3-27b-it"):
        try: 
            self.chat_model = chat_model
            self.relevance_threshold = 0.5
            print("Using Gemini API key:", gemini_api_key[:5]) 
            print("Creating embeddings...")
            self.embedding_function = GoogleGenerativeAIEmbeddings(
                model=embedding_model,
                google_api_key=gemini_api_key
            )
            print("Embeddings ready.")

            genai.configure(api_key=gemini_api_key)
            self.genai_model = genai.GenerativeModel(self.chat_model)

            self.db = Chroma(
                persist_directory=chroma_path,
                embedding_function=self.embedding_function,
            )
            col = self.db._collection.get()["metadatas"]
            self.meta_index = {
                field: {m[field] for m in col if field in m and m[field]}
                for field in [
                    "Year","Month","Day",
                    "Crop","DistrictName","Season","Sector","StateName"
                ]
            }
        except Exception as e:
            print(f"[gemini init error] {e}")

    def _create_metadata_filter(self, question):
        q = question.lower()
        filt = {}
        for field, vals in self.meta_index.items():
            for val in vals:
                if str(val).lower() in q:
                    filt[field] = val
                    break
        return filt or None

    def _get_unique_metadata_values(self, field):
        collection = self.db._collection
        metadatas = collection.get()['metadatas']
        return list(set(meta.get(field, '') for meta in metadatas if meta.get(field)))

    def cosine_sim(self, a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    def rerank_documents(self, question:str, results, top_k:int=5):
        query_embedding = self.embedding_function.embed_query(question)
        scored = []
        for doc, _ in results:
            d_emb = self.embedding_function.embed_query(doc.page_content)
            score = self.cosine_sim(query_embedding, d_emb)
            scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]

    def construct_prompt(self, metadata: dict, context: str, question: str) -> str:
        return self.PROMPT_TEMPLATE.format(
            **metadata,
            context=context,
            question=question
        )

    def get_answer(self, question: str) -> str:
        metadata_filter = self._create_metadata_filter(question)
        raw_results = self.db.similarity_search_with_score(question, k=10, filter=metadata_filter)
        relevant_docs = self.rerank_documents(question, raw_results)

        if not relevant_docs:
            return "No relevant information found in the knowledge base."

        md  = relevant_docs[0].metadata
        context = relevant_docs[0].page_content

        prompt = self.construct_prompt(md, context, question)

        response = self.genai_model.generate_content(
            contents=prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1024,
            )
        )
        return response.text.strip()

if __name__ == "__main__":
    chroma_path = r"agrichat-backend\RAG pipeline v3\chromaDb"
    question = "Give information regarding wheat cultivation, any disease and their cure all information related?"
    gemini_api_key = "AIzaSyDZ2ZOEd9bIwOAHmk4wjVuKrpAP4x56EPI"

    query_handler = ChromaQueryHandler(chroma_path, gemini_api_key)
    answer = query_handler.get_answer(question)
    pprint(f"Answer: {answer}")