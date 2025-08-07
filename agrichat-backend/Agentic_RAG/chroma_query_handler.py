from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import numpy as np
from numpy.linalg import norm
import logging


logger = logging.getLogger("uvicorn.error")


class ChromaQueryHandler:
    STRUCTURED_PROMPT = """
You are an expert agricultural assistant. Using only the provided context (do not mention or reveal any metadata such as date, district, state, or season unless the user asks for it), answer the user's question in a detailed and structured manner. Stay strictly within the scope of the user's question and do not introduce unrelated information.

### Detailed Explanation
- Provide a comprehensive, step-by-step explanation using both the context and your own agricultural knowledge, but only as it directly relates to the user's question.
- Use bullet points, sub-headings, or tables to clarify complex information.
- Reference and explain all relevant data points from the context.
- Briefly define technical terms inline if needed.
- Avoid botanical or scientific explanations that are not relevant to farmers unless explicitly asked.

**If the context does not contain relevant information, reply exactly:**   
`I don't have enough information to answer that.`

### Context
{context}

### User Question
{question}
---
### Your Answer:
"""

    def __init__(self, chroma_path: str, gemini_api_key: str, embedding_model: str = "models/text-embedding-004", chat_model: str = "gemma-3-27b-it"):
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=gemini_api_key
        )
        genai.configure(api_key=gemini_api_key)
        self.genai_model = genai.GenerativeModel(chat_model)
        self.db = Chroma(
            persist_directory=chroma_path,
            embedding_function=self.embedding_function,
        )
        col = self.db._collection.get()["metadatas"]
        self.meta_index = {
            field: {m[field] for m in col if field in m and m[field]}
            for field in [
                "Year","Month","Day",
                "Crop","District","Season","Sector","State"
            ]
        }

    def _create_metadata_filter(self, question):
        q = question.lower()
        filt = {}
        for field, vals in self.meta_index.items():
            for val in vals:
                if str(val).lower() in q:
                    filt[field] = val
                    break
        return filt or None

    def cosine_sim(self, a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    def rerank_documents(self, question: str, results, top_k: int = 10):
        query_embedding = self.embedding_function.embed_query(question)
        scored = []
        for doc, _ in results:
            d_emb = self.embedding_function.embed_query(doc.page_content)
            score = self.cosine_sim(query_embedding, d_emb)
            scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored[:top_k] if doc.page_content.strip()]

    def construct_structured_prompt(self, context: str, question: str) -> str:
        return self.STRUCTURED_PROMPT.format(
            context=context,
            question=question
        )

    def get_answer(self, question: str) -> str:
        try:
            metadata_filter = self._create_metadata_filter(question)
            raw_results = self.db.similarity_search_with_score(question, k=10, filter=metadata_filter)
            relevant_docs = self.rerank_documents(question, raw_results)

            if relevant_docs and relevant_docs[0].page_content.strip() and "I don't have enough information to answer that." not in relevant_docs[0].page_content:
                context = relevant_docs[0].page_content
                prompt = self.construct_structured_prompt(context, question)
                response = self.genai_model.generate_content(
                    contents=prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=1024,
                    )
                )
                return response.text.strip()
            else:
                return "I don't have enough information to answer that."
        except Exception as e:
            logger.error(f"[Error] {e}")
            return "I don't have enough information to answer that."

