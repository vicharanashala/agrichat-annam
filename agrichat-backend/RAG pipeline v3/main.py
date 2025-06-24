from langchain_community.vectorstores import Chroma
from openai import OpenAI
from pprint import pprint
import numpy as np
from numpy.linalg import norm
from ollama_embedding import OllamaEmbeddings


class ChromaQueryHandler:
    PROMPT_TEMPLATE = """
You are an AI assistant specialized in agricultural advisory. Use only the provided metadata and context to answer the user's question—nothing more.

### Record Metadata:
- BlockName:    {BlockName}
- Category:     {Category}
- Date:         {Year}-{Month}-{Day}
- Crop:         {Crop}
- District:     {DistrictName}
- QueryType:    {QueryType}
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

    def __init__(self, chroma_path: str, model_name : str="gemma3:4b", base_url:str='http://localhost:11434/v1'):
        self.model_name = model_name
        self.relevance_threshold = 0.5
        self.client = OpenAI(base_url=base_url, api_key='ollama')
        self.embedding_function = OllamaEmbeddings(self.client)
        
        self.db = Chroma(
            persist_directory=chroma_path,
            embedding_function = self.embedding_function,
        )
        col = self.db._collection.get()["metadatas"]
        self.meta_index = {
            field: {m[field] for m in col if field in m and m[field]}
            for field in [
                "BlockName","Category","Year","Month","Day",
                "Crop","DistrictName","QueryType","Season","Sector","StateName"
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

        # print("relevant_docs: ", relevant_docs)
        md  = relevant_docs[0].metadata
        # print(md)
        context = relevant_docs[0].page_content

        # pprint(f"Context: {context}")

        prompt = self.construct_prompt(md, context, question)
        messages = [
            {"role": "system",  "content": "You are a helpful assistant who answers agricultural questions using only the context provided."},
            {"role": "user",    "content": prompt}
        ]
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    chroma_path = r"C:\Users\amank\agrichat-annam\RAG pipeline v3\chromaDb"
    question = "information regarding control of termite in wheat"

    query_handler = ChromaQueryHandler(chroma_path)
    answer = query_handler.get_answer(question)
    pprint(f"Answer: {answer}")
