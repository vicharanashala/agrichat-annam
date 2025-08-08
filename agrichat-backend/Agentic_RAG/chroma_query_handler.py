from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import numpy as np
from numpy.linalg import norm
import logging
from typing import List, Dict, Optional
from context_manager import ConversationContext

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

### Special Instructions for Disease, Fertilizer, Fungicide, or Tonic Queries
- Whenever a question relates to disease, pest attacks, use of fertilizers, fungicides, plant tonics, or similar agricultural inputs—even if not explicitly stated—include both:
- Standard recommendations involving chemical fertilizers, fungicides, or plant protection chemicals.
- Quick low-cost household/natural solutions suitable for farmers seeking alternative approaches.
- For each method, explain when and why it may be preferable, and note any precautions if relevant.
- Ensure both professional and practical solutions are always offered unless the question strictly forbids one or the other.

### Additional Guidance for General Crop Management Questions (e.g., maximizing yield, disease prevention, necessary precautions)
- If a general question is asked about growing a particular crop (such as groundnut/peanut) and the database contains information related to that crop, analyze the context of the user’s question (such as disease prevention, yield maximization, or best practices).
- Retrieve and provide all relevant guidance from the database about that crop, including:

    - Disease management
    - Best agronomic practices to maximize yield
    - Important precautions and crop requirements
    - Fertilizer and input recommendations
    - Any risks and general crop care tips

### To keep responses concise and focused, the answer should:
    - Only address the specific question asked, using clear bullet points, tables, or short sub-headings as needed.
    - Make sure explanations are actionable, practical, and relevant for farmers—avoiding lengthy background or scientific context unless requested.
    - For questions about diseases, fertilizers, fungicides, or tonics:
    - Briefly provide both standard (chemical) and quick low-cost/natural solutions, each with a very short explanation and clear usage or caution notes.
    - For broader crop management questions, summarize key data points (disease management, input use, care tips, risks) in a succinct, easy-to-use manner—only including what's relevant to the query.
    - Never add unrelated information, avoid detailed paragraphs unless multiple issues are asked, and always keep the response direct and farmer-friendly.

- Even if the database information does not directly match the question, use context and reasoning to include all data points from the database that could help answer the user's general query about that crop.
- Your response should synthesize the relevant parts of the database connected to the user's request, offering a complete, actionable answer.

**If the context does not contain relevant information, reply exactly:**   
`I don't have enough information to answer that.`

### Context
{context}

### User Question
{question}
---
### Your Answer:
"""

    CLASSIFIER_PROMPT = """
You are a smart classifier assistant. Categorize the user query strictly into one of the following categories:

- AGRICULTURE: if the question is related to farming, crops, fertilizers, pests, soil, irrigation, harvest, agronomy, etc.
- GREETING: if it is a greeting, salutation, or polite conversational opening like "hi", "hello", "good morning", etc.
- NON_AGRI: if the question is not agriculture-related or contains inappropriate, offensive, or irrelevant content.

Respond with only one of these words: AGRICULTURE, GREETING, or NON_AGRI.

### User Query:
{question}
### Category:
"""

    GREETING_RESPONSE_PROMPT = """
You are a friendly assistant. The user has sent the following message:

"{question}"

Generate a short, warm, and polite greeting or salutation in response, encouraging the user to ask their farming or agricultural question.
"""

    NON_AGRI_RESPONSE_PROMPT = """
You are an agriculture assistant. The user has sent the following message:

"{question}"

Generate a polite, respectful message to inform the user that you can only answer agriculture-related queries. Do not be rude. If the question is inappropriate or offensive, gently warn them to stay respectful.
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
        
        self.context_manager = ConversationContext(
            max_context_pairs=3,
            max_context_tokens=500
        )
        
        col = self.db._collection.get()["metadatas"]
        self.meta_index = {
            field: {m[field] for m in col if field in m and m[field]}
            for field in [
                "Year", "Month", "Day",
                "Crop", "District", "Season", "Sector", "State"
            ]
        }

    def expand_query(self, query):
        """
        Expand query with variations to handle 'twisted' questions better
        """
        expanded = [query]
        
        try:
            agri_keywords = self._extract_agricultural_terms(query)
            
            if agri_keywords:
                keyword_query = " ".join(agri_keywords)
                expanded.append(keyword_query)
            
            simplified = self._simplify_query(query)
            if simplified and simplified != query:
                expanded.append(simplified)
            
            alternative = self._rephrase_query(query)
            if alternative and alternative not in expanded:
                expanded.append(alternative)
            
        except Exception as e:
            logger.warning(f"[RAG] Error expanding query: {e}")
        
        return expanded[:5]
    
    def _extract_agricultural_terms(self, text):
        """Extract agricultural terms from text"""
        agricultural_terms = {
            'crop', 'crops', 'farming', 'agriculture', 'cultivation', 'harvest', 'seeds', 'soil',
            'fertilizer', 'pesticide', 'irrigation', 'planting', 'sowing', 'yield', 'farm', 'field',
            'rice', 'wheat', 'corn', 'maize', 'cotton', 'sugarcane', 'vegetables', 'fruits',
            'organic', 'disease', 'pest', 'insect', 'weed', 'water', 'rain', 'drought', 'season',
            'growth', 'production', 'farmer', 'agricultural', 'land', 'machinery', 'equipment'
        }
        
        words = text.lower().split()
        found_terms = [word for word in words if word in agricultural_terms]
        return found_terms
    
    def _simplify_query(self, query):
        """Create simplified version of query focusing on core terms"""
        question_words = {'what', 'how', 'when', 'where', 'why', 'which', 'can', 'is', 'are', 'do', 'does', 'did'}
        words = query.lower().split()
        content_words = [word for word in words if word not in question_words and len(word) > 2]
        return " ".join(content_words) if content_words else query
    
    def _rephrase_query(self, query):
        """Create alternative phrasing of the query"""
        if "how to" in query.lower():
            return query.lower().replace("how to", "method for").replace("how can", "way to")
        elif "what is" in query.lower():
            return query.lower().replace("what is", "information about")
        elif query.endswith("?"):
            return query[:-1]
        return query

    def _create_metadata_filter(self, question):
        """
        Create metadata filter from question, but avoid overly restrictive filters
        that might cause ChromaDB query errors
        """
        q = question.lower()
        filt = {}
        
        safe_fields = ["Crop", "District", "State", "Season"]
        
        for field in safe_fields:
            if field in self.meta_index:
                for val in self.meta_index[field]:
                    if str(val).lower() in q and str(val).lower() != "other" and str(val).lower() != "-":
                        filt[field] = val
                        break
        
        if len(filt) > 2:
            filt = dict(list(filt.items())[:2])
            
        return filt or None

    def cosine_sim(self, a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    def rerank_documents(self, question: str, results, top_k: int = 5):
        """Enhanced reranking with better similarity scoring and multiple strategies"""
        query_embedding = self.embedding_function.embed_query(question)
        scored = []
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        for doc, original_score in results:
            content = doc.page_content.strip()
            
            if (content.lower().count('others') > 3 or 
                'Question: Others' in content or 
                'Answer: Others' in content or
                content.count('Others') > 5):
                continue
            
            d_emb = self.embedding_function.embed_query(content)
            cosine_score = self.cosine_sim(query_embedding, d_emb)
            
            content_words = set(content.lower().split())
            keyword_overlap = len(question_words.intersection(content_words)) / len(question_words) if question_words else 0
            
            agri_terms = {'crop', 'plant', 'disease', 'pest', 'fertilizer', 'soil', 'water', 'harvest', 'seed', 'growth', 'yield'}
            question_agri_terms = question_words.intersection(agri_terms)
            content_agri_terms = content_words.intersection(agri_terms)
            agri_overlap = len(question_agri_terms.intersection(content_agri_terms)) / max(len(question_agri_terms), 1)
            
            combined_score = (cosine_score * 0.6) + (keyword_overlap * 0.25) + (agri_overlap * 0.15)
            
            scored.append((doc, combined_score, cosine_score, original_score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, combined_score, cosine_score, orig_score in scored[:top_k] 
                if doc.page_content.strip() and combined_score > 0.1]

    def construct_structured_prompt(self, context: str, question: str) -> str:
        return self.STRUCTURED_PROMPT.format(
            context=context,
            question=question
        )

    def classify_query(self, question: str) -> str:
        prompt = self.CLASSIFIER_PROMPT.format(question=question)
        try:
            response = self.genai_model.generate_content(
                contents=prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0,
                    max_output_tokens=20,
                )
            )
            category = response.text.strip().upper()
            if category in {"AGRICULTURE", "GREETING", "NON_AGRI"}:
                return category
            else:
                return "NON_AGRI"
        except Exception as e:
            logger.error(f"[Classifier Error] {e}")
            return "NON_AGRI"

    def generate_dynamic_response(self, question: str, mode: str) -> str:
        if mode == "GREETING":
            prompt = self.GREETING_RESPONSE_PROMPT.format(question=question)
        else:
            prompt = self.NON_AGRI_RESPONSE_PROMPT.format(question=question)
        try:
            response = self.genai_model.generate_content(
                contents=prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=100,
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"[Dynamic Response Error] {e}")
            return "Sorry, I can only help with agriculture-related questions."

    def get_answer(self, question: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Get answer with optional conversation context for follow-up queries
        
        Args:
            question: Current user question
            conversation_history: List of previous Q&A pairs for context
            
        Returns:
            Generated response
        """
        try:
            processing_query = question
            context_used = False
            
            if conversation_history:
                if self.context_manager.should_use_context(question, conversation_history):
                    processing_query = self.context_manager.build_contextual_query(question, conversation_history)
                    context_used = True
                    logger.info(f"[Context] Using context for follow-up query. Original: {question[:100]}...")
            
            if question.lower().strip() in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'howdy', 'greetings']:
                response = self.generate_dynamic_response(question, mode="GREETING")
                return f"__NO_SOURCE__{response}"

            try:
                expanded_queries = self.expand_query(processing_query)
                all_results = []
                seen_content = set()
                
                for query_variant in expanded_queries:
                    try:
                        metadata_filter = self._create_metadata_filter(query_variant)
                        variant_results = self.db.similarity_search_with_score(query_variant, k=10, filter=metadata_filter)
                        
                        if len(variant_results) < 3:
                            variant_results_no_filter = self.db.similarity_search_with_score(query_variant, k=15, filter=None)
                            variant_results.extend(variant_results_no_filter)
                        
                        for doc, score in variant_results:
                            if doc.page_content not in seen_content:
                                all_results.append((doc, score))
                                seen_content.add(doc.page_content)
                    
                    except Exception as variant_error:
                        logger.warning(f"[RAG] Error with query variant '{query_variant}': {variant_error}")
                        continue
                
                raw_results = all_results[:25]
                
                if not raw_results:
                    logger.warning(f"[RAG] No results from expanded queries, falling back to original query")
                    raw_results = self.db.similarity_search_with_score(processing_query, k=15, filter=None)
                    
            except Exception as filter_error:
                logger.warning(f"[ChromaDB Error] {filter_error}. Using fallback search.")
                raw_results = self.db.similarity_search_with_score(processing_query, k=15, filter=None)
                
            relevant_docs = self.rerank_documents(processing_query, raw_results)

            if relevant_docs and relevant_docs[0].page_content.strip():
                content = relevant_docs[0].page_content.strip()
                
                if (content.lower().count('others') > 3 or 
                    'Question: Others' in content or 
                    'Answer: Others' in content or
                    len(content) < 50):
                    
                    if context_used:
                        logger.info(f"[Context] No good RAG content found, but providing contextual response")
                        return "I understand you're asking about the topic we were discussing, but I don't have specific information in my database to provide a detailed answer. Could you please be more specific about what you'd like to know?"
                    else:
                        logger.info(f"[RAG] No useful content found, checking classification")
                        relevant_docs = [] 
                
                if relevant_docs:  
                    similarity_score = None
                    for doc, score in raw_results:
                        if doc.page_content.strip() == content:
                            similarity_score = score
                            break
                    
                    if context_used:
                        should_use_rag = True
                        logger.info(f"[Context] Using RAG content for contextual query (score: {similarity_score})")
                    else:
                        score_threshold = 0.4  
                        should_use_rag = similarity_score is None or similarity_score <= score_threshold
                    
                    if should_use_rag:
                        final_question = question if context_used else processing_query
                        prompt = self.construct_structured_prompt(content, final_question)
                        
                        response = self.genai_model.generate_content(
                            contents=prompt,
                            generation_config=genai.GenerationConfig(
                                temperature=0.3,
                                max_output_tokens=1024,
                            )
                        )
                        
                        generated_response = response.text.strip()
                        
                        if context_used:
                            logger.info(f"[Context] Response generated with conversation context")
                        else:
                            logger.info(f"[RAG] Response generated from RAG database (score: {similarity_score})")
                            
                        return generated_response
            
            if context_used:
                if relevant_docs and relevant_docs[0].page_content.strip():
                    context = relevant_docs[0].page_content
                    final_question = question
                    prompt = self.construct_structured_prompt(context, final_question)
                    
                    response = self.genai_model.generate_content(
                        contents=prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=0.3,
                            max_output_tokens=1024,
                        )
                    )
                    
                    generated_response = response.text.strip()
                    logger.info(f"[Context] Used marginal RAG content for contextual query")
                    return generated_response
            
            category = self.classify_query(question)
            
            if category == "NON_AGRI":
                response = self.generate_dynamic_response(question, mode="NON_AGRI")
                return f"__NO_SOURCE__{response}"
            
            return "I don't have enough information to answer that."
            
        except Exception as e:
            logger.error(f"[Error] {e}")
            return "I don't have enough information to answer that."

    def get_context_summary(self, conversation_history: List[Dict]) -> Optional[str]:
        """
        Get a brief summary of conversation context for debugging
        """
        return self.context_manager.get_context_summary(conversation_history)
