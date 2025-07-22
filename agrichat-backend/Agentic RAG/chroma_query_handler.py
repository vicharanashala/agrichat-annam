# Agentic Rag/chroma_query_handler.py

import logging
import numpy as np
from numpy.linalg import norm
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os # Import os for path joining if needed, though direct paths are fine here

# Import necessary modules for the external models
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pandas as pd
import json # For loading vocabulary

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

    FALLBACK_PROMPT = """
You are an expert agricultural assistant. The database could not provide a relevant answer. Using only your own training and knowledge, answer the following question in a detailed, structured, and explanatory manner. Do not mention or reveal any metadata such as date, district, state, or season unless the user asks for it. Stay strictly within the scope of the user's question and do not introduce unrelated information.

### Detailed Explanation
- Provide a comprehensive, step-by-step explanation only as it directly relates to the user's question.
- Use bullet points, sub-headings, or tables to clarify complex information.
- Reference and explain all relevant data points.
- Avoid botanical or scientific explanations that are not relevant to farmers unless explicitly asked.

### User Question
{question}

---
### Your Answer:
"""

    def __init__(self, chroma_path: str, gemini_api_key: str,
                 embedding_model: str = "models/text-embedding-004",
                 chat_model: str = "gemma-3-27b-it",
                 agri_model_local_path: str = "models/agritext_bert_checkpoint/",
                 toxicity_model_local_path: str = "models/toxicity_model.h5",
                 toxicity_vocab_local_path: str = "models/toxicity_vocab.json",
                 toxicity_train_data_path: str = "Agentic Rag/data/train.csv" # Path to train.csv for fallback adaptation
                ):
        self.chat_model = chat_model
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=gemini_api_key
        )
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

        # --- Load Agriculture Classifier ---
        self.agri_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.agri_tokenizer = BertTokenizer.from_pretrained(agri_model_local_path)
            self.agri_model = BertForSequenceClassification.from_pretrained(agri_model_local_path)
            self.agri_model.eval()
            self.agri_model.to(self.agri_device)
            logger.info("Agriculture classifier loaded successfully from local path.")
        except Exception as e:
            logger.error(f"Error loading agriculture classifier from '{agri_model_local_path}': {e}. "
                         f"Please ensure the path is correct and the model files are present.")
            self.agri_tokenizer = None
            self.agri_model = None

        # --- Load Bad Text Classifier Components ---
        self.MAX_FEATURES_TOXICITY = 200000
        self.OUTPUT_SEQUENCE_LENGTH_TOXICITY = 1800

        self.toxicity_vectorizer = None
        self.toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'] # Default fallback labels

        # Attempt to load vocabulary from JSON first
        try:
            with open(toxicity_vocab_local_path, 'r', encoding='utf-8') as f:
                loaded_vocab = json.load(f)
            self.toxicity_vectorizer = TextVectorization(max_tokens=self.MAX_FEATURES_TOXICITY,
                                                       output_sequence_length=self.OUTPUT_SEQUENCE_LENGTH_TOXICITY,
                                                       output_mode='int',
                                                       vocabulary=loaded_vocab)
            logger.info("Toxicity vectorizer loaded successfully with saved vocabulary.")
        except FileNotFoundError:
            logger.warning(f"Toxicity vocabulary file not found at '{toxicity_vocab_local_path}'. Attempting to adapt from train.csv.")
            # Fallback: if vocab file not found, try to adapt from train.csv
            try:
                df_toxicity = pd.read_csv(toxicity_train_data_path)
                X_toxicity = df_toxicity['comment_text']
                self.toxicity_vectorizer = TextVectorization(max_tokens=self.MAX_FEATURES_TOXICITY,
                                                           output_sequence_length=self.OUTPUT_SEQUENCE_LENGTH_TOXICITY,
                                                           output_mode='int')
                self.toxicity_vectorizer.adapt(X_toxicity.values)
                self.toxicity_labels = df_toxicity.columns[2:].tolist()
                logger.info("Toxicity vectorizer adapted successfully from train.csv.")
            except FileNotFoundError:
                logger.error(f"train.csv not found at '{toxicity_train_data_path}'. Cannot adapt toxicity vectorizer. Toxicity classification may not work.")
                self.toxicity_vectorizer = None
            except Exception as e:
                logger.error(f"Error adapting toxicity vectorizer from train.csv: {e}")
                self.toxicity_vectorizer = None
        except Exception as e:
            logger.error(f"Error loading toxicity vectorizer vocabulary from '{toxicity_vocab_local_path}': {e}. Toxicity classification may not work.")
            self.toxicity_vectorizer = None

        # Load the bad text classification model
        self.toxicity_model = None # Initialize to None
        try:
            self.toxicity_model = tf.keras.models.load_model(toxicity_model_local_path)
            logger.info("Toxicity model loaded successfully from local path.")
        except Exception as e:
            logger.error(f"Error loading toxicity model from '{toxicity_model_local_path}': {e}. "
                         f"Please check the path and file integrity. Toxicity classification may not work.")


    def _create_metadata_filter(self, question):
        q = question.lower()
        filt = {}
        for field, vals in self.meta_index.items():
            for val in vals:
                # IMPORTANT: Use str(val).lower() to handle non-string metadata values
                # and ensure consistent lowercasing for matching.
                if str(val).lower() in q:
                    filt[field] = val
                    break
        return filt or None

    def cosine_sim(self, a, b):
        # Add a small epsilon to avoid division by zero if norm is 0
        epsilon = 1e-8
        return np.dot(a, b) / (norm(a) * norm(b) + epsilon)

    def rerank_documents(self, question:str, results, top_k:int=5):
        query_embedding = self.embedding_function.embed_query(question)
        scored = []
        for doc, _ in results:
            # Ensure doc.page_content is a string before embedding
            if not isinstance(doc.page_content, str):
                logger.warning(f"Document content is not a string: {type(doc.page_content)}. Skipping.")
                continue
            
            d_emb = self.embedding_function.embed_query(doc.page_content)
            score = self.cosine_sim(query_embedding, d_emb)
            scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        # Filter out documents with empty content after scoring
        return [doc for doc, score in scored[:top_k] if doc.page_content.strip()]

    def construct_structured_prompt(self, context: str, question: str) -> str:
        return self.STRUCTURED_PROMPT.format(
            context=context,
            question=question
        )

    def construct_fallback_prompt(self, question: str) -> str:
        return self.FALLBACK_PROMPT.format(question=question)

    def _predict_agriculture_sentiment(self, text):
        if not self.agri_model or not self.agri_tokenizer:
            logger.warning("Agriculture model or tokenizer not loaded. Cannot perform agriculture classification.")
            # Default to "Agriculture" with low confidence so it might still pass
            # but signals an issue. Or return a specific error flag.
            return "Agriculture", 0.0

        # Ensure text is a string
        if not isinstance(text, str):
            logger.error(f"Input for agriculture sentiment prediction is not a string: {type(text)}")
            return "Non-agriculture", 1.0 # Treat non-string as non-agriculture confidently

        try:
            inputs = self.agri_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {key: val.to(self.agri_device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = self.agri_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                label = "Agriculture" if pred == 1 else "Non-agriculture"
                confidence = probs[0][pred].item()
            return label, confidence
        except Exception as e:
            logger.error(f"Error during agriculture sentiment prediction: {e}")
            return "Non-agriculture", 0.0 # On error, assume non-agriculture with low confidence

    def _predict_toxicity(self, text):
        if not self.toxicity_model or not self.toxicity_vectorizer:
            logger.warning("Toxicity model or vectorizer not loaded. Cannot perform toxicity classification.")
            return {label: False for label in self.toxicity_labels} # Return all false if model not loaded

        # Ensure text is a string
        if not isinstance(text, str):
            logger.error(f"Input for toxicity prediction is not a string: {type(text)}")
            return {label: False for label in self.toxicity_labels} # Treat non-string as non-toxic

        try:
            vectorized_text = self.toxicity_vectorizer([text])
            # Check if vectorized_text is empty or invalid
            if vectorized_text.shape[1] == 0:
                logger.warning(f"Vectorized text for toxicity is empty. Input: '{text}'")
                return {label: False for label in self.toxicity_labels}

            results = self.toxicity_model.predict(vectorized_text, verbose=0)
            
            if not hasattr(self, 'toxicity_labels') or not self.toxicity_labels:
                logger.warning("Toxicity labels not defined. Cannot interpret toxicity prediction results.")
                return {f"unknown_{i}": bool(results[0][i] > 0.5) for i in range(results.shape[1])} # Fallback if labels missing

            # Ensure results array has enough dimensions for labels
            if results.shape[1] != len(self.toxicity_labels):
                 logger.warning(f"Toxicity model output shape ({results.shape[1]}) does not match expected labels ({len(self.toxicity_labels)}).")
                 # Attempt to map best effort or return empty
                 return {f"label_{i}": bool(results[0][i] > 0.5) for i in range(min(results.shape[1], len(self.toxicity_labels)))}

            toxic_flags = {label: bool(results[0][idx] > 0.5) for idx, label in enumerate(self.toxicity_labels)}
            return toxic_flags
        except Exception as e:
            logger.error(f"Error during toxicity prediction: {e}")
            return {label: False for label in self.toxicity_labels} # On error, assume non-toxic


    def get_answer(self, question: str) -> str:
        # Input validation for question
        if not isinstance(question, str) or not question.strip():
            return "Please provide a valid question."

        try:
            # 1. Agriculture Classifier Filter
            agri_label, agri_conf = self._predict_agriculture_sentiment(question)
            # Using a confidence threshold (e.g., 0.7) for non-agriculture
            # You may adjust this based on your model's false positive/negative rates
            if agri_label == "Non-agriculture" and agri_conf > 0.7:
                logger.info(f"User query '{question}' classified as Non-agriculture (Conf: {agri_conf:.2f}).")
                return "Please ask something which is agriculture related."

            # 2. Bad Text Classifier Filter
            toxicity_results = self._predict_toxicity(question)
            # Check if any of the toxic categories are predicted as True
            if any(toxicity_results.values()):
                detected_toxics = [k for k, v in toxicity_results.items() if v]
                logger.warning(f"User query '{question}' detected as harmful: {detected_toxics}.")
                return (f"⚠️ Warning: This text contains harmful content ({', '.join(detected_toxics)}). "
                        "Please rephrase your question appropriately.")

            # Continue with original RAG logic if filters pass
            metadata_filter = self._create_metadata_filter(question)
            raw_results = self.db.similarity_search_with_score(question, k=10, filter=metadata_filter)
            relevant_docs = self.rerank_documents(question, raw_results)

            if relevant_docs and relevant_docs[0].page_content.strip() and "I don't have enough information to answer that." not in relevant_docs[0].page_content:
                context = relevant_docs[0].page_content
                prompt = self.construct_structured_prompt(context, question)
                logger.info("Using structured prompt with retrieved context.")
                response = self.genai_model.generate_content(
                    contents=prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=1024,
                    )
                )
                # Check for "I don't have enough information..." from the structured prompt response
                if response.text.strip() == "I don't have enough information to answer that.":
                    prompt = self.construct_fallback_prompt(question)
                    logger.info("Structured prompt returned 'no info'. Falling back to general knowledge.")
                    response = self.genai_model.generate_content(
                        contents=prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=0.4,
                            max_output_tokens=1024,
                        )
                    )
                return response.text.strip()
            else:
                prompt = self.construct_fallback_prompt(question)
                logger.info("No relevant docs found. Falling back to general knowledge.")
                response = self.genai_model.generate_content(
                    contents=prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.4,
                        max_output_tokens=1024,
                    )
                )
                return response.text.strip()
        except Exception as e:
            logger.error(f"[Critical Error in get_answer] {e}", exc_info=True) # Log full traceback
            return "An unexpected error occurred while processing your request. Please try again later."