

"""
Fast Response Handler - Single coherent implementation
Provides fast responses by using direct tool calls instead of multi-agent workflows.
Returns structured dicts: {'answer': str, 'source': str, 'similarity': float, 'metadata': dict}
"""

import os
import sys
import logging
from typing import List, Dict, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from tools import RAGTool, FallbackAgriTool
from local_llm_interface import OllamaLLMInterface, local_llm
from database_config import DatabaseConfig

logger = logging.getLogger("uvicorn.error")


class FastResponseHandler:
    """Rule-based response handler that returns structured dicts with metadata."""

    def __init__(self):
        if os.path.exists("/app"):
            chroma_db_path = "/app/chromaDb"
        else:
            chroma_db_path = "/home/ubuntu/agrichat-annam/agrichat-backend/chromaDb"

        logger.info(f"[FAST] Initializing with ChromaDB path: {chroma_db_path}")
        self.rag_tool = RAGTool(chroma_path=chroma_db_path)
        self.fallback_tool = FallbackAgriTool()

        # 3-model architecture: qwen3 reasoning + structured fallback flow
        try:
            # qwen3: reasoning, classification, safety analysis
            self.reasoner_llm = OllamaLLMInterface(model_name=os.getenv('OLLAMA_MODEL_REASONER', 'qwen3:1.7b'))
            
            # For structuring RAG/PoPs content (lighter model for formatting)
            self.structurer_llm = OllamaLLMInterface(model_name=os.getenv('OLLAMA_MODEL_STRUCTURER', 'llama3.1:8b'))
            
            # gpt-oss: direct answers for numerical/technical questions
            fallback_model = os.getenv('OLLAMA_MODEL_FALLBACK', 'gpt-oss:latest')
            self.fallback_llm = OllamaLLMInterface(model_name=fallback_model)
            
            logger.info(f"[FAST] 3-model pipeline: Reasoner={self.reasoner_llm.model_name}, Structurer={self.structurer_llm.model_name}, Fallback={fallback_model}")
            logger.info(f"[FAST] RAG → PoPs → Fallback flow with numerical analysis")
        except Exception as e:
            logger.error(f"[FAST] Failed to initialize pipeline LLMs: {e}")
            self.reasoner_llm = None
            self.structurer_llm = None
            self.fallback_llm = None

        self.simple_greetings = [
            'hi', 'hello', 'hey', 'namaste', 'namaskaram', 'vanakkam',
            'good morning', 'good afternoon', 'good evening', 'good day'
        ]

    def _is_simple_greeting(self, question: str) -> bool:
        q = (question or "").strip().lower()
        return len(q) < 30 and any(g in q for g in self.simple_greetings)

    def _greeting_text(self, user_state: Optional[str] = None) -> str:
        state_context = f" in {user_state}" if user_state else " in India"
        return f"Hello! I'm AgriChat, your agricultural assistant in Indian context. How can I help you today?"

    def get_answer(self, question: str, conversation_history: Optional[List[Dict]] = None,
                    user_state: Optional[str] = None, db_config: Optional[DatabaseConfig] = None) -> Dict:
        """Return a dict with answer and source metadata."""
        # Quick greeting shortcut
        if self._is_simple_greeting(question):
            return {'answer': self._greeting_text(user_state), 'source': 'greeting', 'similarity': 1.0, 'metadata': {}}

        # Stage 0: Stream reasoning/thinking from the reasoner model (if available)
        # This prints token-by-token to stdout so CLI users see the chain-of-thought.
        chain_of_thought = ''
        if self.reasoner_llm:
            try:
                reasoner_prompt = (
                    f"You are an expert agricultural reasoner. Think step-by-step about how you'd approach this question and what"
                    f" you would check in the database before answering. Provide your chain-of-thought (short, tokenized) for the user to see.\n\nQuestion: {question}\n"
                )
                thought_tokens = []
                for ev in self.reasoner_llm.stream_generate(reasoner_prompt, model=self.reasoner_llm.model_name, temperature=0.1):
                    if ev.get('type') == 'token':
                        t = ev.get('text')
                        # Print streaming token so CLI shows horizontal thinking
                        print(t, end='', flush=True)
                        thought_tokens.append(t)
                # newline after streamed thought
                if thought_tokens:
                    print('\n')
                chain_of_thought = ''.join(thought_tokens).strip()
            except Exception as e:
                logger.debug(f"[FAST] Reasoner streaming failed: {e}")
                chain_of_thought = ''

        # Try RAG first (best-effort)
        try:
            rag_result = self.rag_tool._handler.get_answer_with_source(question, conversation_history, user_state)
            logger.info(f"[FAST] RAG result keys: {list(rag_result.keys()) if isinstance(rag_result, dict) else 'non-dict'}")
            try:
                logger.info(f"[FAST] RAG result source={rag_result.get('source')} cosine={rag_result.get('cosine_similarity') or rag_result.get('similarity') or rag_result.get('similarity_score')}")
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"[FAST] RAG lookup failed: {e}")
            rag_result = None

        # Decide whether RAG result is usable. If not, explicitly try PoPs before falling back to LLM.
        def _is_no_info_answer(text: Optional[str]) -> bool:
            if not text:
                return True
            t = text.strip().lower()
            return t.startswith("i don't have enough information") or t.startswith("i don't have enough information to answer")

        use_rag = False
        if rag_result and rag_result.get('answer') and rag_result.get('answer') != "__FALLBACK__":
            ans_text = rag_result.get('answer')
            cosine = rag_result.get('cosine_similarity') or rag_result.get('similarity') or rag_result.get('similarity_score') or 0.0
            if (not _is_no_info_answer(ans_text)) and cosine >= 0.6:
                use_rag = True

        if use_rag:
            # Good RAG result — structure and return
            structured_text = rag_result.get('answer')
            if self.structurer_llm:
                try:
                    struct_prompt = (
                        f"You are a content structurer. Given the following factual content from a database, "
                        f"present a clear, concise, farmer-friendly answer.\n\nOriginal question: {question}\n\nDB Content:\n{rag_result.get('answer')}\n"
                    )
                    structured_text = self.structurer_llm.generate_content(struct_prompt, temperature=0.2)
                except Exception:
                    structured_text = rag_result.get('answer')

            return {
                'answer': structured_text,
                'source': rag_result.get('source', 'rag'),
                'similarity': rag_result.get('similarity_score') or rag_result.get('cosine_similarity') or 1.0,
                'metadata': rag_result.get('metadata') or rag_result.get('document_metadata') or {}
            }

        # RAG was not sufficient; explicitly try PoPs then LLM
        logger.info(f"[FAST] RAG not used (use_rag={use_rag}), attempting PoPs lookup next")
        try:
            pops_result = self.rag_tool._handler.get_answer_with_source(question, conversation_history, user_state, database_selection=['pops', 'llm'])
            logger.info(f"[FAST] PoPs result keys: {list(pops_result.keys()) if isinstance(pops_result, dict) else 'non-dict'}")
            try:
                logger.info(f"[FAST] PoPs result source={pops_result.get('source')} cosine={pops_result.get('cosine_similarity') or pops_result.get('similarity')}")
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"[FAST] PoPs lookup failed: {e}")
            pops_result = None

        if pops_result and pops_result.get('answer') and not _is_no_info_answer(pops_result.get('answer')) and pops_result.get('answer') != "__FALLBACK__":
            structured_text = pops_result.get('answer')
            if self.structurer_llm:
                try:
                    struct_prompt = (
                        f"You are a content structurer. Given the following factual content from a PoPs database, "
                        f"present a clear, concise, farmer-friendly answer.\n\nOriginal question: {question}\n\nPoPs Content:\n{pops_result.get('answer')}\n"
                    )
                    structured_text = self.structurer_llm.generate_content(struct_prompt, temperature=0.2)
                except Exception:
                    structured_text = pops_result.get('answer')

            return {
                'answer': structured_text,
                'source': pops_result.get('source', 'pops'),
                'similarity': pops_result.get('cosine_similarity') or pops_result.get('similarity') or 0.0,
                'metadata': pops_result.get('document_metadata') or pops_result.get('metadata') or {}
            }

        # Fallback with gpt-oss for numerical/technical questions
        try:
            if self.fallback_llm:
                # Prompt for numerical analysis
                enhanced_prompt = f"""You are an agricultural expert specializing in numerical analysis and technical data. 
Be direct and specific with numerical values, rates, and measurements. Don't assume typos - treat numbers as given.

If the question contains specific numerical values (like kg/ha, ppm, percentages), analyze them directly and provide:
1. Whether the value is appropriate/excessive/insufficient
2. Typical ranges for comparison  
3. Practical recommendations

Question: {question}

Provide a direct, technical answer:"""
                
                logger.info(f"[FAST] Using gpt-oss fallback for numerical/technical analysis")
                full = []
                for ev in self.fallback_llm.stream_generate(enhanced_prompt, model=self.fallback_llm.model_name, temperature=0.1):
                    if ev.get('type') == 'token':
                        t = ev.get('text')
                        print(t, end='', flush=True)
                        full.append(t)
                print('\n')
                return {'answer': ''.join(full).strip(), 'source': 'gpt_oss_fallback', 'similarity': 0.0, 'metadata': {'model': self.fallback_llm.model_name, 'type': 'numerical_analysis'}}
        except Exception as e:
            logger.debug(f"[FAST] Fallback LLM failed: {e}")

        # Final fallback to tool
        try:
            fb = self.fallback_tool._run(question, conversation_history)
            return {'answer': fb, 'source': 'fallback_tool', 'similarity': 0.0, 'metadata': {}}
        except Exception:
            return {'answer': "Unable to process request. Please try again.", 'source': 'error', 'similarity': 0.0, 'metadata': {}}

    def get_answer_with_thinking_stream(self, question: str, conversation_history: Optional[List[Dict]] = None,
                    user_state: Optional[str] = None, db_config: Optional[DatabaseConfig] = None) -> Dict:
        """Return answer with thinking process collected for streaming."""
        # Quick greeting shortcut
        if self._is_simple_greeting(question):
            return {'answer': self._greeting_text(user_state), 'source': 'greeting', 'similarity': 1.0, 'metadata': {}, 'thinking': ''}

        # Collect thinking process without printing to stdout
        chain_of_thought = ''
        if self.reasoner_llm:
            try:
                reasoner_prompt = (
                    f"Think step-by-step about this agricultural question. Consider what information would be needed "
                    f"from the database and what factors are important. Keep your thinking concise and focused.\n\nQuestion: {question}\n"
                )
                thought_tokens = []
                for ev in self.reasoner_llm.stream_generate(reasoner_prompt, model=self.reasoner_llm.model_name, temperature=0.1):
                    if ev.get('type') == 'token':
                        t = ev.get('text')
                        thought_tokens.append(t)
                chain_of_thought = ''.join(thought_tokens).strip()
            except Exception as e:
                logger.debug(f"[FAST] Reasoner streaming failed: {e}")
                chain_of_thought = ''

        # Try RAG first (same logic as get_answer)
        try:
            rag_result = self.rag_tool._handler.get_answer_with_source(question, conversation_history, user_state)
        except Exception as e:
            logger.debug(f"[FAST] RAG lookup failed: {e}")
            rag_result = None

        def _is_no_info_answer(text: Optional[str]) -> bool:
            if not text:
                return True
            t = text.strip().lower()
            return t.startswith("i don't have enough information") or t.startswith("i don't have enough information to answer")

        use_rag = False
        if rag_result and rag_result.get('answer') and rag_result.get('answer') != "__FALLBACK__":
            ans_text = rag_result.get('answer')
            cosine = rag_result.get('cosine_similarity') or rag_result.get('similarity') or rag_result.get('similarity_score') or 0.0
            if (not _is_no_info_answer(ans_text)) and cosine >= 0.6:
                use_rag = True

        if use_rag:
            structured_text = rag_result.get('answer')
            if self.structurer_llm:
                try:
                    struct_prompt = (
                        f"You are a content structurer. Present this agricultural information in a clear, farmer-friendly way.\n\n"
                        f"Question: {question}\n\nContent:\n{rag_result.get('answer')}\n"
                    )
                    structured_text = self.structurer_llm.generate_content(struct_prompt, temperature=0.2)
                except Exception:
                    structured_text = rag_result.get('answer')

            return {
                'answer': structured_text,
                'thinking': chain_of_thought,
                'source': rag_result.get('source', 'rag'),
                'similarity': rag_result.get('similarity_score') or rag_result.get('cosine_similarity') or 1.0,
                'metadata': rag_result.get('metadata') or rag_result.get('document_metadata') or {}
            }

        # Fallback to LLM if RAG fails
        try:
            if self.fallback_llm:
                prompt = self.fallback_tool.SYSTEM_PROMPT.format(question=question)
                full = []
                for ev in self.fallback_llm.stream_generate(prompt, model=self.fallback_llm.model_name, temperature=0.1):
                    if ev.get('type') == 'token':
                        full.append(ev.get('text'))
                llm_answer = ''.join(full).strip()
                
                return {
                    'answer': llm_answer, 
                    'thinking': chain_of_thought,
                    'source': 'llm_fallback', 
                    'similarity': 0.0, 
                    'metadata': {'model': self.fallback_llm.model_name}
                }
        except Exception as e:
            logger.debug(f"[FAST] Fallback LLM failed: {e}")

        # Final fallback
        try:
            fb = self.fallback_tool._run(question, conversation_history)
            return {
                'answer': fb, 
                'thinking': chain_of_thought,
                'source': 'fallback_tool', 
                'similarity': 0.0, 
                'metadata': {}
            }
        except Exception:
            return {
                'answer': "Unable to process request. Please try again.", 
                'thinking': chain_of_thought,
                'source': 'error', 
                'similarity': 0.0, 
                'metadata': {}
            }


fast_handler = FastResponseHandler()

def get_fast_answer(question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = None) -> str:
    """
    Convenience function for fast answer generation
    """
    return fast_handler.get_answer(question, conversation_history, user_state)
