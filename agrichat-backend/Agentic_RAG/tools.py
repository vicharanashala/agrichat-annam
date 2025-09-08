from pydantic import PrivateAttr, BaseModel, Field
from crewai.tools import BaseTool
from chroma_query_handler import ChromaQueryHandler
from typing import ClassVar, List, Dict, Optional
from local_llm_interface import run_local_llm
import os
import csv
from datetime import datetime


def log_fallback_to_csv(question: str, answer: str, csv_file: str = "fallback_queries.csv"):
    """Log fallback queries and answers to a CSV file."""
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(['timestamp', 'question', 'answer', 'fallback_reason'])
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, question, answer, 'FallbackAgriTool_called'])


class FireCrawlWebSearchTool(BaseTool):
    pass


class RAGToolSchema(BaseModel):
    question: str = Field(description="The user's question")
    conversation_history: Optional[List[Dict]] = Field(default=None, description="Previous conversation history")
    user_state: str = Field(default="", description="User's state/region")


class RAGTool(BaseTool):
    _handler: any = PrivateAttr()
    _classifier: any = PrivateAttr()
    args_schema = RAGToolSchema

    def __init__(self, chroma_path, **kwargs):
        super().__init__(
            name="rag_tool",
            description="Retrieval-Augmented Generation tool using ChromaDB.",
            **kwargs
        )
        self._handler = ChromaQueryHandler(chroma_path)

    def _run(self, question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = "") -> str:
        """
        Run RAG tool with improved fallback detection
        
        Args:
            question: Current user question
            conversation_history: List of previous Q&A pairs for context
            user_state: User's state/region detected from frontend
            
        Returns:
            Generated response with appropriate source attribution or __FALLBACK__ indicator
        """
        import time
        
        rag_tool_start = time.time()
        print(f"[TIMING] RAGTool._run started for: {question[:50]}...")
        print(f"[DEBUG] RAGTool._run called with:")
        print(f"  question: {question}")
        print(f"  conversation_history: {conversation_history}")
        print(f"  user_state: {user_state}")
        
        import inspect
        sig = inspect.signature(self._handler.get_answer)
        print(f"[DEBUG] get_answer signature: {sig}")
        
        chroma_query_start = time.time()
        answer = self._handler.get_answer(question, conversation_history, user_state)
        chroma_query_time = time.time() - chroma_query_start
        print(f"[TIMING] ChromaDB query took: {chroma_query_time:.3f}s")
        
        fallback_check_start = time.time()
        if answer.startswith("__FALLBACK__"):
            log_fallback_to_csv(question, "Database search failed - using LLM fallback", "fallback_queries.csv")
            fallback_check_time = time.time() - fallback_check_start
            print(f"[TIMING] Fallback check took: {fallback_check_time:.3f}s")
            return "__FALLBACK__"
        
        if answer.strip() == "I don't have enough information to answer that.":
            fallback_check_time = time.time() - fallback_check_start
            print(f"[TIMING] Fallback check took: {fallback_check_time:.3f}s")
            return "__FALLBACK__"
        
        if answer.startswith("__NO_SOURCE__"):
            answer = answer.replace("__NO_SOURCE__", "")
        
        fallback_check_time = time.time() - fallback_check_start
        print(f"[TIMING] Fallback check took: {fallback_check_time:.3f}s")
        
        total_rag_time = time.time() - rag_tool_start
        print(f"[TIMING] TOTAL RAGTool processing took: {total_rag_time:.3f}s")
        
        return answer
    
    def run_with_context(self, question: str, conversation_history: List[Dict]) -> str:
        """
        Helper method to explicitly run with conversation context
        """
        return self._run(question, conversation_history)


class FallbackAgriToolSchema(BaseModel):
    question: str = Field(description="The user's question")
    conversation_history: Optional[List[Dict]] = Field(default=None, description="Previous conversation history")


class FallbackAgriTool(BaseTool):
    _classifier: any = PrivateAttr()
    args_schema = FallbackAgriToolSchema

    FALLBACK_PROMPT: ClassVar[str] = """
You are an expert agricultural assistant specializing in Indian agriculture and farming practices. Focus exclusively on Indian context, regional conditions, and India-specific agricultural solutions. Use your expert knowledge to answer agricultural questions relevant to Indian farmers, soil conditions, climate patterns, and crop varieties suited to different Indian states and regions.

IMPORTANT: All responses must be specific to Indian agricultural context, Indian crop varieties, Indian soil types, Indian climate conditions, and farming practices suitable for Indian farmers.

- When providing advice, always consider Indian monsoon patterns, soil types common in India, and crop varieties developed for Indian conditions.
- Reference Indian agricultural practices, local farming techniques, and solutions available to Indian farmers.
- For fertilizers, pesticides, and agricultural inputs, focus on products and brands available in Indian markets.
- Consider regional variations within India (North Indian plains, South Indian conditions, coastal regions, hill states, etc.).

If the question is a normal greeting or salutation (e.g., "hello," "how are you?", "good morning"), respond gently and politely—don't refuse, but give a soft, appropriate answer. Do NOT answer non-agricultural queries except for such greetings.

- Give detailed, step-by-step advice specific to Indian farming conditions and structure your answer with bullet points, headings, or tables when appropriate.
- Stick strictly to the user's topic within Indian agricultural context; do not introduce unrelated information.
- Always provide sources for your information (LLM knowledge or database) and specify if advice is for Indian conditions.
- Keep the response concise, focused on Indian agriculture, and do not provide any preamble or explanations except for the final answer.

### Detailed Explanation (India-Specific)
- Provide comprehensive, step-by-step explanations using Indian agricultural knowledge and practices that work in Indian climate and soil conditions.
- Reference Indian crop varieties, local farming methods, and region-specific practices.
- Use bullet points, sub-headings, or tables to clarify complex information relevant to Indian farmers.
- Reference Indian agricultural research institutes (ICAR, state agricultural universities) when relevant.
- Consider Indian farming seasons (Kharif, Rabi, Zaid) and monsoon patterns in your advice.
- Briefly define technical terms inline if needed, using Indian context and examples.

### Special Instructions for Disease, Fertilizer, Fungicide, or Tonic Queries (Indian Context)
- For questions about diseases, pest attacks, fertilizers, fungicides, plant tonics, or agricultural inputs:
    - Provide standard recommendations using products and chemicals available in Indian markets
    - Include quick, low-cost household/natural solutions suitable for Indian farmers and readily available Indian materials
    - Consider Indian organic farming practices and traditional Indian agricultural methods
    - Reference Indian brands and suppliers when suggesting specific products

- For each method, explain when/why it may be preferable in Indian conditions, with relevant precautions for Indian climate.
- Always offer both modern (chemical) and traditional/natural Indian farming solutions unless the question strictly forbids one or the other.

### Additional Guidance for Indian Crop Management Questions
- If questions relate to growing specific crops, focus on Indian varieties and cultivation practices suited to Indian conditions.
- Provide information about:
    - Indian crop varieties and their regional suitability
    - Soil preparation techniques for Indian soil types
    - Irrigation methods suitable for Indian water conditions
    - Pest and disease management using Indian-available solutions
    - Fertilizer recommendations based on Indian soil testing and availability
    - Harvest and post-harvest handling for Indian market conditions

### Regional Context Requirements:
- Always consider the diversity of Indian agriculture across different states and regions
- Provide region-specific advice when possible (North vs South India, coastal vs inland, plains vs hills)
- Reference local Indian agricultural practices and traditional knowledge
- Consider local Indian market conditions and crop pricing

### User Question
{question}

---
### Your Answer (India-Specific):
"""

    def __init__(self, **kwargs):
        super().__init__(
            name="fallback_agri_tool",
            description="Fallback LLM tool for general agricultural answering.",
            **kwargs
        )

    def _run(self, question: str, conversation_history: Optional[List[Dict]] = None) -> str:
        print(f"[DEBUG] FallbackAgriTool called with question: {question}")
        print(f"[DEBUG] FallbackAgriTool received conversation history: {len(conversation_history) if conversation_history else 0} entries")
        
        if conversation_history and len(conversation_history) > 0:
            recent_context = []
            
            for entry in conversation_history[-2:]:
                q = entry.get('question', '')
                a = entry.get('answer', '')
                recent_context.append(f"User: {q}")
                recent_context.append(f"Assistant: {a}")
            
            context_text = "\n".join(recent_context)
            
            enhanced_prompt = f"""Here's our recent conversation:
{context_text}

Now the user asks: {question}

Please respond naturally, understanding what they're referring to from our conversation context."""

            prompt = enhanced_prompt
        else:
            prompt = f"User asks: {question}\n\nPlease respond as an agricultural expert."
            
        response_text = run_local_llm(prompt, use_fallback=True)
        
        print(f"[SOURCE] Local LLM used for question: {question}")
        log_fallback_to_csv(question, response_text)
        print(f"[DEBUG] Logged fallback call to CSV")
        
        final_response = response_text.strip()
        if not any(greeting in question.lower() for greeting in ['hi', 'hello', 'hey', 'namaste', 'good morning', 'good afternoon', 'good evening']):
            final_response += "\n\n<small><i>Source: LLM Fallback</i></small>"
        
        return final_response


