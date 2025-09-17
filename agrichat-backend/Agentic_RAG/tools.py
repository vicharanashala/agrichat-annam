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


def is_agricultural_query(question: str) -> bool:
    """Detect if a query is agriculture-related using keywords and patterns"""
    question_lower = question.lower()
    
    greetings = ['hi', 'hello', 'hey', 'namaste', 'good morning', 'good afternoon', 'good evening', 'how are you']
    if any(greeting in question_lower for greeting in greetings):
        return True
    
    agri_keywords = [
        'crop', 'crops', 'farming', 'farm', 'agriculture', 'agricultural', 'plant', 'plants', 'seed', 'seeds',
        'soil', 'fertilizer', 'pesticide', 'insecticide', 'fungicide', 'herbicide', 'irrigation', 'water',
        'harvest', 'harvesting', 'yield', 'production', 'cultivation', 'cultivate', 'grow', 'growing',
        'pest', 'pests', 'disease', 'diseases', 'nutrient', 'nutrients', 'organic', 'compost', 'manure',
        'wheat', 'rice', 'maize', 'corn', 'barley', 'millet', 'sorghum', 'sugarcane', 'cotton', 'soybean',
        'tomato', 'potato', 'onion', 'garlic', 'cabbage', 'cauliflower', 'brinjal', 'okra', 'chilli',
        'mango', 'banana', 'apple', 'orange', 'grapes', 'coconut', 'groundnut', 'sunflower', 'mustard',
        'kharif', 'rabi', 'zaid', 'monsoon', 'season', 'weather', 'climate', 'temperature', 'rainfall',
        'field', 'fields', 'garden', 'gardening', 'nursery', 'greenhouse', 'polyhouse', 'drip', 'sprinkler',
        'tractor', 'plough', 'cultivator', 'harrow', 'seeder', 'transplanter', 'thresher', 'combine',
        'organic farming', 'precision agriculture', 'sustainable farming', 'integrated pest management',
        'producer', 'largest producer', 'biggest producer', 'production statistics', 'agricultural state',
        'farming state', 'crop production', 'agricultural output', 'yield statistics', 'farming statistics'
    ]
    
    if any(keyword in question_lower for keyword in agri_keywords):
        return True
    
    non_agri_indicators = [
        'prime minister', 'president', 'politics', 'political', 'government policy', 'election',
        'speed of light', 'physics', 'chemistry', 'mathematics', 'astronomy', 'space', 'nasa', 'mars',
        'moon', 'planet', 'galaxy', 'universe', 'quantum', 'relativity', 'newton', 'einstein',
        'brics', 'population', 'history',
        'classical dance', 'music', 'art', 'literature', 'festival',
        'computer', 'software', 'programming', 'technology', 'internet', 'website', 'app',
        'medicine', 'doctor', 'hospital', 'disease treatment', 'surgery', 'pharmacy',
        'business', 'marketing', 'economics', 'finance', 'stock market', 'investment',
        'sports', 'football', 'cricket', 'tennis', 'olympics', 'games',
        'movie', 'film', 'actor', 'actress', 'cinema', 'entertainment',
        'recipe', 'cooking', 'food preparation', 'kitchen', 'restaurant'
    ]
    
    if any(indicator in question_lower for indicator in non_agri_indicators):
        return False
    
    return True


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

    def _run(self, question: str, conversation_history: Optional[List[Dict]] = None, user_state: str = "", db_filter: str = None) -> str:
        """
        Run RAG tool with improved fallback detection and database filtering
        
        Args:
            question: Current user question
            conversation_history: List of previous Q&A pairs for context
            user_state: User's state/region detected from frontend
            db_filter: Filter to specific database ("golden", "rag", "pops")
            
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
        print(f"  db_filter: {db_filter}")
        
        chroma_query_start = time.time()
        result = self._handler.get_answer_with_source(question, conversation_history, user_state, db_filter)
        chroma_query_time = time.time() - chroma_query_start
        print(f"[TIMING] ChromaDB query took: {chroma_query_time:.3f}s")
        
        print(f"[DEBUG] Query result:")
        print(f"  Source: {result['source']}")
        print(f"  Cosine Similarity: {result['cosine_similarity']:.3f}")
        print(f"  Document Metadata: {result['document_metadata']}")
        
        fallback_check_start = time.time()
        if result['answer'].startswith("__FALLBACK__"):
            log_fallback_to_csv(question, f"Database search failed - using LLM fallback (Source: {result['source']})", "fallback_queries.csv")
            fallback_check_time = time.time() - fallback_check_start
            print(f"[TIMING] Fallback check took: {fallback_check_time:.3f}s")
            return "__FALLBACK__"
        
        if result['answer'].strip() == "I don't have enough information to answer that.":
            fallback_check_time = time.time() - fallback_check_start
            print(f"[TIMING] Fallback check took: {fallback_check_time:.3f}s")
            return "__FALLBACK__"
        
        answer = result['answer']
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
    args_schema = FallbackAgriToolSchema
    
    SYSTEM_PROMPT: ClassVar[str] = """You are an Indian agricultural assistant. Follow these rules strictly:

1. For agricultural questions (farming, crops, livestock, rural development, agricultural production, etc.):
   - Answer directly and helpfully without disclaimers
   - Focus on Indian context, varieties, and practices
   - Provide practical, actionable advice
   - Consider Indian seasons, climate, and regional variations
   - Keep responses concise and helpful
   - DO NOT mention current month/date unless specifically relevant
   - DO NOT add unnecessary disclaimers about being an agricultural assistant

2. For NON-agricultural questions (politics, science, entertainment, general knowledge, etc.):
   - Respond EXACTLY with: "I'm an agricultural assistant focused on Indian farming. I can only help with agriculture-related questions. Please ask about crops, farming practices, soil management, pest control, or other agricultural topics."
   - Do NOT provide any information about the non-agricultural topic
   - Do NOT explain what the topic is about

3. For greetings (hello, hi, namaste, good morning, etc.):
   - Respond politely and introduce yourself as an agricultural assistant
   - Suggest some agricultural topics they can ask about

4. Response format:
   - Be direct and helpful
   - Avoid unnecessary verbosity
   - Focus on practical agricultural information

Question: {question}

Response:"""

    def __init__(self, **kwargs):
        super().__init__(
            name="fallback_agri_tool",
            description="Fallback agricultural assistant for Indian farming queries only.",
            **kwargs
        )

    def _run(self, question: str, conversation_history: Optional[List[Dict]] = None) -> str:
        print(f"[DEBUG] FallbackAgriTool called with question: {question}")
        
        question_lower = question.lower().strip()
        greetings = ['hi', 'hello', 'hey', 'namaste', 'good morning', 'good afternoon', 'good evening', 'how are you']
        
        if any(greeting in question_lower for greeting in greetings):
            return ("Hello! I'm your agricultural assistant specializing in Indian farming. "
                   "I can help you with crop management, soil health, pest control, fertilizers, "
                   "irrigation, farming techniques, and agricultural practices. What would you like to know?")
        
        if not is_agricultural_query(question):
            return ("I'm an agricultural assistant focused on Indian farming. I can only help with agriculture-related questions. "
                   "Please ask about crops, farming practices, soil management, pest control, or other agricultural topics.")
        
        if conversation_history and len(conversation_history) > 0:
            recent_context = []
            for entry in conversation_history[-2:]:
                q = entry.get('question', '')
                a = entry.get('answer', '')
                recent_context.append(f"User: {q}")
                recent_context.append(f"Assistant: {a}")
            
            context_text = "\n".join(recent_context)
            prompt = f"""Previous conversation:
{context_text}

Current question: {question}

{self.SYSTEM_PROMPT.format(question=question)}"""
        else:
            prompt = self.SYSTEM_PROMPT.format(question=question)
        
        response_text = run_local_llm(prompt, use_fallback=True, temperature=0.1)
        
        print(f"[SOURCE] Local LLM used for agricultural question: {question}")
        log_fallback_to_csv(question, response_text)
        
        final_response = response_text.strip()
        if not any(greeting in question.lower() for greeting in greetings):
            final_response += "\n\n<small><i>Source: Fallback to LLM</i></small>"
        
        return final_response


