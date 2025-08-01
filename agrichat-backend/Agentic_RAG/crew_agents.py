from tools import FireCrawlWebSearchTool, RAGTool, FallbackAgriTool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

firecrawl_tool = FireCrawlWebSearchTool(api_key=os.getenv("FIRECRAWL_API_KEY"))
gemini_api_key = os.getenv("GOOGLE_API_KEY")
rag_tool = RAGTool(chroma_path=r"C:\Users\dledlab\agrichat-annam\agrichat-backend\Agentic_RAG\chromaDb", gemini_api_key=gemini_api_key)

from crewai import LLM, Agent
llm = LLM(
    model='gemini/gemini-2.5-flash',  # Fixed: changed from gemini-2.5-flash
    api_key=gemini_api_key,
    temperature=0.0
)

fallback_tool = FallbackAgriTool(
    google_api_key=gemini_api_key,
    model="gemini-2.5-flash",  # Fixed: changed from gemini-2.5-flash
    websearch_tool=firecrawl_tool
)

Retriever_Agent = Agent(
    role="Retriever Agent",
    goal=(
        "Route the user's question to the appropriate tool."
        "Use the RAG tool first to answer agricultural queries. "
        "If the RAG tool returns '__FALLBACK__' or cannot answer confidently, "
        "then invoke the fallback tool (LLM plus web search). "
        "For non-agricultural queries, the tools will respond politely declining."
    ),
    backstory=(
        "You do not answer questions yourself. Instead, you decide which tool to call based on the user's query and the tool's responses. "
        "You prioritize using trusted internal knowledge (RAG tool) before falling back to external web search aided responses."
    ),
    verbose=True,
    llm=llm,
    tools=[rag_tool, fallback_tool],
)

def retriever_response(question: str) -> str:
    print(f"[DEBUG] Processing question: {question}")
    rag_response = rag_tool._run(question)
    print(f"[DEBUG] RAG response: {rag_response[:100]}..." if len(rag_response) > 100 else f"[DEBUG] RAG response: {rag_response}")
    
    if rag_response == "__FALLBACK__":
        print("[DEBUG] RAG returned __FALLBACK__, calling fallback tool...")
        fallback_response = fallback_tool._run(question)
        return fallback_response
    return rag_response

Grader_agent = Agent(
    role='Answer Grader',
    goal='Filter out erroneous retrievals',
    backstory=(
        "You are a grader assessing relevance of a retrieved document to a user question."
        "If the document contains keywords related to the user question, grade it as relevant."
        "It does not need to be a stringent test. You have to make sure that the answer is relevant to the question."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

hallucination_grader = Agent(
    role="Hallucination Grader",
    goal="Filter out hallucination",
    backstory=(
        "You are a hallucination grader assessing whether an answer is grounded in / supported by a set of facts."
        "Make sure you meticulously review the answer and check if the response provided is in alignment with the question asked"
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

answer_grader = Agent(
    role="Answer Grader",
    goal="Filter out hallucination from the answer.",
    backstory=(
        "You are a grader assessing whether an answer is useful to resolve a question."
        "Make sure you meticulously review the answer and check if it makes sense for the question asked."
        "If the answer is relevant generate a clear and concise response."
        "If the answer generated is not relevant then perform a websearch using 'fallback_tool'."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[fallback_tool],
)

