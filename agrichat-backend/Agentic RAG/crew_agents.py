from tools import FireCrawlWebSearchTool, RAGTool, FallbackAgriTool

firecrawl_tool = FireCrawlWebSearchTool(api_key="fc-3042e1475cda4e51b0ce4fdd6ea58578")
gemini_api_key = "AIzaSyCzS2rkrIU-qed90akvU4sjT43W8UANA5A"
rag_tool = RAGTool(chroma_path=r"C:\Users\amank\Gemini_based_processing\chromaDb", gemini_api_key=gemini_api_key)

from crewai import LLM, Agent
llm = LLM(
    model='gemini/gemini-2.5-flash',
    api_key=gemini_api_key,
    temperature=0.0
)

fallback_tool = FallbackAgriTool(
    google_api_key=gemini_api_key,
    model="gemini-2.5-flash",
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
    rag_response = rag_tool._run(question)
    if rag_response == "__FALLBACK__":
        return fallback_tool._run(question)
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

