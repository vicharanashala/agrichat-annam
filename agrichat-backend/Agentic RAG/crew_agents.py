from crewai import Agent
from tools import FireCrawlWebSearchTool, RAGTool



firecrawl_tool = FireCrawlWebSearchTool(api_key="fc-3042e1475cda4e51b0ce4fdd6ea58578")
rag_tool = RAGTool(chroma_path=r"C:\Users\amank\Gemini_based_processing\chromaDb", gemini_api_key="AIzaSyCzS2rkrIU-qed90akvU4sjT43W8UANA5A")

from crewai import LLM
llm = LLM(
    model='gemini/gemini-2.5-flash',
    api_key='AIzaSyCzS2rkrIU-qed90akvU4sjT43W8UANA5A',
    temperature=0.0
)

Retriever_Agent = Agent(
    role="Retriever",
    goal="Always attempt to answer the user query using the RAG tool (vectorstore) first. "
        "If the RAG tool does not provide a relevant or confident answer, "
        "then use the web search tool to find the answer. "
        "Present the final answer in a clear and structured format, indicating the source.",
    backstory=(
        "You are a retrieval specialist who prioritizes trusted, internal knowledge. "
        "Your primary responsibility is to answer questions using the internal vectorstore (RAG tool). "
        "If the knowledge base cannot answer, you seamlessly fall back to web search. "
        "You always make sure the user receives the most relevant and up-to-date information, "
        "clearly indicating whether the answer comes from the knowledge base or the web."
    ),
    verbose=True,
    llm=llm,
    tools=[rag_tool, firecrawl_tool],
)

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
        "If the answer generated is not relevant then perform a websearch using 'FireCrawlWebSearchTool'."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[firecrawl_tool],
)


