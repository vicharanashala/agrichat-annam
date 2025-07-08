from crewai import Agent
import google.generativeai as genai
from tools import firecrawl_tool_dict, rag_tool_dict


def router_tool(question):
    if 'ESOP' in question:
        return 'vectorstore'
    else:
        return 'web_search'

def create_llm_instance(api_key: str, model_name: str = "gemma-3-27b-it"):
    genai.configure(api_key=api_key)
    llm_instance = genai.GenerativeModel(model_name)
    return llm_instance

api_key = "AIzaSyDZ2ZOEd9bIwOAHmk4wjVuKrpAP4x56EPI"
llm = create_llm_instance(api_key, "gemma-3-27b-it")

Router_Agent = Agent(
    role='Router',
    goal='Route user question to a vectorstore or web search',
    backstory=(
        "You are an expert at routing a user question to a vectorstore or web search."
        "Use the vectorstore for questions on concepts related to Retrieval-Augmented Generation."
        "You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search."
    ),
    verbose=True,
    llm=llm,
)

Retriever_Agent = Agent(
    role="Retrieve relevant information to answer the user query: {query}"  ,
    goal=(  "Retrieve the most relevant information from the available sources "
            "for the user query: {query}. Always try to use the PDF search tool first. "
            "If you are not able to retrieve the information from the PDF search tool, "
            "then try to use the web search tool."),
    backstory=(
        """You're a meticulous analyst with a keen eye for detail. 
                You're known for your ability understand the user query: {query} 
                and retrieve knowledge from the most suitable knowledge base."""
    ),
    verbose=True,
    llm=llm,
    tools=[firecrawl_tool_dict, rag_tool_dict],
)

# Grader_agent = Agent(
#     role='Answer Grader',
#     goal='Filter out erroneous retrievals',
#     backstory=(
#         "You are a grader assessing relevance of a retrieved document to a user question."
#         "If the document contains keywords related to the user question, grade it as relevant."
#         "It does not need to be a stringent test. You have to make sure that the answer is relevant to the question."
#     ),
#     verbose=True,
#     allow_delegation=False,
#     llm=llm,
# )

# hallucination_grader = Agent(
#     role="Hallucination Grader",
#     goal="Filter out hallucination",
#     backstory=(
#         "You are a hallucination grader assessing whether an answer is grounded in / supported by a set of facts."
#         "Make sure you meticulously review the answer and check if the response provided is in alignment with the question asked"
#     ),
#     verbose=True,
#     allow_delegation=False,
#     llm=llm,
# )

# answer_grader = Agent(
#     role="Answer Grader",
#     goal="Filter out hallucination from the answer.",
#     backstory=(
#         "You are a grader assessing whether an answer is useful to resolve a question."
#         "Make sure you meticulously review the answer and check if it makes sense for the question asked."
#         "If the answer is relevant generate a clear and concise response."
#         "If the answer generated is not relevant then perform a websearch using 'FireCrawlWebSearchTool'."
#     ),
#     verbose=True,
#     allow_delegation=False,
#     llm=llm,
#     tools=[firecrawl_tool_dict],
# )


sample_question = "What are the best irrigation practices for brinjal?"
response = Retriever_Agent.tools[0]['function'](sample_question)
print("Sample Tool Response:", response)
