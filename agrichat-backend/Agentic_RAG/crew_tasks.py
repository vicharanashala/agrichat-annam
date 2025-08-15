from crewai import Task
from crew_agents import Retriever_Agent, Grader_agent, hallucination_grader, answer_grader
from tools import RAGTool, FallbackAgriTool, FireCrawlWebSearchTool

retriever_task = Task(
    description=(
        "For the question {question}, you MUST follow this exact sequence:\n"
        "1. FIRST: Always call the RAG tool to search the vectorstore database\n"
        "   - If conversation_history is provided: {conversation_history}, pass it to the RAG tool for context-aware responses\n"
        "   - If user_state is provided: {user_state}, pass it to the RAG tool for location-aware responses\n"
        "   - This enables chain of thought processing for follow-up queries and regional filtering\n"
        "2. ONLY IF the RAG tool returns '__FALLBACK__' or 'I don't have enough information': then call the fallback tool\n"
        "3. Return the answer with the correct source label\n"
        "CRITICAL: You must attempt the RAG tool first before any other tool. Do not skip this step.\n"
        "IMPORTANT: Always include conversation_history and user_state when calling the RAG tool, even if they are empty."
    ),
    expected_output=(
        "The answer from the appropriate tool:\n"
        "- RAG database if information is available\n"
        "- Fallback LLM if RAG returns '__FALLBACK__'\n"
        "Keep the response concise, structured, and user-friendly without technical source labels."
    ),
    agent=Retriever_Agent
)

grader_task = Task(
    description=(
        "Based on the response from the retriever task for the question {question}, evaluate whether the retrieved content is relevant to the question."
    ),
    expected_output=(
        "Binary score 'yes' or 'no' to indicate if the retrieved answer is relevant to the question. "
        "Answer 'yes' only if it meaningfully addresses the question. "
        "Answer 'no' if irrelevant or off-topic. "
        "Reply only 'yes' or 'no' without any explanation or preamble."
    ),
    agent=Grader_agent,
    context=[retriever_task],
)

hallucination_task = Task(
    description=(
        "Based on the graded response for the question {question}, assess if the answer is grounded in facts and supported by evidence."
    ),
    expected_output=(
        "Binary score 'yes' or 'no' to indicate if the answer is factually sound and aligned with the question. "
        "Answer 'yes' if the answer is useful and factually supported; 'no' otherwise. "
        "Reply only 'yes' or 'no' without any explanation."
    ),
    agent=hallucination_grader,
    context=[grader_task],
)

answer_task = Task(
    description=(
        "Based on the hallucination grading for the question {question}, decide whether to return the answer or perform a fallback web search."
        "If grading is 'yes', return a clear, concise answer. "
        "If grading is 'no', perform a web search (using the fallback tool) and return the retrieved information. "
        "If unable to produce a valid answer, respond with 'Sorry! unable to find a valid response'."
    ),
    expected_output=(
        "Return a clear and concise answer if hallucination task graded 'yes'. "
        "If 'no', invoke the fallback tool's web search and return the answer. "
        "Otherwise, respond with 'Sorry! unable to find a valid response'."
    ),
    agent=answer_grader,
    context=[hallucination_task],
)



