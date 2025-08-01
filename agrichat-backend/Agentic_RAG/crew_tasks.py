from crewai import Task
from crew_agents import Retriever_Agent, Grader_agent, hallucination_grader, answer_grader
from tools import RAGTool, FallbackAgriTool, FireCrawlWebSearchTool

# Task: Retrieve answer from vectorstore (RAG tool) or fallback LLM+WebSearch
retriever_task = Task(
    description=(
        "For the question {question}, always attempt to answer using the RAG tool (vectorstore) first. "
        "If the RAG tool does not provide a relevant or confident answer (signals fallback), then use the fallback tool (LLM + web search) to find the answer. "
        "Present the final answer clearly and specify the source."
    ),
    expected_output=(
        "Present the final answer in a clear and structured format, clearly indicating the source (e.g., 'Source: RAG Database' or 'Source: LLM knowledge & Web Search')."
    ),
    agent=Retriever_Agent
)

# Task: Grade whether the document retrieved is relevant to the question
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

# Task: Grade whether the answer is grounded, factual, and not hallucinated
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

# Task: Final answer generation or fallback web search based on hallucination grading
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



