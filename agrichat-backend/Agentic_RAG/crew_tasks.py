from crewai import Task
from crew_agents import Retriever_Agent, Grader_agent, hallucination_grader, answer_grader
from tools import RAGTool, FallbackAgriTool, FireCrawlWebSearchTool

retriever_task = Task(
    description=(
        "For the Indian agricultural question {question}, you MUST follow this exact sequence for India-specific responses:\n"
        "1. FIRST: Always call the RAG tool to search the vectorstore database for Indian agricultural information\n"
        "   - If conversation_history is provided: {conversation_history}, pass it to the RAG tool for context-aware responses about Indian farming\n"
        "   - This enables chain of thought processing for follow-up queries about Indian agriculture\n"
        "   - Ensure all responses focus on Indian crop varieties, Indian soil conditions, and Indian farming practices\n"
        "2. ONLY IF the RAG tool returns '__FALLBACK__' or 'I don't have enough information': then call the fallback tool for Indian agricultural knowledge\n"
        "3. Return the answer with Indian agricultural context and the correct source label\n"
        "CRITICAL: You must attempt the RAG tool first before any other tool. Do not skip this step.\n"
        "IMPORTANT: Always include conversation_history when calling the RAG tool, even if it's an empty list.\n"
        "ESSENTIAL: All responses must be specific to Indian agriculture, Indian regions, and Indian farming conditions."
    ),
    expected_output=(
        "The answer from the appropriate tool focused on Indian agriculture:\n"
        "- RAG database if Indian agricultural information is available\n"
        "- Fallback LLM with Indian agricultural knowledge if RAG returns '__FALLBACK__'\n"
        "Keep the response concise, structured, user-friendly, and specifically applicable to Indian farming conditions without technical source labels.\n"
        "Ensure all advice is relevant to Indian soil types, climate patterns, crop varieties, and regional farming practices."
    ),
    agent=Retriever_Agent
)

grader_task = Task(
    description=(
        "Based on the response from the retriever task for the Indian agricultural question {question}, evaluate whether the retrieved content is relevant to the question and applicable to Indian farming conditions."
    ),
    expected_output=(
        "Binary score 'yes' or 'no' to indicate if the retrieved answer is relevant to the question and applicable to Indian agriculture. "
        "Answer 'yes' only if it meaningfully addresses the question with Indian agricultural context. "
        "Answer 'no' if irrelevant, off-topic, or not applicable to Indian farming conditions. "
        "Reply only 'yes' or 'no' without any explanation or preamble."
    ),
    agent=Grader_agent,
    context=[retriever_task],
)

hallucination_task = Task(
    description=(
        "Based on the graded response for the Indian agricultural question {question}, assess if the answer is grounded in Indian agricultural facts and supported by evidence applicable to Indian farming conditions."
    ),
    expected_output=(
        "Binary score 'yes' or 'no' to indicate if the answer is factually sound for Indian agriculture and aligned with the question. "
        "Answer 'yes' if the answer is useful, factually supported, and applicable to Indian farming conditions; 'no' otherwise. "
        "Reply only 'yes' or 'no' without any explanation."
    ),
    agent=hallucination_grader,
    context=[grader_task],
)

answer_task = Task(
    description=(
        "Based on the hallucination grading for the Indian agricultural question {question}, decide whether to return the answer or perform a fallback search for Indian agricultural information. "
        "If grading is 'yes', return a clear, concise answer focused on Indian agricultural context. "
        "If grading is 'no', perform a search using the fallback tool for Indian agricultural knowledge and return India-specific information. "
        "If unable to produce a valid answer about Indian agriculture, respond with 'Sorry! unable to find a valid response for Indian agricultural conditions'."
    ),
    expected_output=(
        "Return a clear and concise answer focused on Indian agriculture if hallucination task graded 'yes'. "
        "If 'no', invoke the fallback tool's search for Indian agricultural information and return the India-specific answer. "
        "All responses must be applicable to Indian farming conditions, Indian crop varieties, and Indian regional agriculture. "
        "Otherwise, respond with 'Sorry! unable to find a valid response for Indian agricultural conditions'."
    ),
    agent=answer_grader,
    context=[hallucination_task],
)



