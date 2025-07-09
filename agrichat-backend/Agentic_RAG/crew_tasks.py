from crewai import Task
from crew_agents import (
    Router_Agent, Retriever_Agent, Grader_agent,
    hallucination_grader, answer_grader, router_tool
)

router_task = Task(
    description=(
        "Analyse the keywords in the question {question}. "
        "Based on the keywords decide whether it is eligible for a vectorstore search or a web search. "
        "Return a single word 'vectorstore' if it is eligible for vectorstore search. "
        "Return a single word 'websearch' if it is eligible for web search. "
        "Do not provide any other preamble or explanation."
    ),
    expected_output=(
        "Give a binary choice 'websearch' or 'vectorstore' based on the question. "
        "Do not provide any other preamble or explanation."
    ),
    agent=Router_Agent,
    tools=[router_tool],
)

retriever_task = Task(
    description=(
        "Based on the response from the router task, extract information for the question {question} with the help of the respective tool. "
        "Use the web_search_tool to retrieve information from the web in case the router task output is 'websearch'. "
        "Use the rag_tool to retrieve information from the vectorstore in case the router task output is 'vectorstore'."
    ),
    expected_output=(
        "You should analyse the output of the 'router_task'. "
        "If the response is 'websearch' then use the web_search_tool to retrieve information from the web. "
        "If the response is 'vectorstore' then use the rag_tool to retrieve information from the vectorstore. "
        "Return a clear and concise text as response."
    ),
    agent=Retriever_Agent,
    context=[router_task],
)

grader_task = Task(
    description=(
        "Based on the response from the retriever task for the question {question}, evaluate whether the retrieved content is relevant to the question."
    ),
    expected_output=(
        "Binary score 'yes' or 'no' to indicate whether the document is relevant to the question. "
        "You must answer 'yes' if the response from the 'retriever_task' is in alignment with the question asked. "
        "You must answer 'no' if the response from the 'retriever_task' is not in alignment with the question asked. "
        "Do not provide any preamble or explanations except for 'yes' or 'no'."
    ),
    agent=Grader_agent,
    context=[retriever_task],
)

hallucination_task = Task(
    description=(
        "Based on the response from the grader task for the question {question}, evaluate whether the answer is grounded in/supported by a set of facts."
    ),
    expected_output=(
        "Binary score 'yes' or 'no' to indicate whether the answer is in sync with the question asked. "
        "Respond 'yes' if the answer is useful and contains facts about the question asked. "
        "Respond 'no' if the answer is not useful and does not contain facts about the question asked. "
        "Do not provide any preamble or explanations except for 'yes' or 'no'."
    ),
    agent=hallucination_grader,
    context=[grader_task],
)

answer_task = Task(
    description=(
        "Based on the response from the hallucination task for the question {question}, evaluate whether the answer is useful to resolve the question. "
        "If the answer is 'yes' return a clear and concise answer. "
        "If the answer is 'no' then perform a 'websearch' and return the response."
    ),
    expected_output=(
        "Return a clear and concise response if the response from 'hallucination_task' is 'yes'. "
        "Perform a web search using 'web_search_tool' and return a clear and concise response only if the response from 'hallucination_task' is 'no'. "
        "Otherwise respond as 'Sorry! unable to find a valid response'."
    ),
    context=[hallucination_task],
    agent=answer_grader,
)
