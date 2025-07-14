from crewai import Task, Agent, Crew, Process
from crew_agents import (
    Retriever_Agent, Grader_agent,
    hallucination_grader, answer_grader
)
from tools import FireCrawlWebSearchTool, RAGTool
firecrawl_tool = FireCrawlWebSearchTool(api_key="fc-3042e1475cda4e51b0ce4fdd6ea58578")
rag_tool = RAGTool(chroma_path=r"C:\Users\amank\Gemini_based_processing\chromaDb", gemini_api_key="AIzaSyCzS2rkrIU-qed90akvU4sjT43W8UANA5A")

# router_task = Task(
#     description=(
#         "Analyse the keywords in the question {question}. "
#         "Based on the keywords decide whether it is eligible for a vectorstore search or a web search. "
#         "Return a single word 'vectorstore' if it is eligible for vectorstore search. "
#         "Return a single word 'websearch' if it is eligible for web search. "
#         "Do not provide any other preamble or explanation."
#     ),
#     expected_output=(
#         "Give a binary choice 'websearch' or 'vectorstore' based on the question. "
#         "Do not provide any other preamble or explanation."
#     ),
#     agent=Router_Agent
# )

retriever_task = Task(
    description=(
        "For the question {question}, always attempt to answer using the RAG tool (vectorstore) only."
        "Present the final answer in a clear and structured format, clearly indicating the source."
    ),
    expected_output=(
        "A clear and concise answer to the question."
    ),
    agent=Retriever_Agent
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


crew = Crew(
			tasks=[retriever_task],
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)

result = crew.kickoff(inputs={"question": "What is Potato Blight and how can it be controlled?"})