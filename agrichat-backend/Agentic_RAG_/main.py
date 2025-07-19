# from crewai import Crew
# from .crew_agents import (
#     Retriever_Agent, Grader_agent,
#     hallucination_grader, answer_grader
# )
# from .crew_tasks import (
#     retriever_task, grader_task,
#     hallucination_task, answer_task
# )

# def get_answer(question):
#     print("here")
#     rag_crew = Crew(
#         agents=[
#             Retriever_Agent
#         ],
#         tasks=[
#             retriever_task
#         ],
#         verbose=True,
#     )
#     inputs = {"question": question}
#     result = rag_crew.kickoff(inputs=inputs)
#     print(question)
#     print("Result object from crew.kickoff():", result)

#     if hasattr(result, "output"):
#         return result.output
#     elif hasattr(result, "text"):
#         return result.text
#     elif isinstance(result, str):
#         return result
#     else:
#         return str(result) 

# if __name__ == "__main__":
#     while True:
#         question = input("Ask your agricultural question or type 'exit' to quit: ")
#         if question.strip().lower() == "exit":
#             break
#         answer = get_answer(question)


from crewai import Crew
from .crew_agents import (
    Retriever_Agent, Grader_agent,
    hallucination_grader, answer_grader
)

from .chroma_query_handler import ChromaQueryHandler
from .tools import inject_chroma_handler, get_rag_tool
from .crew_agents import set_tools
from .crew_tasks import get_tasks
from crewai import Crew

def initialize_handler():
    chroma_handler = ChromaQueryHandler(
        chroma_path="Agentic_RAG_/chromaDb",
        gemini_api_key="AIzaSyCzS2rkrIU-qed90akvU4sjT43W8UANA5A"
    )
    inject_chroma_handler(chroma_handler)

    
    rag_tool = get_rag_tool()
    set_tools(rag_tool)

    
    retriever_task, grader_task, hallucination_task, answer_task = get_tasks()

    def get_answer(question: str):
        rag_crew = Crew(
            agents=[Retriever_Agent],
            tasks=[retriever_task],
            verbose=True,
        )
        inputs = {"question": question}
        result = rag_crew.kickoff(inputs=inputs)

        if hasattr(result, "output"):
            return result.output
        elif hasattr(result, "text"):
            return result.text
        elif isinstance(result, str):
            return result
        else:
            return str(result)

    return get_answer
