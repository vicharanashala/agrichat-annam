from crewai import Crew
from crew_agents import (
    Router_Agent, Retriever_Agent, Grader_agent,
    hallucination_grader, answer_grader
)
from crew_tasks import (
    router_task, retriever_task, grader_task,
    hallucination_task, answer_task
)

def run_agentic_rag(question):
    rag_crew = Crew(
        agents=[
            Router_Agent, Retriever_Agent, Grader_agent,
            hallucination_grader, answer_grader
        ],
        tasks=[
            router_task, retriever_task, grader_task,
            hallucination_task, answer_task
        ],
        verbose=True,
    )
    inputs = {"question": question}
    result = rag_crew.kickoff(inputs=inputs)
    return result

if __name__ == "__main__":
    while True:
        question = input("Ask your agricultural question or type 'exit' to quit: ")
        if question.strip().lower() == "exit":
            break
        answer = run_agentic_rag(question)
        print(answer)
