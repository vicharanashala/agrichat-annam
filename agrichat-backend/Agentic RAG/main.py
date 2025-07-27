from crewai import Crew
from crew_agents import (
    Retriever_Agent, Grader_agent,
    hallucination_grader, answer_grader
)
from crew_tasks import (
    retriever_task, grader_task,
    hallucination_task, answer_task
)

def get_answer(question):
    rag_crew = Crew(
        agents=[
            Retriever_Agent
        ],
        tasks=[
            retriever_task
        ],
        verbose=True,
    )
    inputs = {"question": question}
    result = rag_crew.kickoff(inputs=inputs)
    return result

if __name__ == "__main__":
    while True:
        question = input("Ask your question or type 'exit' to quit: ")
        if question.strip().lower() == "exit":
            break
        answer = get_answer(question)
        print(answer)
