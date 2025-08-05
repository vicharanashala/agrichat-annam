from crewai import Crew
from crew_agents import (
    Retriever_Agent, Grader_agent,
    hallucination_grader, answer_grader
)
from crew_tasks import (
    retriever_task, grader_task,
    hallucination_task, answer_task
)

conversation_history = []

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
    
    inputs = {
        "question": question,
        "conversation_history": conversation_history
    }
    result = rag_crew.kickoff(inputs=inputs)
    
    conversation_history.append({
        "question": question,
        "answer": str(result)
    })
    
    if len(conversation_history) > 5:
        conversation_history.pop(0)
    
    return result

if __name__ == "__main__":
    print("AgriChat Local Test - Chain of Thought Enabled")
    print("=" * 50)
    
    while True:
        question = input("Ask your question or type 'exit' to quit: ")
        if question.strip().lower() == "exit":
            break
        answer = get_answer(question)
        print(answer)
