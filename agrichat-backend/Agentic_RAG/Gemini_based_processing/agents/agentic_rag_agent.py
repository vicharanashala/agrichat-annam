"""
Module: agents.agentic_rag_agent
Purpose: Implements the agent logic using LangChain's agent framework and the retriever tool.
"""

from langchain.agents import initialize_agent, AgentType
import google.generativeai as genai
from retrieval.chroma_retriever import ChromaRetriever, get_retriever_tool
from prompts.prompt_templates import AGENT_PROMPT

class AgenticRAGAgent:
    """
    Orchestrates retrieval and answer generation using an agentic approach.
    """
    def __init__(self, chroma_retriever, gemini_api_key, chat_model="gemma-3-27b-it"):
        self.retriever_tool = get_retriever_tool(chroma_retriever)
        genai.configure(api_key=gemini_api_key)
        self.llm = genai.GenerativeModel(chat_model)
        self.agent = initialize_agent(
            tools=[self.retriever_tool],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def answer(self, question):
        """
        Uses the agent to answer a user question with retrieval-augmented generation.
        """
        return self.agent.run(question)

def main():
    """
    Main entry point for running the agentic RAG pipeline.
    """
    chroma_path = r"C:\Users\amank\agrichat-annam\Agentic_RAG\Gemini_based_processing\chromaDb"
    gemini_api_key = "AIzaSyDZ2ZOEd9bIwOAHmk4wjVuKrpAP4x56EPI"
    chroma_retriever = ChromaRetriever(chroma_path, api_key=gemini_api_key)
    agent = AgenticRAGAgent(chroma_retriever, gemini_api_key)
    question = "Give information regarding wheat cultivation, any disease and their cure all information related?"
    answer = agent.answer(question)
    print(f"Answer: {answer}")
