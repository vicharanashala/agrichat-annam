"""
Module: main
Purpose: Orchestrates the full pipeline: data ingestion, retrieval, and agentic answer generation.
"""

from data_ingestion import chroma_db_builder
from retrieval import chroma_retriever
from agents import agentic_rag_agent

def main():
    """
    Runs the full pipeline in sequence: builds DB, tests retrieval, and runs the agent.
    """
    chroma_db_builder.main()
    chroma_retriever.main()
    agentic_rag_agent.main()

if __name__ == "__main__":
    main()
