"""
AgriChat Agentic RAG System - Agents Package

This package contains AI agents for agricultural knowledge processing and response generation.
"""

from .crew_agents import AgriAgentsManager, get_agents_manager

# Legacy imports for backward compatibility
from .crew_agents import (
    Retriever_Agent,
    Grader_agent, 
    hallucination_grader,
    answer_grader,
    retriever_response
)

__all__ = [
    'AgriAgentsManager',
    'get_agents_manager',
    # Legacy exports
    'Retriever_Agent',
    'Grader_agent',
    'hallucination_grader', 
    'answer_grader',
    'retriever_response'
]
