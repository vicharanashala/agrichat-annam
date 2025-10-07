"""
AgriChat Agentic RAG System - Tasks Package

This package contains task definitions for agricultural knowledge processing workflow.
"""

from .crew_tasks import AgriTasksManager, create_legacy_tasks

__all__ = [
    'AgriTasksManager',
    'create_legacy_tasks'
]
