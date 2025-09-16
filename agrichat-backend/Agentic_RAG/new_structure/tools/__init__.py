"""
Tools package initialization for AgriChat Agentic RAG system.
"""

from .rag_tool import AgriRAGTool, RAGToolSchema
from .fallback_tool import AgriFallbackTool, FallbackToolSchema

# Legacy imports for backward compatibility
RAGTool = AgriRAGTool
FallbackAgriTool = AgriFallbackTool

__all__ = [
    # New OOP classes
    "AgriRAGTool",
    "AgriFallbackTool", 
    "RAGToolSchema",
    "FallbackToolSchema",
    
    # Legacy compatibility
    "RAGTool",
    "FallbackAgriTool"
]
