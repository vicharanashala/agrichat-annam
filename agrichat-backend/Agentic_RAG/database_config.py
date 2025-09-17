from pydantic import BaseModel
from typing import Optional, List

class DatabaseConfig(BaseModel):
    rag_enabled: bool = True  # Includes Golden + RAG databases
    pops_enabled: bool = True
    llm_enabled: bool = True
    
    def is_any_enabled(self) -> bool:
        return any([
            self.rag_enabled,
            self.pops_enabled,
            self.llm_enabled
        ])
    
    def is_traditional_mode(self) -> bool:
        # Traditional mode is when all are enabled (current default behavior)
        return self.rag_enabled and self.pops_enabled and self.llm_enabled
    
    def get_enabled_databases(self) -> List[str]:
        """Get list of enabled databases in processing order"""
        enabled = []
        if self.rag_enabled:
            enabled.append("rag")
        if self.pops_enabled:
            enabled.append("pops")
        if self.llm_enabled:
            enabled.append("llm")
        return enabled
    
    def get_database_selection(self) -> List[str]:
        """Get database selection for the query handler"""
        return self.get_enabled_databases()
