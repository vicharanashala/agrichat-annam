from pydantic import BaseModel
from typing import Optional

class DatabaseConfig(BaseModel):
    golden_db_enabled: bool = False
    rag_db_enabled: bool = False
    pops_db_enabled: bool = False
    llm_fallback_enabled: bool = False
    
    def is_any_enabled(self) -> bool:
        return any([
            self.golden_db_enabled,
            self.rag_db_enabled, 
            self.pops_db_enabled,
            self.llm_fallback_enabled
        ])
    
    def is_traditional_mode(self) -> bool:
        return not self.is_any_enabled()
    
    def get_enabled_databases(self) -> list:
        enabled = []
        if self.golden_db_enabled:
            enabled.append("golden")
        if self.rag_db_enabled:
            enabled.append("rag")
        if self.pops_db_enabled:
            enabled.append("pops")
        if self.llm_fallback_enabled:
            enabled.append("llm")
        return enabled
