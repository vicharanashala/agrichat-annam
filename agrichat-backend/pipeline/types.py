from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrieverHit:
    source: str
    content: str
    metadata: Dict[str, Any]
    distance: Optional[float]
    cosine: Optional[float]
    state_used: Optional[str] = None


@dataclass
class PipelineResult:
    answer: str
    source: str
    similarity: Optional[float] = None
    distance: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)
    clarifying_questions: List[str] = field(default_factory=list)


@dataclass
class RetrievalDiagnostics:
    state_attempts: List[str] = field(default_factory=list)
    golden_hits: List[RetrieverHit] = field(default_factory=list)
    pops_hits: List[RetrieverHit] = field(default_factory=list)
