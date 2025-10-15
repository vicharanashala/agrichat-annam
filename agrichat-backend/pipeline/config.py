from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SourceThresholds:
    """Similarity thresholds for a knowledge source."""

    max_distance: Optional[float] = None
    min_cosine: float = 0.35

    def passes(self, distance: Optional[float], cosine: Optional[float]) -> bool:
        if cosine is None:
            return False
        if cosine < self.min_cosine:
            return False
        if self.max_distance is None or distance is None:
            return True
        return distance <= self.max_distance


@dataclass
class PipelineConfig:
    """Runtime configuration for the lightweight pipeline."""

    enable_golden: bool = True
    enable_pops: bool = True
    enable_llm: bool = True
    golden_thresholds: SourceThresholds = field(
        default_factory=lambda: SourceThresholds(max_distance=0.45, min_cosine=0.5)
    )
    pops_thresholds: SourceThresholds = field(
        default_factory=lambda: SourceThresholds(max_distance=None, min_cosine=0.65)
    )
    pops_dynamic_distance_multiplier: float = 5.0
    golden_k: int = 5
    pops_k: int = 8
    aggregation_k: int = 12
    use_conversation_history: bool = True
    clarify_with_llm: bool = True
    clarification_threshold: float = 0.45
    clarification_max_questions: int = 2
    llm_model: str = "gpt-oss:latest"
    clarification_temperature: float = 0.2
    answer_temperature: float = 0.2
    max_answer_tokens: int = 1024
    enable_logging: bool = True
    show_diagnostics: bool = True
    use_llm_intent_classifier: bool = True


DEFAULT_CONFIG = PipelineConfig()
