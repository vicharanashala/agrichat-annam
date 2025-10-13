from typing import Any, Callable, Dict, List, Optional

from .config import DEFAULT_CONFIG, PipelineConfig
from .runner import PipelineRunner
from .types import PipelineResult

_default_runner = PipelineRunner()


def configure_pipeline(config: PipelineConfig) -> None:
    global _default_runner
    _default_runner = PipelineRunner(config=config)


def run_pipeline(
    question: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    user_state: Optional[str] = None,
    *,
    stream: bool = False,
    token_callback: Optional[Callable[[str], None]] = None,
    intent_metadata: Optional[Dict[str, Optional[bool]]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> PipelineResult:
    return _default_runner.answer(
        question,
        conversation_history,
        user_state,
        stream=stream,
        token_callback=token_callback,
        intent_metadata=intent_metadata,
        config_overrides=config_overrides,
    )


def classify_question_intent(question: str) -> Dict[str, Optional[bool]]:
    """Expose intent classification metadata for external callers."""
    return _default_runner.classify_question_intent(question)
