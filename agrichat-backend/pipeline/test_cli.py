#!/usr/bin/env python3
"""Ad-hoc CLI harness for exercising the new pipeline module.

Example usages:
    python3 -m pipeline.test_cli -q "What is gladiolus?" --state Punjab
    python3 -m pipeline.test_cli --no-golden --interactive
    python3 -m pipeline.test_cli --questions-file sample_questions.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import configure_pipeline, run_pipeline
from .config import DEFAULT_CONFIG, PipelineConfig
from .types import PipelineResult


def _load_questions_from_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")
    questions: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            questions.append(line)
    return questions


def _summarize_result(index: int, question: str, result: PipelineResult, *, show_answer: bool = True) -> None:
    header = f"\n==== Q{index} :: {question} ===="
    print(header)
    print(f"Source: {result.source}")
    if result.similarity is not None:
        print(f"Similarity: {result.similarity:.3f}")
    if result.distance is not None:
        print(f"Distance: {result.distance:.3f}")
    if result.clarifying_questions:
        print("Clarifications suggested:")
        for follow_up in result.clarifying_questions:
            print(f"  - {follow_up}")
    if show_answer:
        print("Answer:\n")
        print(result.answer)
    else:
        print("Answer: [streamed above]\n")
    if result.reasoning:
        print("\nReasoning trace:")
        for step in result.reasoning:
            print(f"  • {step}")
    diagnostics = (result.metadata or {}).get("diagnostics")
    if diagnostics:
        print("\nDiagnostics:")
        print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


class SessionRunner:
    """Maintains a conversation history while running multiple questions."""

    def __init__(self, config: PipelineConfig, state: Optional[str], *, stream: bool):
        self.history: List[Dict[str, str]] = []
        self.state = state
        self.stream = stream
        configure_pipeline(config)
        self._index = 1

    def ask(self, question: str, *, stream: Optional[bool] = None) -> PipelineResult:
        use_stream = self.stream if stream is None else stream

        tokens_emitted = False
        answer_streamed = False

        def _stream_callback(chunk: str) -> None:
            nonlocal tokens_emitted, answer_streamed
            if not tokens_emitted:
                print("\n-- Streaming response --")
                tokens_emitted = True
            if chunk.startswith("[thinking]"):
                print(chunk)
            else:
                answer_streamed = True
                print(chunk, end="", flush=True)

        callback = _stream_callback if use_stream else None

        result = run_pipeline(
            question,
            conversation_history=self.history,
            user_state=self.state,
            stream=use_stream,
            token_callback=callback,
        )

        if tokens_emitted:
            print("\n-- End of stream --\n")

        self.history.append({
            "question": question,
            "answer": result.answer,
        })
        _summarize_result(self._index, question, result, show_answer=not answer_streamed)
        self._index += 1
        return result


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    config = DEFAULT_CONFIG
    config = replace(
        config,
        enable_golden=not args.no_golden,
        enable_pops=not args.no_pops,
        enable_llm=not args.no_llm,
        show_diagnostics=not args.no_diagnostics,
    )
    if args.clarify is not None:
        config = replace(config, clarify_with_llm=args.clarify)
    return config


def _iter_questions(args: argparse.Namespace) -> Iterable[str]:
    questions: List[str] = []
    if args.questions_file:
        questions.extend(_load_questions_from_file(Path(args.questions_file)))
    if args.question:
        questions.extend(args.question)
    return questions


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Exercise the lightweight pipeline with toggles.")
    parser.add_argument("-q", "--question", action="append", help="Question to run (can be repeated)")
    parser.add_argument("--questions-file", help="Path to a text file with one question per line")
    parser.add_argument("--state", help="User state to prioritize (e.g. Punjab)")
    parser.add_argument("--no-golden", action="store_true", help="Disable Golden database retrieval")
    parser.add_argument("--no-pops", action="store_true", help="Disable PoPs retrieval")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM fallback")
    parser.add_argument("--no-diagnostics", action="store_true", help="Hide diagnostics payload in output")
    parser.add_argument("--clarify", dest="clarify", action="store_true", help="Enable LLM clarification suggestions")
    parser.add_argument("--no-clarify", dest="clarify", action="store_false", help="Disable LLM clarification suggestions")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive REPL after scripted questions")
    parser.add_argument("--no-stream", action="store_true", help="Disable LLM streaming output")
    parser.set_defaults(clarify=None)

    args = parser.parse_args(argv)

    config = _build_config(args)
    runner = SessionRunner(config, state=args.state, stream=not args.no_stream)

    scripted_questions = list(_iter_questions(args))
    if scripted_questions:
        for question in scripted_questions:
            runner.ask(question)

    if args.interactive or not scripted_questions:
        print("\nEntering interactive mode (type '/quit' to exit).\n")
        try:
            while True:
                raw = input("Question> ").strip()
                if not raw:
                    continue
                if raw.lower() in {"exit", "quit", "/quit"}:
                    break
                runner.ask(raw)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
