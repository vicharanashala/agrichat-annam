import csv
import logging
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import DEFAULT_CONFIG, PipelineConfig
from .intent_dictionary import AGRICULTURE_KEYWORDS
from .llm import GENERAL_REFUSAL, LLMResponder
from .state_utils import prioritize_states
from .types import PipelineResult, RetrievalDiagnostics, RetrieverHit
from .vectorstores import VectorStores
from .retrievers import GoldenRetriever, PopsRetriever

logger = logging.getLogger(__name__)

def _is_agricultural_question(question: str) -> bool:
    if not question:
        return False
    q = question.lower()
    if any(term in q for term in AGRICULTURE_KEYWORDS):
        return True

    patterns = [
        r"\bvarieties? of [a-z]+",
        r"\bhow to (grow|cultivate)",
        r"\bspacing for [a-z]+",
        r"\bseed rate",
        r"\bfertilizer (schedule|recommendation)",
        r"\bpest management",
        r"\bdisease control",
        r"\bpackage of practices",
        r"\b(kg|tonnes?|tons)\s+per\s+(acre|hectare)",
        r"\bnutrient (management|plan)",
        r"\bwhat can i grow",
    ]
    return any(re.search(pattern, q) for pattern in patterns)


class PipelineRunner:
    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.config = config
        self.stores = VectorStores()
        self.golden = GoldenRetriever(self.stores.golden, config)
        self.pops = PopsRetriever(self.stores.pops, config)
        self.llm = LLMResponder(config)
        self._fallback_log_path = Path(__file__).resolve().parents[2] / "fallback_queries.csv"

    @staticmethod
    def _clamp_threshold(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _apply_config_overrides(self, config: PipelineConfig, overrides: Dict[str, Any]) -> PipelineConfig:
        if "enable_golden" in overrides:
            config.enable_golden = bool(overrides["enable_golden"])
        if "enable_pops" in overrides:
            config.enable_pops = bool(overrides["enable_pops"])
        if "enable_llm" in overrides:
            config.enable_llm = bool(overrides["enable_llm"])

        if "golden_min_cosine" in overrides:
            try:
                config.golden_thresholds.min_cosine = self._clamp_threshold(float(overrides["golden_min_cosine"]))
            except (TypeError, ValueError):
                logger.warning("Invalid golden_min_cosine override: %s", overrides["golden_min_cosine"])
        if "pops_min_cosine" in overrides:
            try:
                config.pops_thresholds.min_cosine = self._clamp_threshold(float(overrides["pops_min_cosine"]))
            except (TypeError, ValueError):
                logger.warning("Invalid pops_min_cosine override: %s", overrides["pops_min_cosine"])

        if not overrides.get("adaptive_thresholds", True):
            config.pops_dynamic_distance_multiplier = min(config.pops_dynamic_distance_multiplier, 1.0)

        if overrides.get("strict_validation"):
            config.golden_thresholds.min_cosine = max(config.golden_thresholds.min_cosine, 0.6)
            config.pops_thresholds.min_cosine = max(config.pops_thresholds.min_cosine, 0.45)
            if config.golden_thresholds.max_distance is not None:
                config.golden_thresholds.max_distance = min(config.golden_thresholds.max_distance, 0.35)

        return config

    _STOPWORDS: set[str] = {
        "what",
        "when",
        "where",
        "which",
        "that",
        "this",
        "with",
        "from",
        "have",
        "each",
        "into",
        "your",
        "about",
        "will",
        "more",
        "than",
        "much",
        "many",
        "take",
        "giving",
        "give",
        "makes",
        "make",
        "need",
        "needs",
        "should",
        "could",
        "would",
        "please",
        "kindly",
        "some",
        "also",
        "per",
        "acre",
        "hectare",
        "apply",
        "applied",
        "applying",
        "use",
        "using",
        "for",
        "help",
        "want",
        "know",
    }

    @staticmethod
    def _trim_content(text: str, limit: int = 800) -> str:
        collapsed = " ".join(text.split())
        if len(collapsed) <= limit:
            return collapsed
        return collapsed[: limit - 1].rstrip() + "…"

    def _format_hits_for_context(
        self,
        label: str,
        hits: List[RetrieverHit],
        max_items: int = 2,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if not hits:
            return "", []

        selected = hits[:max_items]
        lines = []
        meta: List[Dict[str, Any]] = []
        for idx, hit in enumerate(selected, start=1):
            score = hit.cosine if hit.cosine is not None else None
            meta.append(
                {
                    "state": hit.state_used,
                    "cosine": score,
                    "distance": hit.distance,
                }
            )
            snippet = self._trim_content(hit.content)
            score_display = f"{score:.3f}" if score is not None else "n/a"
            lines.append(
                f"[{label} #{idx} | state={hit.state_used or 'unknown'} | cosine={score_display}]"
                f"\n{snippet}"
            )
        heading = f"{label} context:"
        return heading + "\n" + "\n\n".join(lines), meta

    def _build_llm_context(
        self,
        golden_hits: List[RetrieverHit],
        pops_hits: List[RetrieverHit],
        clarifying_questions: List[str],
    ) -> Tuple[str, Dict[str, List[Dict[str, Optional[float]]]]]:
        sections: List[str] = []
        meta: Dict[str, List[Dict[str, Optional[float]]]] = {}

        golden_section, golden_meta = self._format_hits_for_context("Golden", golden_hits)
        if golden_section:
            sections.append(golden_section)
            meta["golden"] = golden_meta

        pops_section, pops_meta = self._format_hits_for_context("PoPs", pops_hits, max_items=3)
        if pops_section:
            sections.append(pops_section)
            meta["pops"] = pops_meta

        if clarifying_questions:
            sections.append(
                "Clarifying prompts:\n" + "\n".join(f"- {question}" for question in clarifying_questions)
            )

        context = "\n\n".join(sections).strip()
        return context, meta

    def _log_fallback(self, question: str, answer: str, reason: str) -> None:
        if not self.config.enable_logging:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        path = self._fallback_log_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = path.exists()
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if not file_exists:
                    writer.writerow(["timestamp", "question", "answer", "fallback_reason"])
                writer.writerow([timestamp, question, answer, reason])
        except PermissionError:
            logger.warning("Disabling fallback logging because %s is not writable", path)
            self.config.enable_logging = False
        except Exception:  # pragma: no cover - logging failure should not break pipeline
            logger.exception("Failed to log fallback query")

    def _evaluate_hits(self, hits, thresholds, dynamic_multiplier=1.0):
        for hit in hits:
            if hit.cosine is None:
                continue
            distance_ok = True
            if thresholds.max_distance is not None and hit.distance is not None:
                dynamic_limit = thresholds.max_distance * dynamic_multiplier
                distance_ok = hit.distance <= dynamic_limit
            if distance_ok and hit.cosine >= thresholds.min_cosine:
                return hit
        return None
    @staticmethod
    def _extract_keywords(question: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z]{4,}", question.lower())
        keywords: List[str] = []
        for token in tokens:
            if token in PipelineRunner._STOPWORDS:
                continue
            keywords.append(token)
        return keywords

    @staticmethod
    def _keyword_variants(keyword: str) -> List[str]:
        variants = {keyword}
        if keyword.endswith("ies") and len(keyword) > 3:
            variants.add(keyword[:-3] + "y")
        if keyword.endswith("es") and len(keyword) > 2:
            variants.add(keyword[:-2])
        if keyword.endswith("s") and len(keyword) > 1:
            variants.add(keyword[:-1])
        if keyword.endswith("ing") and len(keyword) > 3:
            variants.add(keyword[:-3])
        return list(variants)

    @classmethod
    def _keyword_in_text(cls, keyword: str, text: str) -> bool:
        for variant in cls._keyword_variants(keyword):
            if not variant:
                continue
            if re.search(rf"\b{re.escape(variant)}\b", text):
                return True
        return False

    @classmethod
    def _hit_has_keyword_overlap(cls, keywords: List[str], content: str) -> bool:
        if not keywords:
            return True
        text = content.lower()
        unique_keywords: List[str] = []
        seen: set[str] = set()
        for keyword in keywords:
            if keyword in seen:
                continue
            seen.add(keyword)
            unique_keywords.append(keyword)
        if not unique_keywords:
            return True
        matches = sum(1 for keyword in unique_keywords if cls._keyword_in_text(keyword, text))
        total = len(unique_keywords)
        if total == 1:
            required = 1
        elif total == 2:
            required = 2
        elif total <= 5:
            required = 2
        else:
            required = 3
        required = min(required, total)
        return matches >= required

    @staticmethod
    def _hit_passes_threshold(hit: RetrieverHit, thresholds, dynamic_multiplier: float = 1.0) -> bool:
        if hit.cosine is None:
            return False
        if hit.cosine < thresholds.min_cosine:
            return False
        if thresholds.max_distance is None or hit.distance is None:
            return True
        return hit.distance <= thresholds.max_distance * dynamic_multiplier

    def _evaluate_hits(
        self,
        hits: List[RetrieverHit],
        thresholds,
        keywords: List[str],
        *,
        dynamic_multiplier: float = 1.0,
    ) -> Tuple[Optional[RetrieverHit], bool]:
        filtered_for_context = False
        for hit in hits:
            if not self._hit_passes_threshold(hit, thresholds, dynamic_multiplier):
                continue
            if not self._hit_has_keyword_overlap(keywords, hit.content):
                filtered_for_context = True
                continue
            return hit, filtered_for_context
        return None, filtered_for_context

    def classify_question_intent(self, question: str) -> Dict[str, Optional[bool]]:
        """Determine whether a question is agricultural using dictionary + LLM."""
        heuristic_intent = _is_agricultural_question(question)
        intent_metadata: Dict[str, Optional[bool]] = {
            "heuristic": heuristic_intent,
            "llm_used": False,
            "llm_result": None,
        }

        intent_allowed = heuristic_intent
        if not heuristic_intent and self.config.use_llm_intent_classifier and self.config.enable_llm:
            intent_metadata["llm_used"] = True
            llm_result = self.llm.classify_question_intent(question)
            intent_metadata["llm_result"] = llm_result
            if llm_result is not None:
                intent_allowed = llm_result

        intent_metadata["final"] = bool(intent_allowed)
        return intent_metadata

    def answer(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_state: Optional[str] = None,
        *,
        stream: bool = False,
        token_callback: Optional[Callable[[str], None]] = None,
        intent_metadata: Optional[Dict[str, Optional[bool]]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        diagnostics = RetrievalDiagnostics()
        config = deepcopy(self.config)
        overrides_payload = config_overrides or {}
        if overrides_payload:
            config = self._apply_config_overrides(config, overrides_payload)
        raw_config_metadata = overrides_payload.get("raw_database_config") if overrides_payload else None

        golden_retriever = self.golden if not overrides_payload else GoldenRetriever(self.stores.golden, config)
        pops_retriever = self.pops if not overrides_payload else PopsRetriever(self.stores.pops, config)
        llm_responder = self.llm if not overrides_payload else LLMResponder(config)

        if intent_metadata is None:
            heuristic_intent = _is_agricultural_question(question)
            intent_metadata = {
                "heuristic": heuristic_intent,
                "llm_used": False,
                "llm_result": None,
            }
            intent_allowed = heuristic_intent
            if not heuristic_intent and config.use_llm_intent_classifier and config.enable_llm:
                intent_metadata["llm_used"] = True
                llm_result = llm_responder.classify_question_intent(question)
                intent_metadata["llm_result"] = llm_result
                if llm_result is not None:
                    intent_allowed = llm_result
            intent_metadata["final"] = bool(intent_allowed)
        else:
            intent_allowed = bool(intent_metadata.get("final"))

        if not intent_allowed:
            logger.debug("Question classified as non-agricultural", extra={"intent_metadata": intent_metadata})
            return PipelineResult(
                answer=GENERAL_REFUSAL,
                source="Policy",
                reasoning=["Non-agricultural intent detected"],
                metadata={
                    "intent_classification": intent_metadata,
                    **({"database_config": raw_config_metadata} if raw_config_metadata else {}),
                },
            )

        states = prioritize_states(question, user_state)
        diagnostics.state_attempts = states
        keywords = self._extract_keywords(question)

        golden_hits: List[RetrieverHit] = []
        pops_hits: List[RetrieverHit] = []
        golden_hit: Optional[RetrieverHit] = None
        pops_hit: Optional[RetrieverHit] = None
        golden_context_filtered = False
        pops_context_filtered = False

        if config.enable_golden:
            golden_hits = golden_retriever.search(question, states)
            diagnostics.golden_hits = golden_hits
            golden_hit, golden_context_filtered = self._evaluate_hits(
                golden_hits,
                config.golden_thresholds,
                keywords,
            )

        # Priority logic: If Golden Database has relevant content, skip PoPs search
        pops_dynamic = config.pops_dynamic_distance_multiplier
        if config.enable_pops and not golden_hit:
            # Only search PoPs if Golden Database didn't provide relevant content
            pops_hits = pops_retriever.search(question, states)
            diagnostics.pops_hits = pops_hits
            pops_hit, pops_context_filtered = self._evaluate_hits(
                pops_hits,
                config.pops_thresholds,
                keywords,
                dynamic_multiplier=pops_dynamic,
            )
        elif golden_hit:
            # Golden Database has relevant content, skip PoPs entirely
            logger.info("Golden Database provided relevant content, skipping PoPs search")
            diagnostics.pops_hits = []  # Empty list since we didn't search

        golden_context_hits: List[RetrieverHit] = [golden_hit] if golden_hit else []
        pops_context_hits: List[RetrieverHit] = [pops_hit] if pops_hit else []
        context_provided = bool(golden_context_hits or pops_context_hits)

        def _diag_payload() -> Optional[Dict[str, Any]]:
            if not config.show_diagnostics:
                return None
            return {
                "states_tried": diagnostics.state_attempts,
                "intent_classification": intent_metadata,
                "keywords": keywords,
                "golden": {
                    "hit_count": len(golden_hits),
                    "filtered_for_context": golden_context_filtered,
                    "context_used": bool(golden_context_hits),
                    "top_hits": [
                        {
                            "state": hit.state_used,
                            "cosine": hit.cosine,
                            "distance": hit.distance,
                            "preview": hit.content[:160],
                        }
                        for hit in golden_hits[:3]
                    ],
                },
                "pops": {
                    "hit_count": len(pops_hits),
                    "filtered_for_context": pops_context_filtered,
                    "context_used": bool(pops_context_hits),
                    "top_hits": [
                        {
                            "state": hit.state_used,
                            "cosine": hit.cosine,
                            "distance": hit.distance,
                            "preview": hit.content[:160],
                        }
                        for hit in pops_hits[:3]
                    ],
                },
            }

        keyword_meta = {
            "keywords": keywords,
            "golden_filtered": golden_context_filtered,
            "pops_filtered": pops_context_filtered,
            "context_provided": context_provided,
        }

        clarifying_questions: List[str] = []
        if config.enable_llm and config.clarify_with_llm:
            clarifying_questions = llm_responder.suggest_clarifications(
                question, [hit.source for hit in golden_hits + pops_hits]
            )

        if not config.enable_llm:
            if context_provided:
                primary_hit = golden_hit or pops_hit
                assert primary_hit is not None  # for type checkers
                metadata = {
                    **(primary_hit.metadata or {}),
                    "intent_classification": intent_metadata,
                    "retrieval_keyword_overlap": keyword_meta,
                    "clarifications": clarifying_questions,
                }
                diag = _diag_payload()
                if diag:
                    metadata["diagnostics"] = diag
                if raw_config_metadata:
                    metadata["database_config"] = raw_config_metadata
                source_label = "Golden Database" if golden_hit else "PoPs Database"
                return PipelineResult(
                    answer=primary_hit.content,
                    source=source_label,
                    similarity=primary_hit.cosine,
                    distance=primary_hit.distance,
                    metadata=metadata,
                    reasoning=[f"{source_label} used without LLM synthesis (LLM disabled)"],
                    clarifying_questions=clarifying_questions,
                )

            fallback_metadata: Dict[str, Any] = {
                "intent_classification": intent_metadata,
                "retrieval_keyword_overlap": keyword_meta,
                "clarifications": clarifying_questions,
            }
            diag = _diag_payload()
            if diag:
                fallback_metadata["diagnostics"] = diag
            if raw_config_metadata:
                fallback_metadata["database_config"] = raw_config_metadata
            return PipelineResult(
                answer="No knowledge source produced an answer and LLM fallback is disabled.",
                source="Pipeline",
                reasoning=[
                    "Golden database disabled or no matches",
                    "PoPs database disabled or no matches",
                ],
                clarifying_questions=clarifying_questions,
                metadata=fallback_metadata,
            )

        if golden_hit and golden_context_hits and not pops_hit:
            # Extract only the answer portion from Golden Database content
            raw_content = golden_context_hits[0].content
            
            # Parse the content to extract just the answer
            if "Answer:" in raw_content:
                # Split by "Answer:" and take everything after it
                answer_part = raw_content.split("Answer:", 1)[1].strip()
                direct_answer = answer_part
            else:
                # If no "Answer:" delimiter found, use the full content
                direct_answer = raw_content
            
            direct_reasoning = ["Golden Database"]
            
            direct_metadata: Dict[str, Any] = {
                "clarifications": clarifying_questions,
                "intent_classification": intent_metadata,
                "retrieval_keyword_overlap": keyword_meta,
                "context_provided": True,
                "direct_golden_match": True,
            }
            diag = _diag_payload()
            if diag:
                direct_metadata["diagnostics"] = diag
            if raw_config_metadata:
                direct_metadata["database_config"] = raw_config_metadata

            direct_metadata["retrieved_sources"] = [
                {
                    "source": "Golden Database",
                    "state": golden_hit.state_used if golden_hit else None,
                    "cosine": golden_hit.cosine if golden_hit else None,
                    "distance": golden_hit.distance if golden_hit else None,
                }
            ]
            
            return PipelineResult(
                answer=direct_answer,
                source="Golden Database",
                similarity=golden_hit.cosine,
                distance=golden_hit.distance,
                metadata=direct_metadata,
                reasoning=direct_reasoning,
                clarifying_questions=clarifying_questions,
            )

        llm_context, context_meta = self._build_llm_context(
            golden_context_hits,
            pops_context_hits,
            clarifying_questions,
        )

        llm_error: Optional[str] = None
        try:
            answer = llm_responder.generate_answer(
                question,
                conversation_history,
                context=llm_context if context_provided else "",
                stream=stream,
                token_callback=token_callback,
            )
        except Exception as exc:  # pragma: no cover - network/runtime failure
            logger.exception("LLM fallback generation failed")
            answer = (
                "The AI reasoning engine is currently unavailable. Please confirm the Ollama service is running "
                "and retry your question shortly."
            )
            llm_error = str(exc)

        reasoning: List[str] = []
        if config.enable_golden:
            if golden_hit:
                reasoning.append("Golden database supplied context for the LLM")
            elif not golden_hits:
                reasoning.append("Golden database returned no results")
            else:
                if golden_context_filtered:
                    reasoning.append("Golden hits filtered due to low keyword overlap")
                else:
                    reasoning.append("Golden database did not meet thresholds")
        else:
            reasoning.append("Golden database disabled")

        if config.enable_pops:
            if pops_hit:
                reasoning.append("PoPs database supplied context for the LLM")
            elif not pops_hits:
                reasoning.append("PoPs database returned no results")
            else:
                if pops_context_filtered:
                    reasoning.append("PoPs hits filtered due to low keyword overlap")
                else:
                    reasoning.append("PoPs database did not meet thresholds")
        else:
            reasoning.append("PoPs database disabled")

        if clarifying_questions:
            reasoning.append("Clarification suggested to user")

        metadata: Dict[str, Any] = {
            "clarifications": clarifying_questions,
            "intent_classification": intent_metadata,
            "retrieval_keyword_overlap": keyword_meta,
            "context_provided": context_provided,
        }
        if context_meta:
            metadata["llm_context_sources"] = context_meta
        diag = _diag_payload()
        if diag:
            metadata["diagnostics"] = diag
        if llm_error:
            metadata["llm_error"] = llm_error
        if raw_config_metadata:
            metadata["database_config"] = raw_config_metadata

        metadata["retrieved_sources"] = [
            {
                "source": "Golden Database",
                "state": golden_hit.state_used if golden_hit else None,
                "cosine": golden_hit.cosine if golden_hit else None,
                "distance": golden_hit.distance if golden_hit else None,
            }
            if golden_hit
            else None,
            {
                "source": "PoPs Database",
                "state": pops_hit.state_used if pops_hit else None,
                "cosine": pops_hit.cosine if pops_hit else None,
                "distance": pops_hit.distance if pops_hit else None,
            }
            if pops_hit
            else None,
        ]
        metadata["retrieved_sources"] = [item for item in metadata["retrieved_sources"] if item]

        if not context_provided:
            metadata["context_note"] = "LLM invoked without retrieval context"

        reason_parts: List[str] = []
        if reasoning:
            reason_parts.append("; ".join(reasoning))
        if llm_error:
            reason_parts.append(f"LLM error: {llm_error}")
        self._log_fallback(
            question,
            answer,
            reason=" ; ".join(reason_parts) if reason_parts else "LLM fallback invoked",
        )

        actual_source = "AI Reasoning Engine (gpt-oss)" 
        if golden_hit and pops_hit:
            actual_source = "Package of Practices + Agricultural Database"
        elif pops_hit:
            actual_source = "Package of Practices (PoPs)"
        elif golden_hit:
            actual_source = "Agricultural Database (Golden)"
        elif context_provided:
            actual_source = "AI Reasoning Engine with Agricultural Context"
        
        return PipelineResult(
            answer=answer,
            source=actual_source,
            metadata=metadata,
            reasoning=reasoning,
            clarifying_questions=clarifying_questions,
        )
