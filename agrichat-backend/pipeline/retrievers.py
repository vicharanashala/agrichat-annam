import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from langchain_community.vectorstores import Chroma

from .llm_adapter import local_embeddings

from .config import PipelineConfig
from .types import RetrieverHit


@dataclass
class RetrieverConfig:
    k: int
    thresholds: float


GENERAL_STATE_TOKENS = {"general", "india", "nationwide", "all", "pan-india"}
GENERAL_CROP_TOKENS = {
    "general",
    "all",
    "multiple crops",
    "various crops",
    "mixed crops",
    "multi crops",
    "all crops",
    "pan-india",
    "india",
}


def _normalize_state(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _is_general_state(value: Optional[str]) -> bool:
    normalized = _normalize_state(value)
    return normalized in GENERAL_STATE_TOKENS or normalized == ""


def _compute_cosine(text_a: str, text_b: str) -> float:
    query_vec = local_embeddings.embed_query(text_a)
    doc_vec = local_embeddings.embed_query(text_b)
    dot = sum(a * b for a, b in zip(query_vec, doc_vec))
    norm_q = sum(a * a for a in query_vec) ** 0.5
    norm_d = sum(a * a for a in doc_vec) ** 0.5
    if norm_q == 0 or norm_d == 0:
        return 0.0
    return dot / (norm_q * norm_d)


class GoldenRetriever:
    def __init__(self, store: Chroma, config: PipelineConfig):
        self.store = store
        self.config = config
        self._question_cache: dict[str, str] = {}

    def _question_lower(self, question: str) -> str:
        cached = self._question_cache.get(question)
        if cached is not None:
            return cached
        lowered = question.lower()
        self._question_cache[question] = lowered
        return lowered

    @staticmethod
    def _question_mentions_phrase(question_lower: str, phrase: Optional[str]) -> bool:
        if not phrase:
            return True
        normalized = phrase.strip().lower()
        if not normalized or normalized in GENERAL_CROP_TOKENS:
            return True

        if normalized in question_lower:
            return True

        tokens = [token for token in re.split(r"[\s,()/\\|-]+", normalized) if token]
        if not tokens:
            return True

        for token in tokens:
            if token in GENERAL_CROP_TOKENS:
                return True
            if re.search(rf"\b{re.escape(token)}\b", question_lower):
                return True
        return False

    def _iter_results(
        self,
        question: str,
        enforce_match: bool,
        state: str,
    ) -> Iterable[tuple]:
        """Yield (doc, distance) pairs with graceful fallbacks for metadata casing mismatches."""

        if not enforce_match:
            try:
                yield from self.store.similarity_search_with_score(question, k=self.config.golden_k)
            except Exception:
                return
            return

        candidate_filters = [state]
        upper_state = state.upper()
        if upper_state not in candidate_filters:
            candidate_filters.append(upper_state)
        title_state = state.title()
        if title_state not in candidate_filters:
            candidate_filters.append(title_state)

        for candidate in candidate_filters:
            try:
                results = self.store.similarity_search_with_score(
                    question,
                    k=self.config.golden_k,
                    filter={"State": candidate},
                )
            except Exception:
                results = []
            if results:
                for item in results:
                    yield item
                return

        try:
            results = self.store.similarity_search_with_score(question, k=self.config.golden_k)
        except Exception:
            results = []
        for item in results:
            yield item

    def search(self, question: str, states: Sequence[str]) -> List[RetrieverHit]:
        hits: List[RetrieverHit] = []
        question_lower = self._question_lower(question)
        for state in states:
            normalized_state = _normalize_state(state)
            filter_dict = None
            enforce_match = normalized_state not in GENERAL_STATE_TOKENS and normalized_state != ""
            results_iter = self._iter_results(question, enforce_match, state)
            state_hits: List[RetrieverHit] = []
            for doc, distance in results_iter:
                metadata = getattr(doc, "metadata", {}) or {}
                doc_state = metadata.get("State")
                normalized_doc_state = _normalize_state(doc_state)

                if enforce_match and normalized_doc_state != normalized_state:
                    continue

                if not enforce_match and not _is_general_state(normalized_doc_state):
                    continue

                crop_label = metadata.get("Crop") or metadata.get("crop")
                if not self._question_mentions_phrase(question_lower, crop_label):
                    continue

                cosine = _compute_cosine(question, doc.page_content)
                state_hits.append(
                    RetrieverHit(
                        source="Golden Database",
                        content=doc.page_content,
                        metadata=metadata,
                        distance=distance,
                        cosine=cosine,
                        state_used=doc_state or state,
                    )
                )

            if state_hits:
                state_hits.sort(key=lambda h: (-(h.cosine or 0.0), h.distance or 9999.0))
                return state_hits

        return hits


class PopsRetriever:
    def __init__(self, store: Optional[Chroma], config: PipelineConfig):
        self.store = store
        self.config = config

    def available(self) -> bool:
        return self.store is not None

    def search(self, question: str, states: Sequence[str]) -> List[RetrieverHit]:
        if not self.available():
            return []

        hits: List[RetrieverHit] = []
        question_lower = question.lower()
        for state in states:
            normalized_state = _normalize_state(state)
            enforce_match = normalized_state not in GENERAL_STATE_TOKENS and normalized_state != ""
            candidate_filters = [state]
            if enforce_match:
                upper_state = state.upper()
                if upper_state not in candidate_filters:
                    candidate_filters.append(upper_state)
                title_state = state.title()
                if title_state not in candidate_filters:
                    candidate_filters.append(title_state)
            else:
                candidate_filters = [None]

            results = []
            for candidate in candidate_filters:
                try:
                    results = self.store.similarity_search_with_score(
                        question,
                        k=self.config.pops_k,
                        filter=None if candidate is None else {"State": candidate},
                    )
                except Exception:
                    results = []
                if results:
                    break
            if not results and enforce_match:
                try:
                    results = self.store.similarity_search_with_score(question, k=self.config.pops_k)
                except Exception:
                    results = []

            state_hits: List[RetrieverHit] = []
            for doc, distance in results:
                cosine = _compute_cosine(question, doc.page_content)
                metadata = getattr(doc, "metadata", {}) or {}
                doc_state = metadata.get("State")
                normalized_doc_state = _normalize_state(doc_state)

                if enforce_match and normalized_doc_state != normalized_state:
                    continue

                if not enforce_match and not _is_general_state(normalized_doc_state):
                    continue

                crop_label = metadata.get("Crop") or metadata.get("crop")
                if not GoldenRetriever._question_mentions_phrase(question_lower, crop_label):
                    continue

                hits.append(
                    RetrieverHit(
                        source="PoPs Database",
                        content=doc.page_content,
                        metadata=metadata,
                        distance=distance,
                        cosine=cosine,
                        state_used=doc_state or state,
                    )
                )
                state_hits.append(hits[-1])

            if state_hits and enforce_match:
                break
        hits.sort(key=lambda h: (-(h.cosine or 0.0), h.distance or 9999.0))
        return hits
