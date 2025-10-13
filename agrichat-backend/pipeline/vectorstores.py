import os
from functools import lru_cache
from typing import Optional

from langchain_community.vectorstores import Chroma

from .llm_adapter import local_embeddings


@lru_cache(maxsize=1)
def _resolve_chroma_path() -> str:
    if os.path.exists("/app"):
        return "/app/chromaDb"
    return "/home/ubuntu/agrichat-annam/agrichat-backend/chromaDb"


class VectorStores:
    """Lazy-initialized handles to Golden and PoPs Chroma collections."""

    def __init__(self, chroma_path: Optional[str] = None):
        self._chroma_path = chroma_path or _resolve_chroma_path()
        self._golden = None
        self._pops = None

    @property
    def golden(self) -> Chroma:
        if self._golden is None:
            self._golden = Chroma(
                collection_name="langchain",
                persist_directory=self._chroma_path,
                embedding_function=local_embeddings,
            )
        return self._golden

    @property
    def pops(self) -> Optional[Chroma]:
        if self._pops is None:
            try:
                self._pops = Chroma(
                    collection_name="package_of_practices",
                    persist_directory=self._chroma_path,
                    embedding_function=local_embeddings,
                )
            except Exception:
                self._pops = None
        return self._pops
