"""把 V2 只读候选接到现有 RAG 的 shadow 观测。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, cast

from chunk_model import SourceChunk
from domain_models import RetrievalScope
from tokenizer import query_terms
from v2_retrieval import V2ReadRepository, V2SearchHit


class _Embedder(Protocol):
    def encode(
        self, texts: list[str], *, normalize_embeddings: bool, show_progress_bar: bool
    ): ...


@dataclass(frozen=True)
class V2ShadowObservation:
    v1_count: int
    v2_count: int
    missing_count: int
    extra_count: int
    elapsed_ms: int

    def payload(self) -> dict[str, object]:
        return {
            "v1_count": self.v1_count,
            "v2_count": self.v2_count,
            "missing_count": self.missing_count,
            "extra_count": self.extra_count,
            "elapsed_ms": self.elapsed_ms,
        }


def _v1_key(source: SourceChunk) -> str:
    return f"{source.novel}:{source.chunk_id}"


def _v2_legacy_key(hit: V2SearchHit) -> str | None:
    legacy_novel = hit.document.metadata.get("legacy_novel")
    if not isinstance(legacy_novel, str) or not legacy_novel:
        return None
    return f"{legacy_novel}:{hit.chunk.ordinal}"


class V2ShadowReader:
    """只读执行 V2 候选并与当前 V1 候选比较；绝不影响最终回答。"""

    def __init__(self, embedder: object, repository: V2ReadRepository | None = None):
        self.embedder = embedder
        self.repository = repository or V2ReadRepository()

    def observe(
        self,
        question: str,
        v1_sources: list[SourceChunk],
        *,
        top_k: int,
        scope: RetrievalScope | None = None,
    ) -> V2ShadowObservation:
        started = time.perf_counter()
        encoded = cast(_Embedder, self.embedder).encode(
            [question], normalize_embeddings=True, show_progress_bar=False
        )
        v2_hits = self.repository.vector_search(encoded[0], top_k=top_k, scope=scope)
        v2_hits.extend(
            self.repository.keyword_search(query_terms(question), top_k=top_k, scope=scope)
        )
        v1_keys = {_v1_key(source) for source in v1_sources}
        v2_keys = {key for hit in v2_hits if (key := _v2_legacy_key(hit)) is not None}
        return V2ShadowObservation(
            v1_count=len(v1_keys),
            v2_count=len(v2_keys),
            missing_count=len(v1_keys - v2_keys),
            extra_count=len(v2_keys - v1_keys),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
