"""V1/V2 shadow 结果的纯函数比较。

shadow 比较只对稳定身份、数量、来源定位和候选稳定键做判断，不比较浮点向量值，
也不宣称两个索引的向量召回结果必然完全一致。它可以在没有数据库和模型的情况下
被单测或离线校验入口复用。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from domain_models import Document, DocumentChunk, DocumentVersion, SourceLocator
from legacy_novel import LegacyNovelAdapter

ShadowMismatchCategory = Literal[
    "document_identity",
    "version_identity",
    "source_hash",
    "chunk_count",
    "chunk_missing",
    "chunk_extra",
    "source_locator",
    "candidate_missing",
    "candidate_extra",
    "candidate_order",
]


@dataclass(frozen=True)
class ShadowChunk:
    stable_key: str
    ordinal: int
    locator: SourceLocator


@dataclass(frozen=True)
class ShadowSnapshot:
    document_key: str
    version_key: str
    source_hash: str | None
    chunks: tuple[ShadowChunk, ...]
    candidate_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShadowMismatch:
    category: ShadowMismatchCategory
    key: str
    detail: str


@dataclass(frozen=True)
class ShadowComparison:
    mismatches: tuple[ShadowMismatch, ...]

    @property
    def matches(self) -> bool:
        return not self.mismatches

    @property
    def categories(self) -> tuple[ShadowMismatchCategory, ...]:
        return tuple(dict.fromkeys(item.category for item in self.mismatches))


def _candidate_key(version_id: str, ordinal: int) -> str:
    return f"{version_id}:{ordinal}"


def _legacy_locator(ordinal: int, chapter_title: str | None) -> SourceLocator:
    return SourceLocator(
        kind="chunk",
        value=str(ordinal),
        label=chapter_title or f"片段 {ordinal}",
    )


def _generic_locator(chunk: DocumentChunk) -> SourceLocator:
    if chunk.page_number is not None:
        return SourceLocator(
            kind="page",
            value=str(chunk.page_number),
            label=f"第 {chunk.page_number} 页",
        )
    if chunk.section_path:
        return SourceLocator(
            kind="heading",
            value="/".join(chunk.section_path),
            label=chunk.section_path[-1],
        )
    return SourceLocator(kind="chunk", value=str(chunk.ordinal), label=f"片段 {chunk.ordinal}")


def _machine_locator(locator: SourceLocator) -> tuple[str, str]:
    """只比较机器定位主键；label 是展示文本，不应制造 shadow 噪声。"""

    return locator.kind, locator.value


def snapshot_from_v1_rows(
    novel: str,
    rows: Sequence[Mapping[str, object]],
    *,
    source_hash: str | None = None,
    candidate_ordinals: Sequence[int] = (),
) -> ShadowSnapshot:
    """从 V1 ``novel_chunks`` 行构造不含正文的 shadow 快照。"""

    version_id = LegacyNovelAdapter.version_id(novel, source_hash=source_hash)
    chunks: list[ShadowChunk] = []
    for row in sorted(rows, key=lambda item: int(str(item["chunk_id"]))):
        ordinal = int(str(row["chunk_id"]))
        chapter_title = row.get("chapter_title")
        chunks.append(
            ShadowChunk(
                stable_key=_candidate_key(version_id, ordinal),
                ordinal=ordinal,
                locator=_legacy_locator(
                    ordinal,
                    str(chapter_title) if chapter_title is not None else None,
                ),
            )
        )
    return ShadowSnapshot(
        document_key=LegacyNovelAdapter.document_id(novel),
        version_key=version_id,
        source_hash=source_hash,
        chunks=tuple(chunks),
        candidate_keys=tuple(
            _candidate_key(version_id, ordinal) for ordinal in candidate_ordinals
        ),
    )


def snapshot_from_v2_document(
    document: Document,
    version: DocumentVersion,
    chunks: Sequence[DocumentChunk],
    *,
    candidate_ordinals: Sequence[int] = (),
) -> ShadowSnapshot:
    """从 V2 领域对象构造不含正文的 shadow 快照。"""

    return ShadowSnapshot(
        document_key=document.id,
        version_key=version.id,
        source_hash=version.source_hash,
        chunks=tuple(
            ShadowChunk(
                stable_key=_candidate_key(version.id, chunk.ordinal),
                ordinal=chunk.ordinal,
                locator=_generic_locator(chunk),
            )
            for chunk in sorted(chunks, key=lambda item: item.ordinal)
        ),
        candidate_keys=tuple(
            _candidate_key(version.id, ordinal) for ordinal in candidate_ordinals
        ),
    )


def compare_shadow(v1: ShadowSnapshot, v2: ShadowSnapshot) -> ShadowComparison:
    """比较 V1/V2 的可解释稳定键；不比较 embedding 浮点值。"""

    mismatches: list[ShadowMismatch] = []
    if v1.document_key != v2.document_key:
        mismatches.append(
            ShadowMismatch("document_identity", v1.document_key, f"V2={v2.document_key}")
        )
    if v1.version_key != v2.version_key:
        mismatches.append(
            ShadowMismatch("version_identity", v1.version_key, f"V2={v2.version_key}")
        )
    if v1.source_hash != v2.source_hash:
        mismatches.append(
            ShadowMismatch(
                "source_hash", v1.version_key, f"V1={v1.source_hash!r}, V2={v2.source_hash!r}"
            )
        )
    if len(v1.chunks) != len(v2.chunks):
        mismatches.append(
            ShadowMismatch(
                "chunk_count", v1.version_key, f"V1={len(v1.chunks)}, V2={len(v2.chunks)}"
            )
        )

    v1_chunks = {chunk.stable_key: chunk for chunk in v1.chunks}
    v2_chunks = {chunk.stable_key: chunk for chunk in v2.chunks}
    for key in sorted(v1_chunks.keys() - v2_chunks.keys()):
        mismatches.append(ShadowMismatch("chunk_missing", key, "V2 缺少该 chunk"))
    for key in sorted(v2_chunks.keys() - v1_chunks.keys()):
        mismatches.append(ShadowMismatch("chunk_extra", key, "V2 多出该 chunk"))
    for key in sorted(v1_chunks.keys() & v2_chunks.keys()):
        if _machine_locator(v1_chunks[key].locator) != _machine_locator(
            v2_chunks[key].locator
        ):
            mismatches.append(
                ShadowMismatch(
                    "source_locator",
                    key,
                    f"V1={v1_chunks[key].locator.model_dump(mode='json')}, "
                    f"V2={v2_chunks[key].locator.model_dump(mode='json')}",
                )
            )

    v1_candidates = set(v1.candidate_keys)
    v2_candidates = set(v2.candidate_keys)
    for key in sorted(v1_candidates - v2_candidates):
        mismatches.append(ShadowMismatch("candidate_missing", key, "V2 候选中不存在"))
    for key in sorted(v2_candidates - v1_candidates):
        mismatches.append(ShadowMismatch("candidate_extra", key, "V2 候选中新增"))
    if v1_candidates == v2_candidates and v1.candidate_keys != v2.candidate_keys:
        mismatches.append(
            ShadowMismatch(
                "candidate_order",
                v1.version_key,
                f"V1={v1.candidate_keys}, V2={v2.candidate_keys}",
            )
        )
    return ShadowComparison(mismatches=tuple(mismatches))
