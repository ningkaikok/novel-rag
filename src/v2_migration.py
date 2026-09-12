"""LegacyNovel → V2 的确定性迁移计划与校验器。

本阶段不直接写数据库。迁移器接收从 V1 查询出的 snapshot，生成完整、可重复比较的
V2 对象计划；只有计划通过校验，未来的显式 apply 步骤才可以安全落库。这样可以先在
CI、备份副本或 dry-run 中发现重复、断链、哈希不一致和引用定位问题。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

from domain_models import Collection, Document, DocumentChunk, DocumentVersion
from legacy_novel import LegacyNovelAdapter


@dataclass(frozen=True)
class LegacyNovelSnapshot:
    """一次从 V1 读取的小说快照；正文只在运行时传入，不由迁移器写入仓库。"""

    novel: str
    source_hash: str
    pipeline_hash: str
    chunks: tuple[Mapping[str, object], ...]
    terms: Mapping[int, Mapping[str, int]]


@dataclass(frozen=True)
class V2ChunkTerm:
    chunk_id: str
    term: str
    tf: int


@dataclass(frozen=True)
class V2IndexManifest:
    document_version_id: str
    source_hash: str
    pipeline_hash: str
    chunk_count: int


@dataclass(frozen=True)
class V2MigrationDocument:
    collection: Collection
    document: Document
    version: DocumentVersion
    chunks: tuple[DocumentChunk, ...]
    terms: tuple[V2ChunkTerm, ...]
    manifest: V2IndexManifest


@dataclass(frozen=True)
class V2MigrationPlan:
    schema_version: str
    documents: tuple[V2MigrationDocument, ...]

    @property
    def collections(self) -> tuple[Collection, ...]:
        return tuple(item.collection for item in self.documents)

    @property
    def chunk_count(self) -> int:
        return sum(len(item.chunks) for item in self.documents)

    @property
    def term_count(self) -> int:
        return sum(len(item.terms) for item in self.documents)

    def idempotency_fingerprint(self) -> str:
        """按身份/版本/哈希/定位生成稳定指纹，不把正文放进摘要。"""

        payload = [
            {
                "collection_id": item.collection.id,
                "document_id": item.document.id,
                "version_id": item.version.id,
                "source_hash": item.version.source_hash,
                "pipeline_hash": item.manifest.pipeline_hash,
                "chunks": [
                    {
                        "id": chunk.id,
                        "ordinal": chunk.ordinal,
                        "section_path": chunk.section_path,
                    }
                    for chunk in item.chunks
                ],
                "terms": [
                    {"chunk_id": term.chunk_id, "term": term.term, "tf": term.tf}
                    for term in item.terms
                ],
            }
            for item in self.documents
        ]
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return sha256(encoded).hexdigest()

    def summary(self) -> dict[str, object]:
        """返回 dry-run 可打印的元数据摘要，不包含原文。"""

        return {
            "schema_version": self.schema_version,
            "documents": len(self.documents),
            "collections": len(self.collections),
            "chunks": self.chunk_count,
            "terms": self.term_count,
            "document_ids": [item.document.id for item in self.documents],
            "idempotency_fingerprint": self.idempotency_fingerprint(),
        }


@dataclass(frozen=True)
class MigrationValidation:
    valid: bool
    errors: tuple[str, ...]


class MigrationPlanError(ValueError):
    """输入快照不能安全转换为 V2 计划。"""


def validate_v2_plan(plan: V2MigrationPlan) -> MigrationValidation:
    """校验父子关系、重复键、chunk 定位、term 关系和 manifest 一致性。"""

    errors: list[str] = []
    collection_ids: set[str] = set()
    document_ids: set[str] = set()
    version_ids: set[str] = set()
    chunk_ids: set[str] = set()
    for item in plan.documents:
        if item.collection.id in collection_ids:
            errors.append(f"重复 collection id: {item.collection.id}")
        collection_ids.add(item.collection.id)
        if item.document.id in document_ids:
            errors.append(f"重复 document id: {item.document.id}")
        document_ids.add(item.document.id)
        if item.version.id in version_ids:
            errors.append(f"重复 version id: {item.version.id}")
        version_ids.add(item.version.id)
        if item.document.collection_id != item.collection.id:
            errors.append(f"document 未归属 collection: {item.document.id}")
        if item.version.document_id != item.document.id:
            errors.append(f"version 未归属 document: {item.version.id}")
        if item.manifest.document_version_id != item.version.id:
            errors.append(f"manifest 未归属 version: {item.version.id}")
        if item.manifest.source_hash != item.version.source_hash:
            errors.append(f"manifest/source hash 不一致: {item.version.id}")
        if item.manifest.chunk_count != len(item.chunks):
            errors.append(f"manifest chunk_count 不一致: {item.version.id}")

        local_chunk_ids: set[str] = set()
        local_ordinals: set[int] = set()
        for chunk in item.chunks:
            if chunk.id in chunk_ids:
                errors.append(f"重复 chunk id: {chunk.id}")
            chunk_ids.add(chunk.id)
            if chunk.id in local_chunk_ids:
                errors.append(f"版本内重复 chunk id: {chunk.id}")
            local_chunk_ids.add(chunk.id)
            if chunk.ordinal in local_ordinals:
                errors.append(f"版本内重复 ordinal: {chunk.ordinal}")
            local_ordinals.add(chunk.ordinal)
            if chunk.document_version_id != item.version.id:
                errors.append(f"chunk 未归属 version: {chunk.id}")

        term_chunk_ids = {term.chunk_id for term in item.terms}
        missing = term_chunk_ids - local_chunk_ids
        errors.extend(f"term 指向不存在的 chunk: {chunk_id}" for chunk_id in sorted(missing))
        for term in item.terms:
            if not term.term:
                errors.append(f"term 为空: {term.chunk_id}")
            if term.tf <= 0:
                errors.append(f"term tf 必须为正数: {term.chunk_id}/{term.term}")

    return MigrationValidation(valid=not errors, errors=tuple(errors))


def build_v2_migration_plan(
    snapshots: Sequence[LegacyNovelSnapshot],
) -> V2MigrationPlan:
    """把 V1 snapshot 转为确定性 V2 计划；同一输入顺序不同也得到同一计划。"""

    ordered = sorted(snapshots, key=lambda item: item.novel)
    novels = [item.novel for item in ordered]
    if len(set(novels)) != len(novels):
        duplicates = sorted({novel for novel in novels if novels.count(novel) > 1})
        raise MigrationPlanError(f"snapshot 中存在重复小说: {duplicates}")

    documents: list[V2MigrationDocument] = []
    for snapshot in ordered:
        if not snapshot.novel or not snapshot.source_hash or not snapshot.pipeline_hash:
            raise MigrationPlanError("novel/source_hash/pipeline_hash 都不能为空")
        collection = LegacyNovelAdapter.collection(snapshot.novel)
        document = LegacyNovelAdapter.document(snapshot.novel)
        version = LegacyNovelAdapter.version(
            snapshot.novel,
            source_hash=snapshot.source_hash,
        )

        chunks: list[DocumentChunk] = []
        chunk_by_ordinal: dict[int, DocumentChunk] = {}
        for row in sorted(snapshot.chunks, key=lambda value: int(str(value["chunk_id"]))):
            if str(row.get("novel", snapshot.novel)) != snapshot.novel:
                raise MigrationPlanError(
                    f"chunk 的 novel 与 snapshot 不一致: {snapshot.novel}"
                )
            chunk = LegacyNovelAdapter.chunk_from_row(
                row,
                source_hash=snapshot.source_hash,
            )
            if chunk.ordinal in chunk_by_ordinal:
                raise MigrationPlanError(
                    f"小说 {snapshot.novel} 存在重复 chunk_id: {chunk.ordinal}"
                )
            chunk_by_ordinal[chunk.ordinal] = chunk
            chunks.append(chunk)

        terms: list[V2ChunkTerm] = []
        unknown_term_chunks = set(snapshot.terms) - set(chunk_by_ordinal)
        if unknown_term_chunks:
            raise MigrationPlanError(
                f"小说 {snapshot.novel} 的 term 指向不存在的 chunk: "
                f"{sorted(unknown_term_chunks)}"
            )
        for ordinal in sorted(snapshot.terms):
            chunk = chunk_by_ordinal[ordinal]
            for term, tf in sorted(snapshot.terms[ordinal].items()):
                if not term or int(tf) <= 0:
                    raise MigrationPlanError(f"非法 term: {snapshot.novel}/{ordinal}/{term!r}")
                terms.append(V2ChunkTerm(chunk_id=chunk.id, term=term, tf=int(tf)))

        documents.append(
            V2MigrationDocument(
                collection=collection,
                document=document,
                version=version,
                chunks=tuple(chunks),
                terms=tuple(terms),
                manifest=V2IndexManifest(
                    document_version_id=version.id,
                    source_hash=snapshot.source_hash,
                    pipeline_hash=snapshot.pipeline_hash,
                    chunk_count=len(chunks),
                ),
            )
        )

    plan = V2MigrationPlan(schema_version="1", documents=tuple(documents))
    validation = validate_v2_plan(plan)
    if not validation.valid:
        raise MigrationPlanError("; ".join(validation.errors))
    return plan
