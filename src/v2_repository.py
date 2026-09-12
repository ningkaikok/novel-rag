"""V2 repository 与索引发布契约。

本模块只接收已经由现有 ingest 路径算好的 embedding 和 BM25 term 频率，不加载
模型，也不重复实现分词。发布前先把输入收敛成确定性的计划；显式调用
``publish_v2_index`` 后，才会在一个事务内按父表、chunk、term、manifest 的顺序
upsert 到 ``knowledge_v2``。manifest 是最后一个写入对象，任何失败都应由 executor
的事务上下文回滚。

默认 ``STORAGE_SCHEMA=v1`` 时发布会被拒绝。模块导入、dry-run 和计划构造都不会
连接数据库或改变 V1 表。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Literal, Protocol

from domain_models import Collection, Document, DocumentChunk, DocumentVersion
from postgres import vector_literal
from v2_migration import V2ChunkTerm, V2MigrationDocument, V2MigrationPlan

V2_SCHEMA_NAME = "knowledge_v2"
StorageSchema = Literal["v1", "v2", "shadow"]


class V2Executor(Protocol):
    """psycopg connection 所需的最小可 mock 接口。"""

    def execute(self, query: str, params: Sequence[object] = ()) -> object: ...

    def transaction(self) -> AbstractContextManager[object]: ...


@dataclass(frozen=True)
class V2IndexInput:
    """一个待发布文档及其已计算好的索引产物。

    ``embeddings`` 的 key 必须是 ``DocumentChunk.id``；``terms`` 由现有
    ``tokenizer.term_frequencies`` 产出后传入，repository 不会再次调用 tokenizer。
    """

    collection: Collection
    document: Document
    version: DocumentVersion
    chunks: tuple[DocumentChunk, ...]
    embeddings: Mapping[str, Sequence[float]]
    terms: Mapping[str, Mapping[str, int]]
    pipeline_hash: str


@dataclass(frozen=True)
class V2IndexedChunk:
    chunk: DocumentChunk
    embedding: tuple[float, ...]
    terms: tuple[V2ChunkTerm, ...]

    @property
    def token_count(self) -> int:
        return sum(term.tf for term in self.terms)


@dataclass(frozen=True)
class V2PublicationDocument:
    collection: Collection
    document: Document
    version: DocumentVersion
    chunks: tuple[V2IndexedChunk, ...]
    pipeline_hash: str

    @property
    def source_hash(self) -> str:
        source_hash = self.version.source_hash
        if not source_hash:
            raise V2PublicationError(f"version 缺少 source_hash: {self.version.id}")
        return source_hash


@dataclass(frozen=True)
class V2PublicationPlan:
    embedding_dimension: int
    documents: tuple[V2PublicationDocument, ...]

    @property
    def chunk_count(self) -> int:
        return sum(len(document.chunks) for document in self.documents)

    @property
    def term_count(self) -> int:
        return sum(
            len(chunk.terms) for document in self.documents for chunk in document.chunks
        )

    def summary(self) -> dict[str, object]:
        """返回不含正文和向量的 dry-run 摘要。"""

        return {
            "schema": V2_SCHEMA_NAME,
            "documents": len(self.documents),
            "chunks": self.chunk_count,
            "terms": self.term_count,
            "embedding_dimension": self.embedding_dimension,
            "version_ids": [document.version.id for document in self.documents],
        }


class V2PublicationError(ValueError):
    """索引发布输入不满足 V2 不变量。"""


class V2PublishDisabled(RuntimeError):
    """调用者没有明确选择允许 V2/shadow 发布。"""


def _configured_storage_schema() -> str:
    # 延迟读取，方便显式入口在进程测试或启动配置变更后仍读取当前配置。
    from config import STORAGE_SCHEMA

    return STORAGE_SCHEMA


def _json_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_dimension(dimension: int) -> int:
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise V2PublicationError("embedding dimension must be a positive integer")
    return dimension


def _normalise_embedding(
    values: Sequence[float], *, dimension: int, chunk_id: str
) -> tuple[float, ...]:
    embedding = tuple(float(value) for value in values)
    if len(embedding) != dimension:
        raise V2PublicationError(
            f"embedding 维度不一致: {chunk_id} 期望 {dimension}，实际 {len(embedding)}"
        )
    if not all(math.isfinite(value) for value in embedding):
        raise V2PublicationError(f"embedding 包含非有限值: {chunk_id}")
    return embedding


def _normalise_terms(
    chunk_id: str,
    frequencies: Mapping[str, int],
) -> tuple[V2ChunkTerm, ...]:
    terms: list[V2ChunkTerm] = []
    for term, tf in sorted(frequencies.items()):
        if not term or isinstance(tf, bool) or int(tf) <= 0:
            raise V2PublicationError(f"非法 BM25 term: {chunk_id}/{term!r}")
        terms.append(V2ChunkTerm(chunk_id=chunk_id, term=term, tf=int(tf)))
    return tuple(terms)


def build_v2_publication_plan(
    inputs: Sequence[V2IndexInput],
    *,
    embedding_dimension: int,
) -> V2PublicationPlan:
    """校验并构造确定性的 V2 upsert 计划。

    调用方负责复用 ``src/ingest.py`` 的 embedding 模型和 ``tokenizer``，这里只
    验证结果并排序。输入顺序不影响最终 SQL 发布顺序，因此重复 apply 是幂等的。
    """

    dimension = _validate_dimension(embedding_dimension)
    documents: list[V2PublicationDocument] = []
    seen_collections: dict[str, Collection] = {}
    seen_documents: set[str] = set()
    seen_versions: set[str] = set()
    seen_chunks: set[str] = set()
    for item in sorted(inputs, key=lambda value: (value.document.id, value.version.id)):
        if item.document.id in seen_documents:
            raise V2PublicationError(f"重复 document: {item.document.id}")
        if item.version.id in seen_versions:
            raise V2PublicationError(f"重复 version: {item.version.id}")
        if item.document.collection_id != item.collection.id:
            raise V2PublicationError(f"document 未归属 collection: {item.document.id}")
        previous_collection = seen_collections.get(item.collection.id)
        if previous_collection is not None and previous_collection != item.collection:
            raise V2PublicationError(f"同一 collection id 的定义不一致: {item.collection.id}")
        seen_collections[item.collection.id] = item.collection
        if item.version.document_id != item.document.id:
            raise V2PublicationError(f"version 未归属 document: {item.version.id}")
        if not item.version.source_hash:
            raise V2PublicationError(f"version 缺少 source_hash: {item.version.id}")
        if not item.pipeline_hash:
            raise V2PublicationError(f"缺少 pipeline_hash: {item.version.id}")

        ordinals = [chunk.ordinal for chunk in item.chunks]
        if ordinals != list(range(len(item.chunks))):
            raise V2PublicationError(
                f"chunk ordinal 必须从 0 连续递增: {item.version.id} -> {ordinals}"
            )
        chunk_ids = {chunk.id for chunk in item.chunks}
        if set(item.embeddings) != chunk_ids:
            missing = sorted(chunk_ids - set(item.embeddings))
            extra = sorted(set(item.embeddings) - chunk_ids)
            raise V2PublicationError(
                f"embedding 与 chunk 不一致: missing={missing}, extra={extra}"
            )
        extra_terms = sorted(set(item.terms) - chunk_ids)
        if extra_terms:
            raise V2PublicationError(f"term 指向不存在的 chunk: {extra_terms}")

        indexed_chunks: list[V2IndexedChunk] = []
        for chunk in item.chunks:
            if chunk.document_version_id != item.version.id:
                raise V2PublicationError(f"chunk 未归属 version: {chunk.id}")
            if chunk.id in seen_chunks:
                raise V2PublicationError(f"重复 chunk: {chunk.id}")
            seen_chunks.add(chunk.id)
            indexed_chunks.append(
                V2IndexedChunk(
                    chunk=chunk,
                    embedding=_normalise_embedding(
                        item.embeddings[chunk.id],
                        dimension=dimension,
                        chunk_id=chunk.id,
                    ),
                    terms=_normalise_terms(chunk.id, item.terms.get(chunk.id, {})),
                )
            )
        seen_documents.add(item.document.id)
        seen_versions.add(item.version.id)
        documents.append(
            V2PublicationDocument(
                collection=item.collection,
                document=item.document,
                version=item.version,
                chunks=tuple(indexed_chunks),
                pipeline_hash=item.pipeline_hash,
            )
        )
    return V2PublicationPlan(embedding_dimension=dimension, documents=tuple(documents))


def publication_input_from_migration(
    item: V2MigrationDocument, embeddings: Mapping[str, Sequence[float]]
) -> V2IndexInput:
    """把 Phase 2 migration document 接到当前 ingest 的 embedding 结果。"""

    terms: dict[str, dict[str, int]] = {}
    for term in item.terms:
        terms.setdefault(term.chunk_id, {})[term.term] = term.tf
    return V2IndexInput(
        collection=item.collection,
        document=item.document,
        version=item.version,
        chunks=item.chunks,
        embeddings=embeddings,
        terms=terms,
        pipeline_hash=item.manifest.pipeline_hash,
    )


def publication_inputs_from_migration(
    plan: V2MigrationPlan,
    embeddings_by_chunk_id: Mapping[str, Sequence[float]],
) -> tuple[V2IndexInput, ...]:
    """把 Phase 2 全部 migration documents 转成发布输入。"""

    return tuple(
        publication_input_from_migration(item, embeddings_by_chunk_id)
        for item in plan.documents
    )


_UPSERT_COLLECTION = f"""
INSERT INTO {V2_SCHEMA_NAME}.collections
    (id, name, description, metadata)
VALUES (%s, %s, %s, %s::jsonb)
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    metadata = EXCLUDED.metadata
"""

_UPSERT_DOCUMENT = f"""
INSERT INTO {V2_SCHEMA_NAME}.documents
    (id, collection_id, title, source_type, metadata)
VALUES (%s, %s, %s, %s, %s::jsonb)
ON CONFLICT (id) DO UPDATE SET
    collection_id = EXCLUDED.collection_id,
    title = EXCLUDED.title,
    source_type = EXCLUDED.source_type,
    metadata = EXCLUDED.metadata
"""

_UPSERT_VERSION = f"""
INSERT INTO {V2_SCHEMA_NAME}.document_versions
    (id, document_id, version_no, source_hash, parser_name, parser_version, pipeline_hash)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    document_id = EXCLUDED.document_id,
    version_no = EXCLUDED.version_no,
    source_hash = EXCLUDED.source_hash,
    parser_name = EXCLUDED.parser_name,
    parser_version = EXCLUDED.parser_version,
    pipeline_hash = EXCLUDED.pipeline_hash
"""

_DELETE_TERMS_FOR_VERSION = f"""
DELETE FROM {V2_SCHEMA_NAME}.chunk_terms
WHERE chunk_id IN (
    SELECT id FROM {V2_SCHEMA_NAME}.document_chunks
    WHERE document_version_id = %s
)
"""

_DELETE_CHUNKS_FOR_VERSION = f"""
DELETE FROM {V2_SCHEMA_NAME}.document_chunks
WHERE document_version_id = %s
"""

_UPSERT_CHUNK = f"""
INSERT INTO {V2_SCHEMA_NAME}.document_chunks
    (id, document_version_id, ordinal, section_path, page_number, text, context,
     token_count, embedding, metadata)
VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s::vector, %s::jsonb)
ON CONFLICT (id) DO UPDATE SET
    document_version_id = EXCLUDED.document_version_id,
    ordinal = EXCLUDED.ordinal,
    section_path = EXCLUDED.section_path,
    page_number = EXCLUDED.page_number,
    text = EXCLUDED.text,
    context = EXCLUDED.context,
    token_count = EXCLUDED.token_count,
    embedding = EXCLUDED.embedding,
    metadata = EXCLUDED.metadata
"""

_UPSERT_TERM = f"""
INSERT INTO {V2_SCHEMA_NAME}.chunk_terms (chunk_id, term, tf)
VALUES (%s, %s, %s)
ON CONFLICT (chunk_id, term) DO UPDATE SET tf = EXCLUDED.tf
"""

_UPSERT_MANIFEST = f"""
INSERT INTO {V2_SCHEMA_NAME}.index_manifests
    (document_version_id, source_hash, pipeline_hash, chunk_count, quality_report)
VALUES (%s, %s, %s, %s, %s::jsonb)
ON CONFLICT (document_version_id) DO UPDATE SET
    source_hash = EXCLUDED.source_hash,
    pipeline_hash = EXCLUDED.pipeline_hash,
    chunk_count = EXCLUDED.chunk_count,
    quality_report = EXCLUDED.quality_report,
    indexed_at = NOW()
"""


@dataclass(frozen=True)
class V2PublishResult:
    documents: int
    chunks: int
    terms: int


def dry_run_v2_index(plan: V2PublicationPlan) -> dict[str, object]:
    """显式 dry-run 入口；不会要求 executor，也不会连接数据库。"""

    return plan.summary()


def _ensure_publish_enabled(storage_schema: str | None) -> StorageSchema:
    selected = _configured_storage_schema() if storage_schema is None else storage_schema
    if selected not in {"v2", "shadow"}:
        raise V2PublishDisabled(
            f"V2 发布需要显式选择 STORAGE_SCHEMA=v2 或 shadow，当前为 {selected!r}"
        )
    return selected  # type: ignore[return-value]


def publish_v2_index(
    executor: V2Executor,
    plan: V2PublicationPlan,
    *,
    storage_schema: str | None = None,
) -> V2PublishResult:
    """在一个显式事务内发布 V2 索引。

    该入口不创建 schema、不调用 embedding、不删除 V1、不改变 API/RAG。executor
    必须提供 psycopg 风格的 ``transaction()``，其上下文负责 rollback/commit。
    ``storage_schema`` 省略时读取配置，默认 V1 会直接拒绝。
    """

    _ensure_publish_enabled(storage_schema)
    with executor.transaction():
        collections: dict[str, Collection] = {}
        for document in plan.documents:
            collections[document.collection.id] = document.collection
        for collection in sorted(collections.values(), key=lambda value: value.id):
            executor.execute(
                _UPSERT_COLLECTION,
                (
                    collection.id,
                    collection.name,
                    collection.description,
                    _json_value({}),
                ),
            )

        for document in plan.documents:
            executor.execute(
                _UPSERT_DOCUMENT,
                (
                    document.document.id,
                    document.document.collection_id,
                    document.document.title,
                    document.document.source_type,
                    _json_value(document.document.metadata),
                ),
            )
            executor.execute(
                _UPSERT_VERSION,
                (
                    document.version.id,
                    document.version.document_id,
                    document.version.version_no,
                    document.source_hash,
                    document.version.parser_name,
                    document.version.parser_version,
                    document.pipeline_hash,
                ),
            )
            # 同一 version 重发布可能减少 chunk；先清理 V2 派生行，避免旧 chunk/term
            # 残留。删除与重建处于同一事务，失败会 rollback；V1 表完全不在此范围内。
            executor.execute(_DELETE_TERMS_FOR_VERSION, (document.version.id,))
            executor.execute(_DELETE_CHUNKS_FOR_VERSION, (document.version.id,))
            for indexed in document.chunks:
                chunk = indexed.chunk
                executor.execute(
                    _UPSERT_CHUNK,
                    (
                        chunk.id,
                        chunk.document_version_id,
                        chunk.ordinal,
                        _json_value(list(chunk.section_path)),
                        chunk.page_number,
                        chunk.text,
                        chunk.context,
                        indexed.token_count,
                        vector_literal(indexed.embedding),
                        _json_value(chunk.metadata),
                    ),
                )
                for term in indexed.terms:
                    executor.execute(_UPSERT_TERM, (term.chunk_id, term.term, term.tf))

            # manifest 必须最后发布：上游任一写入失败时，事务上下文不应提交检查点。
            executor.execute(
                _UPSERT_MANIFEST,
                (
                    document.version.id,
                    document.source_hash,
                    document.pipeline_hash,
                    len(document.chunks),
                    "{}",
                ),
            )
    return V2PublishResult(
        documents=len(plan.documents),
        chunks=plan.chunk_count,
        terms=plan.term_count,
    )
