"""knowledge_v2 的只读检索适配器。

这个模块只负责从通用 V2 表读取候选，不负责 embedding、回答生成或写入。它把
数据库行恢复为 ``DocumentChunk`` 和完整的父级身份，调用方可以在 shadow 模式下
把 ``V2SearchHit`` 投影成旧 ``SourceChunk``，也可以直接使用通用领域对象。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from domain_models import Collection, Document, DocumentChunk, DocumentVersion, RetrievalScope
from postgres import connect, vector_literal
from tokenizer import query_terms

V2_SCHEMA_NAME = "knowledge_v2"
DEFAULT_PER_TERM_LIMIT = 200


@dataclass(frozen=True)
class V2SearchHit:
    """V2 检索命中及其父级身份；``distance`` 越小越相关。"""

    collection: Collection
    document: Document
    version: DocumentVersion
    chunk: DocumentChunk
    distance: float


def _scope_conditions(scope: RetrievalScope | None) -> tuple[str, list[object]]:
    if scope is None:
        return "", []
    conditions: list[str] = []
    params: list[object] = []
    if scope.collection_id:
        conditions.append("c.id = %s")
        params.append(scope.collection_id)
    if scope.document_id:
        conditions.append("d.id = %s")
        params.append(scope.document_id)
    if scope.version_id:
        conditions.append("dv.id = %s")
        params.append(scope.version_id)
    return (" AND ".join(conditions), params) if conditions else ("", [])


def _where(conditions: str) -> str:
    return f"WHERE {conditions}" if conditions else ""


def _as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("V2 section_path 必须是 JSON 数组")
    return tuple(str(item) for item in value)


def _as_dict(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("V2 metadata 必须是 JSON 对象")
    return dict(value)


def _hit_from_row(row: Mapping[str, object]) -> V2SearchHit:
    collection = Collection(
        id=str(row["collection_id"]),
        name=str(row["collection_name"]),
        description=str(row.get("collection_description") or ""),
    )
    document = Document(
        id=str(row["document_id"]),
        collection_id=collection.id,
        title=str(row["document_title"]),
        source_type=str(row["source_type"]),  # type: ignore[arg-type]
        metadata=_as_dict(row.get("document_metadata")),
    )
    version = DocumentVersion(
        id=str(row["version_id"]),
        document_id=document.id,
        version_no=int(row["version_no"]),
        source_hash=str(row["source_hash"]) if row.get("source_hash") else None,
        parser_name=str(row["parser_name"]),
        parser_version=str(row["parser_version"]),
    )
    chunk = DocumentChunk(
        id=str(row["chunk_id"]),
        document_version_id=version.id,
        ordinal=int(row["ordinal"]),
        text=str(row["text"]),
        section_path=_as_tuple(row.get("section_path")),
        page_number=int(row["page_number"]) if row.get("page_number") is not None else None,
        context=str(row.get("context") or ""),
        metadata=_as_dict(row.get("chunk_metadata")),
    )
    return V2SearchHit(
        collection=collection,
        document=document,
        version=version,
        chunk=chunk,
        distance=float(row.get("distance", 0.0)),
    )


_SELECT_FIELDS = """
    c.id AS collection_id,
    c.name AS collection_name,
    c.description AS collection_description,
    d.id AS document_id,
    d.title AS document_title,
    d.source_type,
    d.metadata AS document_metadata,
    dv.id AS version_id,
    dv.version_no,
    dv.source_hash,
    dv.parser_name,
    dv.parser_version,
    dc.id AS chunk_id,
    dc.ordinal,
    dc.section_path,
    dc.page_number,
    dc.text,
    dc.context,
    dc.metadata AS chunk_metadata
"""


class V2ReadRepository:
    """V2 的只读 repository；连接由现有 PostgreSQL 连接层管理。"""

    def __init__(self, connection_factory: Callable = connect):
        self._connection_factory = connection_factory

    def vector_search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        scope: RetrievalScope | None = None,
    ) -> list[V2SearchHit]:
        if top_k <= 0:
            return []
        query_vector = vector_literal(query_embedding)
        conditions, scope_params = _scope_conditions(scope)
        sql = f"""
            SELECT {_SELECT_FIELDS}, dc.embedding <=> %s::vector AS distance
            FROM {V2_SCHEMA_NAME}.document_chunks dc
            JOIN {V2_SCHEMA_NAME}.document_versions dv
              ON dv.id = dc.document_version_id
            JOIN {V2_SCHEMA_NAME}.documents d ON d.id = dv.document_id
            JOIN {V2_SCHEMA_NAME}.collections c ON c.id = d.collection_id
            {_where(conditions)}
            ORDER BY dc.embedding <=> %s::vector, dc.id
            LIMIT %s
        """
        params = [query_vector, *scope_params, query_vector, top_k]
        with self._connection_factory() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_hit_from_row(row) for row in rows]

    def keyword_search(
        self,
        terms: Sequence[str],
        *,
        top_k: int = 5,
        scope: RetrievalScope | None = None,
        per_term_limit: int = DEFAULT_PER_TERM_LIMIT,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> list[V2SearchHit]:
        clean_terms = tuple(dict.fromkeys(term for term in terms if term))
        if not clean_terms or top_k <= 0 or per_term_limit <= 0:
            return []
        if not math.isfinite(k1) or not math.isfinite(b) or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 参数无效")

        values_sql = ", ".join("(%s)" for _ in clean_terms)
        corpus_conditions, corpus_params = _scope_conditions(scope)
        df_conditions, df_params = _scope_conditions(scope)
        candidate_conditions, candidate_params = _scope_conditions(scope)
        sql = f"""
            WITH q(term) AS (VALUES {values_sql}),
            corpus AS (
                SELECT COUNT(*)::float8 AS n,
                       NULLIF(AVG(dc.token_count), 0)::float8 AS avgdl
                FROM {V2_SCHEMA_NAME}.document_chunks dc
                JOIN {V2_SCHEMA_NAME}.document_versions dv
                  ON dv.id = dc.document_version_id
                JOIN {V2_SCHEMA_NAME}.documents d ON d.id = dv.document_id
                JOIN {V2_SCHEMA_NAME}.collections c ON c.id = d.collection_id
                {_where(corpus_conditions)}
            ),
            df AS (
                SELECT ct.term, COUNT(*)::float8 AS df
                FROM {V2_SCHEMA_NAME}.chunk_terms ct
                JOIN {V2_SCHEMA_NAME}.document_chunks dc ON dc.id = ct.chunk_id
                JOIN {V2_SCHEMA_NAME}.document_versions dv
                  ON dv.id = dc.document_version_id
                JOIN {V2_SCHEMA_NAME}.documents d ON d.id = dv.document_id
                JOIN {V2_SCHEMA_NAME}.collections c ON c.id = d.collection_id
                JOIN q ON q.term = ct.term
                {_where(df_conditions)}
                GROUP BY ct.term
            )
            SELECT {_SELECT_FIELDS}, l.tf, df.df, corpus.n, corpus.avgdl,
                   -(
                     LN((corpus.n - df.df + 0.5) / (df.df + 0.5) + 1)
                     * l.tf * ({k1} + 1)
                     / (l.tf + {k1} * (1 - {b} + {b} * dc.token_count / corpus.avgdl))
                   ) AS distance
            FROM q
            JOIN df ON df.term = q.term
            CROSS JOIN corpus
            CROSS JOIN LATERAL (
                SELECT ct.chunk_id, ct.tf
                FROM {V2_SCHEMA_NAME}.chunk_terms ct
                JOIN {V2_SCHEMA_NAME}.document_chunks dc ON dc.id = ct.chunk_id
                JOIN {V2_SCHEMA_NAME}.document_versions dv
                  ON dv.id = dc.document_version_id
                JOIN {V2_SCHEMA_NAME}.documents d ON d.id = dv.document_id
                JOIN {V2_SCHEMA_NAME}.collections c ON c.id = d.collection_id
                WHERE ct.term = q.term
                  {"AND " + candidate_conditions if candidate_conditions else ""}
                ORDER BY ct.tf DESC, ct.chunk_id
                LIMIT {int(per_term_limit)}
            ) l
            JOIN {V2_SCHEMA_NAME}.document_chunks dc ON dc.id = l.chunk_id
            JOIN {V2_SCHEMA_NAME}.document_versions dv
              ON dv.id = dc.document_version_id
            JOIN {V2_SCHEMA_NAME}.documents d ON d.id = dv.document_id
            JOIN {V2_SCHEMA_NAME}.collections c ON c.id = d.collection_id
            ORDER BY distance, dc.id
        """
        params = [
            *clean_terms,
            *corpus_params,
            *df_params,
            *candidate_params,
        ]
        with self._connection_factory() as conn:
            rows = conn.execute(sql, params).fetchall()

        ranked: dict[str, tuple[float, Mapping[str, object]]] = {}
        for row in rows:
            chunk_id = str(row["chunk_id"])
            score = float(row["distance"])
            previous = ranked.get(chunk_id)
            # 每个 term 的贡献已经包含在 distance 中；同一 chunk 的最终分数
            # 需要在 Python 端聚合，负值代表 BM25 分数越高。
            if previous is None:
                ranked[chunk_id] = (score, row)
            else:
                ranked[chunk_id] = (previous[0] + score, previous[1])
        selected = sorted(
            ranked.values(), key=lambda item: (item[0], str(item[1]["chunk_id"]))
        )[:top_k]
        return [_hit_from_row(row) for _score, row in selected]


class V2KnowledgeRetriever:
    """满足通用 ``KnowledgeRetriever`` 协议的向量检索入口。"""

    def __init__(self, embedder: Callable, repository: V2ReadRepository | None = None):
        self.embedder = embedder
        self.repository = repository or V2ReadRepository()

    def search(
        self,
        question: str,
        *,
        scope: RetrievalScope | None = None,
        top_k: int = 5,
    ) -> list[V2SearchHit]:
        encoded = self.embedder.encode([question], normalize_embeddings=True)
        return self.repository.vector_search(encoded[0], top_k=top_k, scope=scope)

    def retrieve(
        self,
        question: str,
        *,
        scope: RetrievalScope | None = None,
        top_k: int = 5,
    ) -> Sequence[DocumentChunk]:
        return [hit.chunk for hit in self.search(question, scope=scope, top_k=top_k)]

    def keyword_search(
        self,
        question: str,
        *,
        scope: RetrievalScope | None = None,
        top_k: int = 5,
    ) -> list[V2SearchHit]:
        return self.repository.keyword_search(query_terms(question), top_k=top_k, scope=scope)
