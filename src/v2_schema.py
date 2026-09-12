"""V2 通用知识库 schema 定义。

V2 使用独立的 ``knowledge_v2`` PostgreSQL schema，避免与当前 public 下的
``novel_chunks``、``chunk_terms`` 和 ``index_manifest`` 发生名称或读写冲突。
本模块只提供 DDL 和显式 apply 函数；不会在 import、FastAPI 启动或 V1 索引流程中
自动执行任何迁移。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

V2_SCHEMA_NAME = "knowledge_v2"
V2_SCHEMA_VERSION = "1"


class SQLExecutor(Protocol):
    """当前 psycopg connection/cursor 所需的最小接口，便于 mock 测试。"""

    def execute(self, query: str) -> object: ...


def _validate_dimension(dimension: int) -> int:
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise ValueError("embedding dimension must be a positive integer")
    return dimension


def v2_schema_statements(dimension: int) -> tuple[str, ...]:
    """返回可重复执行的 V2 DDL；只包含 ``IF NOT EXISTS`` 的非破坏性操作。"""

    dimension = _validate_dimension(dimension)
    return (
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"CREATE SCHEMA IF NOT EXISTS {V2_SCHEMA_NAME}",
        f"""
        CREATE TABLE IF NOT EXISTS {V2_SCHEMA_NAME}.collections (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {V2_SCHEMA_NAME}.documents (
            id            TEXT PRIMARY KEY,
            collection_id TEXT NOT NULL REFERENCES {V2_SCHEMA_NAME}.collections(id),
            title         TEXT NOT NULL,
            source_type   TEXT NOT NULL,
            metadata      JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {V2_SCHEMA_NAME}.document_versions (
            id             TEXT PRIMARY KEY,
            document_id    TEXT NOT NULL REFERENCES {V2_SCHEMA_NAME}.documents(id),
            version_no     INTEGER NOT NULL CHECK (version_no >= 1),
            source_hash    TEXT NOT NULL,
            parser_name    TEXT NOT NULL,
            parser_version TEXT NOT NULL,
            pipeline_hash  TEXT,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (document_id, version_no),
            UNIQUE (document_id, source_hash)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {V2_SCHEMA_NAME}.document_chunks (
            id                   TEXT PRIMARY KEY,
            document_version_id  TEXT NOT NULL
                REFERENCES {V2_SCHEMA_NAME}.document_versions(id),
            ordinal              INTEGER NOT NULL CHECK (ordinal >= 0),
            section_path         JSONB NOT NULL DEFAULT '[]'::jsonb,
            page_number          INTEGER CHECK (page_number IS NULL OR page_number >= 1),
            text                 TEXT NOT NULL,
            context              TEXT NOT NULL DEFAULT '',
            token_count          INTEGER NOT NULL DEFAULT 0 CHECK (token_count >= 0),
            embedding            vector({dimension}) NOT NULL,
            metadata             JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            UNIQUE (document_version_id, ordinal)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {V2_SCHEMA_NAME}.chunk_terms (
            chunk_id TEXT NOT NULL REFERENCES {V2_SCHEMA_NAME}.document_chunks(id),
            term     TEXT NOT NULL,
            tf       INTEGER NOT NULL CHECK (tf > 0),
            PRIMARY KEY (chunk_id, term)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {V2_SCHEMA_NAME}.index_manifests (
            document_version_id TEXT PRIMARY KEY
                REFERENCES {V2_SCHEMA_NAME}.document_versions(id),
            source_hash         TEXT NOT NULL,
            pipeline_hash       TEXT NOT NULL,
            chunk_count         INTEGER NOT NULL CHECK (chunk_count >= 0),
            quality_report      JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            indexed_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        f"CREATE INDEX IF NOT EXISTS knowledge_v2_documents_collection_idx ON {V2_SCHEMA_NAME}.documents (collection_id)",
        f"CREATE INDEX IF NOT EXISTS knowledge_v2_document_versions_document_idx ON {V2_SCHEMA_NAME}.document_versions (document_id, version_no DESC)",
        f"CREATE INDEX IF NOT EXISTS knowledge_v2_document_chunks_version_ordinal_idx ON {V2_SCHEMA_NAME}.document_chunks (document_version_id, ordinal)",
        f"CREATE INDEX IF NOT EXISTS knowledge_v2_document_chunks_embedding_hnsw_idx ON {V2_SCHEMA_NAME}.document_chunks USING hnsw (embedding vector_cosine_ops)",
        f"CREATE INDEX IF NOT EXISTS knowledge_v2_chunk_terms_term_idx ON {V2_SCHEMA_NAME}.chunk_terms (term)",
        f"CREATE INDEX IF NOT EXISTS knowledge_v2_chunk_terms_term_tf_idx ON {V2_SCHEMA_NAME}.chunk_terms (term, tf DESC, chunk_id)",
    )


def render_v2_schema_sql(dimension: int) -> str:
    """渲染供审阅/迁移工具输出的 DDL，不连接数据库。"""

    return "\n\n".join(
        f"{statement.strip()};" for statement in v2_schema_statements(dimension)
    )


def apply_v2_schema(conn: SQLExecutor, dimension: int) -> int:
    """显式执行 V2 DDL 并返回语句数；调用方必须自行控制事务。

    该函数不会迁移数据、切换 ``STORAGE_SCHEMA`` 或删除任何 V1 表。当前 CLI 和应用
    启动流程都不会自动调用它。
    """

    statements: Sequence[str] = v2_schema_statements(dimension)
    for statement in statements:
        conn.execute(statement)
    return len(statements)
