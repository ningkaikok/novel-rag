"""PostgreSQL/pgvector 连接与索引工具。"""
from collections.abc import Iterable

import psycopg
from psycopg.rows import dict_row

from config import DATABASE_URL


def connect() -> psycopg.Connection:
    """打开一个使用 dict row 的 PostgreSQL 连接。"""
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def vector_literal(values: Iterable[float]) -> str:
    """将 embedding 转成 pgvector 可解析的文本格式。"""
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def recreate_schema(dimension: int) -> None:
    """重建小说片段表和向量索引。重建索引本来就是全量操作，因此可安全清空。"""
    if dimension <= 0:
        raise ValueError("embedding dimension must be positive")
    with connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("DROP TABLE IF EXISTS novel_chunks")
        conn.execute(
            f"""
            CREATE TABLE novel_chunks (
                id BIGSERIAL PRIMARY KEY,
                novel TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding vector({dimension}) NOT NULL,
                UNIQUE (novel, chunk_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX novel_chunks_novel_chunk_idx "
            "ON novel_chunks (novel, chunk_id)"
        )
        conn.execute(
            "CREATE INDEX novel_chunks_embedding_hnsw_idx "
            "ON novel_chunks USING hnsw (embedding vector_cosine_ops)"
        )


def has_index() -> bool:
    """判断 PostgreSQL 中是否已经有可用的片段表。"""
    with connect() as conn:
        row = conn.execute(
            "SELECT to_regclass('public.novel_chunks') AS table_name"
        ).fetchone()
    return bool(row and row["table_name"])
