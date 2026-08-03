"""PostgreSQL/pgvector 连接与索引工具。"""
import json
from collections.abc import Iterable

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from config import DATABASE_URL, DB_POOL_MAX_SIZE, DB_POOL_MIN_SIZE

# 进程级共享连接池，由 FastAPI 的 lifespan 在启动时调用 init_pool() 初始化。
# 独立脚本（ingest.py、tests/run_qa_tests.py 等）不会调用 init_pool()，
# 此时保持 None，connect() 退化为每次新建一个连接——这些脚本本来就是
# 一次性跑完就退出，不需要池化。
_pool: ConnectionPool | None = None


def init_pool(min_size: int = DB_POOL_MIN_SIZE, max_size: int = DB_POOL_MAX_SIZE) -> None:
    """建一个共享连接池。之前每次 connect() 都是全新握手——单次问答请求
    在检索/会话持久化路径上可能连续开好几个连接，本机 loopback 下感觉不到，
    但完全没考虑并发扩展。池化之后同一进程内的请求复用已建立的连接。

    open=True（psycopg_pool 默认行为）：调用时就建好 min_size 个连接并
    等待就绪，启动时就能发现数据库连不上，而不是等第一次真实请求才报错。
    """
    global _pool
    pool = ConnectionPool(
        DATABASE_URL,
        min_size=min_size,
        max_size=max_size,
        kwargs={"row_factory": dict_row},
    )
    try:
        pool.wait(timeout=10)
    except Exception:
        # 数据库暂时连不上：关掉这个还在后台重试的池子，不要泄漏，
        # 让调用方（FastAPI lifespan）能优雅降级——_pool 保持 None，
        # connect() 退化为逐次新建连接，其余功能（比如书架管理）不受影响。
        pool.close()
        raise
    _pool = pool


def close_pool() -> None:
    """FastAPI 关闭时调用，释放池子里所有连接。"""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def connect():
    """返回一个「连接」的上下文管理器：`with connect() as conn:` 用法不变。

    已经调用过 init_pool() 时从池子里借用/归还一个连接（进程生命周期内
    复用，省去重复握手）；没有池子（独立脚本、测试、pytest）时退化为
    每次新建一个连接，行为和以前完全一样——13 处调用点都不用改。
    """
    if _pool is not None:
        return _pool.connection()
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


def ensure_chat_schema() -> None:
    """建对话历史表（幂等，用 IF NOT EXISTS）。

    刻意和 recreate_schema 分开：那个函数重建向量索引时会 DROP TABLE，
    对话历史不能跟着被清空——重新整理书架不该抹掉用户的聊天记录。

    (session_id, turn_index) 做主键，既是天然去重，也是幂等写入的基础：
    中断时存"已生成的部分"这个写操作可能被重复触发（用户连点 Stop、网络抖动），
    UPSERT 到同一主键只会覆盖，不会插出重复行或报冲突。
    """
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_turns (
                session_id  UUID    NOT NULL,
                turn_index  INTEGER NOT NULL,
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL DEFAULT '',
                sources     JSONB,
                trace       JSONB,
                -- complete：正常生成完；interrupted：用户中断，content 是部分内容
                status      TEXT    NOT NULL DEFAULT 'complete',
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (session_id, turn_index)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS chat_turns_session_idx "
            "ON chat_turns (session_id, turn_index)"
        )


def save_turn(
    session_id: str,
    turn_index: int,
    role: str,
    content: str,
    sources: list | None = None,
    trace: list | None = None,
    status: str = "complete",
) -> None:
    """写入或覆盖一轮对话（幂等）。

    用 ON CONFLICT DO UPDATE 而非 INSERT：中断保存可能被重复调用，
    同一 (session_id, turn_index) 直接覆盖，不会主键冲突。
    """
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO chat_turns
                (session_id, turn_index, role, content, sources, trace, status)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
            ON CONFLICT (session_id, turn_index) DO UPDATE SET
                role    = EXCLUDED.role,
                content = EXCLUDED.content,
                sources = EXCLUDED.sources,
                trace   = EXCLUDED.trace,
                status  = EXCLUDED.status
            """,
            (
                session_id,
                turn_index,
                role,
                content,
                json.dumps(sources, ensure_ascii=False) if sources is not None else None,
                json.dumps(trace, ensure_ascii=False) if trace is not None else None,
                status,
            ),
        )


def load_turns(session_id: str) -> list[dict]:
    """按顺序读回某个会话的全部对话轮次。"""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT turn_index, role, content, sources, trace, status
            FROM chat_turns
            WHERE session_id = %s
            ORDER BY turn_index
            """,
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def next_turn_index(session_id: str) -> int:
    """该会话下一轮的序号；空会话返回 0。"""
    with connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(turn_index) + 1, 0) AS next FROM chat_turns "
            "WHERE session_id = %s",
            (session_id,),
        ).fetchone()
    return int(row["next"]) if row else 0
