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
    """重建小说片段表、向量索引和 BM25 倒排索引。

    这三样都是从同一批原文派生出来的，必须一起重建——只重建其中一个会导致
    向量索引和倒排索引对应的是不同版本的文本，检索结果互相矛盾。
    重建索引本来就是全量操作，因此可安全清空。
    """
    if dimension <= 0:
        raise ValueError("embedding dimension must be positive")
    with connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # chunk_terms 有外键概念上依赖 novel_chunks，先删它
        conn.execute("DROP TABLE IF EXISTS chunk_terms")
        conn.execute("DROP TABLE IF EXISTS novel_chunks")
        conn.execute(
            f"""
            CREATE TABLE novel_chunks (
                id BIGSERIAL PRIMARY KEY,
                novel TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                -- 从 txt 章节标题行识别出的归属。旧索引或无规范标题的文本为 NULL。
                -- 单独存而不是拼进 novel/text，方便界面展示和后续章节级聚合。
                chapter_title TEXT,
                text TEXT NOT NULL,
                embedding vector({dimension}) NOT NULL,
                -- 这个片段有多少个词（分词后）。BM25 的「文档长度归一化」要用：
                -- 长片段天然更容易碰巧包含查询词，不做归一化的话长片段会系统性
                -- 地占便宜。
                token_count INTEGER NOT NULL DEFAULT 0,
                -- Contextual Retrieval 生成的上下文说明（没做增强的片段是空串）。
                -- **必须和 text 分开存**：索引和重排要用「说明 + 原文」，
                -- 但送给大模型生成时只用原文，不把 AI 生成的说明混进正文当依据。
                -- 踩过的坑：一开始只把说明拼进索引、没有单独存，结果重排拿
                -- text 列的原文重新打分，看不到说明——同一个片段，重排给原文
                -- 打 0.0055、给「说明+原文」打 0.9990，把上下文增强的效果
                -- 整个抵消掉了（Q18 因此从第 5 名掉出 top-20）。
                context TEXT NOT NULL DEFAULT '',
                UNIQUE (novel, chunk_id)
            )
            """
        )
        # BM25 倒排索引：一行 = 「某个词」在「某个片段」里出现了几次。
        # 全库约 390 万行（3.3 万个片段 × 每个约 118 个不重复的词）。
        conn.execute(
            """
            CREATE TABLE chunk_terms (
                novel    TEXT    NOT NULL,
                chunk_id INTEGER NOT NULL,
                term     TEXT    NOT NULL,
                -- term frequency：这个词在这个片段里出现几次。
                -- 出现多次说明这个片段更可能真的在讲这个词，是 BM25 的核心信号。
                tf       INTEGER NOT NULL,
                PRIMARY KEY (novel, chunk_id, term)
            )
            """
        )
        # 查询时是「给定几个词，找出所有含这些词的片段」，所以按 term 建索引。
        # 没有这个索引，每次查询都要全表扫 390 万行。
        conn.execute("CREATE INDEX chunk_terms_term_idx ON chunk_terms (term)")

        # GraphRAG 的人物关系图（边表）。和上面两个索引一样是从同一批原文
        # 派生的，所以跟着一起重建。
        #
        # **但抽人名的结果不在这里**——那部分要调 LLM，单独缓存在
        # graph_characters 表里（见 ensure_graph_cache）。这样重建边表很快，
        # 而昂贵的人名抽取能复用。
        conn.execute("DROP TABLE IF EXISTS character_relations")
        conn.execute(
            """
            CREATE TABLE character_relations (
                novel      TEXT    NOT NULL,
                -- 两个人物名，已按字典序排好：关系是无向的，排序后同一对人
                -- 不会因为出现顺序不同被记成两条边
                person_a   TEXT    NOT NULL,
                person_b   TEXT    NOT NULL,
                relation   TEXT    NOT NULL,
                -- 共现次数。这是**唯一的置信度信号**——共现越多越可能是真关系，
                -- 只出现一两次的大概率是偶然同框。查询时靠它排序和过滤。
                weight     INTEGER NOT NULL,
                PRIMARY KEY (novel, person_a, person_b, relation)
            )
            """
        )
        # 查询模式是「给定一个人名和关系类型，找出所有相关的人」，
        # 而人名可能出现在 a 或 b 任一侧，所以两侧都要能走索引。
        conn.execute(
            "CREATE INDEX character_relations_a_idx "
            "ON character_relations (person_a, relation)"
        )
        conn.execute(
            "CREATE INDEX character_relations_b_idx "
            "ON character_relations (person_b, relation)"
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


def ensure_novel_metadata_schema() -> None:
    """为旧索引幂等补上新增的片段元数据列。

    这让升级代码后不必立刻重建索引也能继续问答；旧行的章节名暂时为 NULL。
    想看到章节标题仍需重建一次，因为数据库无法凭旧片段可靠反推出章节边界。
    """
    with connect() as conn:
        table = conn.execute(
            "SELECT to_regclass('public.novel_chunks') AS table_name"
        ).fetchone()
        if table and table["table_name"]:
            conn.execute(
                "ALTER TABLE novel_chunks "
                "ADD COLUMN IF NOT EXISTS chapter_title TEXT"
            )


def save_relations(edges: list[tuple[str, str, str, str, int]]) -> None:
    """批量写入人物关系边。表在 recreate_schema 里已经建好且清空。"""
    if not edges:
        return
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO character_relations "
                "(novel, person_a, person_b, relation, weight) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (novel, person_a, person_b, relation) "
                "DO UPDATE SET weight = EXCLUDED.weight",
                edges,
            )


def query_relations(
    person: str, relation: str, limit: int = 8, min_weight: int = 2
) -> list[tuple[str, int]]:
    """查某个人物在某种关系下的所有对手方，按共现次数从高到低。

    min_weight 默认 2：只共现过一次的边大概率是偶然同框（两个人碰巧出现在
    同一段提到「师父」的话里），过滤掉能显著降噪。

    人名可能存在于 person_a 或 person_b 任一侧（边是无向的、按字典序存的），
    所以两侧都要查，用 UNION ALL 合并。
    """
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT other, SUM(weight) AS weight FROM (
                SELECT person_b AS other, weight FROM character_relations
                WHERE person_a = %s AND relation = %s
                UNION ALL
                SELECT person_a AS other, weight FROM character_relations
                WHERE person_b = %s AND relation = %s
            ) AS both_sides
            GROUP BY other
            HAVING SUM(weight) >= %s
            ORDER BY weight DESC
            LIMIT %s
            """,
            (person, relation, person, relation, min_weight, limit),
        ).fetchall()
    return [(r["other"], int(r["weight"])) for r in rows]


def known_characters(novel: str | None = None) -> list[str]:
    """图里出现过的所有人物名（用于在问题里识别"用户问的是谁"）。"""
    scope = "WHERE novel = %s" if novel else ""
    params = (novel,) if novel else ()
    with connect() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT person_a AS name FROM character_relations {scope} "
            f"UNION SELECT DISTINCT person_b FROM character_relations {scope}",
            params + params,
        ).fetchall()
    return [r["name"] for r in rows]


def ensure_graph_cache() -> None:
    """建人物名抽取的缓存表（幂等，用 IF NOT EXISTS）。

    **和 character_relations 分开**，理由和 chunk_contexts 一样：
    抽人名要调 LLM（实测《凡人修仙传》的「伴侣」关系就要 11 次调用、57 秒），
    而 character_relations 跟着 recreate_schema 一起被 DROP。如果不单独缓存，
    **加一本新书就要把所有书的图重抽一遍**。

    缓存键是 (书名, 关系类型, 采样片段的内容哈希)：
    - 加新书 → 新书的键查不到，只抽新书；老书直接复用
    - 改切分参数 → 采样内容变了，哈希变了，会重抽（这是对的，内容确实变了）
    - 什么都没改 → 全部命中缓存，零调用
    """
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_characters (
                novel        TEXT NOT NULL,
                relation     TEXT NOT NULL,
                sample_hash  TEXT NOT NULL,
                -- 抽到的人名列表（JSON 数组）。存整个列表而不是一行一个名字：
                -- 它是「一次抽取的完整结果」，拆开存反而要额外判断是不是抽全了。
                names        JSONB NOT NULL,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (novel, relation, sample_hash)
            )
            """
        )


def load_cached_graph_characters(
    novel: str, relation: str, sample_hash: str
) -> list[str] | None:
    """读回缓存的人名列表；没缓存返回 None（注意和"缓存了空列表"区分开）。"""
    with connect() as conn:
        row = conn.execute(
            "SELECT names FROM graph_characters "
            "WHERE novel = %s AND relation = %s AND sample_hash = %s",
            (novel, relation, sample_hash),
        ).fetchone()
    return list(row["names"]) if row else None


def save_graph_characters(
    novel: str, relation: str, sample_hash: str, names: list[str]
) -> None:
    """写入抽取结果（幂等）。空列表也会写——"这批确实没抽到人名"本身
    就是有效结果，缓存下来能避免下次重复调用。
    """
    with connect() as conn:
        conn.execute(
            "INSERT INTO graph_characters (novel, relation, sample_hash, names) "
            "VALUES (%s, %s, %s, %s::jsonb) "
            "ON CONFLICT (novel, relation, sample_hash) "
            "DO UPDATE SET names = EXCLUDED.names",
            (novel, relation, sample_hash, json.dumps(names, ensure_ascii=False)),
        )


def ensure_context_cache() -> None:
    """建 Contextual Retrieval 的上下文缓存表（幂等）。

    **刻意和 recreate_schema 分开**，理由和 chat_turns 一样但更关键：
    生成这些上下文说明要调 LLM，实测单条约 4.4 秒。如果跟着向量索引一起
    被 DROP，用户在界面上点一次「重新整理书架」就要重跑几十分钟到几小时。
    分开存之后，重建索引时能按内容哈希直接复用已有结果。

    主键用**片段原文的哈希**而不是 (书名, chunk_id)：切分参数一变，同一个
    chunk_id 对应的文本就变了，用位置做键会取到过期的上下文。用内容哈希则
    天然正确——文本没变就复用，变了自然查不到、会重新生成。
    """
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_contexts (
                text_hash  TEXT PRIMARY KEY,
                context    TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )


def load_cached_contexts(hashes: list[str]) -> dict[str, str]:
    """批量读回已缓存的上下文说明，返回 {哈希: 说明}。"""
    if not hashes:
        return {}
    with connect() as conn:
        rows = conn.execute(
            "SELECT text_hash, context FROM chunk_contexts WHERE text_hash = ANY(%s)",
            (hashes,),
        ).fetchall()
    return {row["text_hash"]: row["context"] for row in rows}


def save_contexts(items: list[tuple[str, str]]) -> None:
    """批量写入上下文说明（幂等，重复写同一个哈希只会覆盖）。

    items 是 (哈希, 说明) 列表。空说明（生成失败）不写入——留着下次重试，
    而不是把失败结果也缓存起来。
    """
    rows = [(h, c) for h, c in items if c]
    if not rows:
        return
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO chunk_contexts (text_hash, context) VALUES (%s, %s) "
                "ON CONFLICT (text_hash) DO UPDATE SET context = EXCLUDED.context",
                rows,
            )


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
