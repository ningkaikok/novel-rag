"""PostgreSQL/pgvector 连接与索引工具。

初学者可以把这里看成 RAG 的“持久化层”，主要表之间的关系是：

    novel_chunks         一行 = 一个原文片段 + 向量 + 章节等元数据
         │ (novel, chunk_id)
         ├── chunk_terms 一行 = 该片段中某个词的词频，供 BM25 使用
         └── character_relations 以 novel 为范围保存可选的人物关系边

    index_manifest       一行 = 一本书的文件哈希、流水线哈希和片段数
    hierarchy_summaries  一行 = 一个章节/全书导航摘要 + 原文片段范围 + 向量
    hierarchy_manifest   一行 = 一本书的层级算法指纹和摘要节点数

``novel_chunks`` 是最终回答可引用的事实来源；其余索引表都是可以从小说原文重新
计算出来的派生数据。两个 manifest 不是检索索引，而是两条增量流水线各自的
“检查点”：基础切分规则变化时重建片段，只有摘要规则变化时只补建层级节点。

本模块同时服务 FastAPI 和命令行脚本，所以 ``connect()`` 返回统一的上下文管理器：
Web 请求复用连接池，脚本则临时创建连接。业务代码不需要知道连接来自哪里。
"""
import json
from collections.abc import Callable, Iterable

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


def _ensure_index_schema(conn, dimension: int) -> None:
    """在一个现有事务里幂等创建索引相关表。"""
    if dimension <= 0:
        raise ValueError("embedding dimension must be positive")
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS novel_chunks (
            id BIGSERIAL PRIMARY KEY,
            novel TEXT NOT NULL,
            chunk_id INTEGER NOT NULL,
            chapter_title TEXT,
            text TEXT NOT NULL,
            embedding vector({dimension}) NOT NULL,
            token_count INTEGER NOT NULL DEFAULT 0,
            context TEXT NOT NULL DEFAULT '',
            UNIQUE (novel, chunk_id)
        )
        """
    )
    # 兼容 M1 之前已经存在的表。其余列在最早的 PostgreSQL schema 中就存在。
    conn.execute(
        "ALTER TABLE novel_chunks ADD COLUMN IF NOT EXISTS chapter_title TEXT"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_terms (
            novel    TEXT    NOT NULL,
            chunk_id INTEGER NOT NULL,
            term     TEXT    NOT NULL,
            tf       INTEGER NOT NULL,
            PRIMARY KEY (novel, chunk_id, term)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS character_relations (
            novel      TEXT    NOT NULL,
            person_a   TEXT    NOT NULL,
            person_b   TEXT    NOT NULL,
            relation   TEXT    NOT NULL,
            weight     INTEGER NOT NULL,
            PRIMARY KEY (novel, person_a, person_b, relation)
        )
        """
    )
    # 文件清单不和某次任务绑定。只有一本书的两套索引在同一事务里完整写入后，
    # 才更新这里的哈希；任务失败或取消时事务回滚，下次会安全地再次识别为待处理。
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS index_manifest (
            novel         TEXT PRIMARY KEY,
            source_hash   TEXT NOT NULL,
            pipeline_hash TEXT NOT NULL,
            chunk_count   INTEGER NOT NULL,
            quality_report JSONB NOT NULL DEFAULT '{}'::jsonb,
            indexed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    conn.execute(
        "ALTER TABLE index_manifest ADD COLUMN IF NOT EXISTS quality_report JSONB "
        "NOT NULL DEFAULT '{}'::jsonb"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS chunk_terms_term_idx ON chunk_terms (term)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS character_relations_a_idx "
        "ON character_relations (person_a, relation)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS character_relations_b_idx "
        "ON character_relations (person_b, relation)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS novel_chunks_novel_chunk_idx "
        "ON novel_chunks (novel, chunk_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS novel_chunks_embedding_hnsw_idx "
        "ON novel_chunks USING hnsw (embedding vector_cosine_ops)"
    )
    _ensure_hierarchy_schema(conn, dimension)


def _ensure_hierarchy_schema(conn, dimension: int) -> None:
    """幂等准备章节/全书摘要索引；与片段表共用同一个 embedding 维度。"""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS hierarchy_summaries (
            novel          TEXT NOT NULL,
            level          TEXT NOT NULL CHECK (level IN ('chapter', 'novel')),
            node_id        TEXT NOT NULL,
            title          TEXT NOT NULL,
            node_order     INTEGER NOT NULL,
            start_chunk_id INTEGER NOT NULL,
            end_chunk_id   INTEGER NOT NULL,
            summary        TEXT NOT NULL,
            embedding      vector({dimension}) NOT NULL,
            PRIMARY KEY (novel, level, node_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hierarchy_manifest (
            novel         TEXT PRIMARY KEY,
            source_hash   TEXT NOT NULL,
            pipeline_hash TEXT NOT NULL,
            node_count    INTEGER NOT NULL,
            indexed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS hierarchy_summaries_embedding_hnsw_idx "
        "ON hierarchy_summaries USING hnsw (embedding vector_cosine_ops)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS hierarchy_summaries_scope_idx "
        "ON hierarchy_summaries (level, novel, node_order)"
    )


def ensure_index_schema(dimension: int) -> None:
    """幂等准备可增量更新的索引表，不删除任何现有小说。"""
    with connect() as conn:
        _ensure_index_schema(conn, dimension)


def recreate_schema(dimension: int) -> None:
    """兼容旧脚本的全量清空入口；Web 增量任务不再调用它。"""
    with connect() as conn:
        conn.execute("DROP TABLE IF EXISTS hierarchy_summaries")
        conn.execute("DROP TABLE IF EXISTS hierarchy_manifest")
        conn.execute("DROP TABLE IF EXISTS chunk_terms")
        conn.execute("DROP TABLE IF EXISTS novel_chunks")
        conn.execute("DROP TABLE IF EXISTS character_relations")
        conn.execute("DROP TABLE IF EXISTS index_manifest")
        _ensure_index_schema(conn, dimension)


def load_index_manifest() -> dict[str, dict]:
    """返回数据库记录的文件哈希清单；首次升级、表不存在时返回空。"""
    with connect() as conn:
        table = conn.execute(
            "SELECT to_regclass('public.index_manifest') AS table_name"
        ).fetchone()
        if not table or not table["table_name"]:
            return {}
        column = conn.execute(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'index_manifest'
              AND column_name = 'quality_report'
            """
        ).fetchone()
        fields = "novel, source_hash, pipeline_hash, chunk_count"
        if column:
            fields += ", quality_report"
        rows = conn.execute(f"SELECT {fields} FROM index_manifest").fetchall()
    return {row["novel"]: dict(row) for row in rows}


def load_hierarchy_manifest() -> dict[str, dict]:
    """返回层级摘要清单；旧数据库尚未建表时返回空，供无损补建使用。"""
    with connect() as conn:
        table = conn.execute(
            "SELECT to_regclass('public.hierarchy_manifest') AS table_name"
        ).fetchone()
        if not table or not table["table_name"]:
            return {}
        rows = conn.execute(
            "SELECT novel, source_hash, pipeline_hash, node_count "
            "FROM hierarchy_manifest"
        ).fetchall()
    return {row["novel"]: dict(row) for row in rows}


def indexed_novels() -> set[str]:
    """列出片段表中实际存在的书，兼容尚未建立 manifest 的旧索引。"""
    with connect() as conn:
        table = conn.execute(
            "SELECT to_regclass('public.novel_chunks') AS table_name"
        ).fetchone()
        if not table or not table["table_name"]:
            return set()
        rows = conn.execute("SELECT DISTINCT novel FROM novel_chunks").fetchall()
    return {row["novel"] for row in rows}


def replace_novel_index(
    novel: str,
    rows: list[tuple],
    per_chunk_terms: list[dict[str, int]],
    source_hash: str,
    pipeline_hash: str,
    relations: list[tuple[str, str, str, str, int]] | None = None,
    cancel_check: Callable[[], None] | None = None,
    hierarchy_rows: list[tuple] | None = None,
    hierarchy_hash: str | None = None,
    quality_report: dict | None = None,
) -> None:
    """在一个事务里原子替换单本书的全部派生数据。

    embedding 和分词在进入这里之前已经准备好。删除旧数据、写向量、COPY BM25、
    写关系边和更新 manifest 共享同一事务；任何异常（包括用户取消）都会回滚，
    因此检索永远只会看到旧版或完整新版，不会看到“向量已换、BM25 只写一半”。
    """
    if len(rows) != len(per_chunk_terms):
        raise ValueError("rows and per_chunk_terms must have the same length")
    check = cancel_check or (lambda: None)
    # psycopg 的连接上下文管理器就是这里的事务边界：正常离开自动 COMMIT，
    # 中途任何 SQL、COPY 或 check() 抛异常都会自动 ROLLBACK。不要在循环里手动
    # commit，否则“用户取消”可能只回滚后半段，留下向量和 BM25 版本不一致。
    with connect() as conn:
        check()
        conn.execute("DELETE FROM chunk_terms WHERE novel = %s", (novel,))
        conn.execute("DELETE FROM novel_chunks WHERE novel = %s", (novel,))
        conn.execute("DELETE FROM character_relations WHERE novel = %s", (novel,))
        if hierarchy_rows is not None:
            conn.execute("DELETE FROM hierarchy_summaries WHERE novel = %s", (novel,))
            conn.execute("DELETE FROM hierarchy_manifest WHERE novel = %s", (novel,))
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO novel_chunks "
                "(novel, chunk_id, chapter_title, text, embedding, token_count, context) "
                "VALUES (%s, %s, %s, %s, %s::vector, %s, %s)",
                rows,
            )
        check()
        with conn.cursor() as cursor:
            # BM25 的“一个片段 × 多个词”会产生大量行，COPY 比逐条 INSERT 快得多。
            # COPY 仍属于外层同一个事务，并不会削弱原子性。
            with cursor.copy(
                "COPY chunk_terms (novel, chunk_id, term, tf) FROM STDIN"
            ) as copy:
                for index, (row, freqs) in enumerate(zip(rows, per_chunk_terms)):
                    if index % 50 == 0:
                        check()
                    chunk_id = row[1]
                    for term, tf in freqs.items():
                        copy.write_row((novel, chunk_id, term, tf))
        if relations:
            with conn.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO character_relations "
                    "(novel, person_a, person_b, relation, weight) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    relations,
                )
        if hierarchy_rows is not None:
            with conn.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO hierarchy_summaries "
                    "(novel, level, node_id, title, node_order, start_chunk_id, "
                    "end_chunk_id, summary, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)",
                    hierarchy_rows,
                )
            conn.execute(
                """
                INSERT INTO hierarchy_manifest
                    (novel, source_hash, pipeline_hash, node_count, indexed_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (novel) DO UPDATE SET
                    source_hash = EXCLUDED.source_hash,
                    pipeline_hash = EXCLUDED.pipeline_hash,
                    node_count = EXCLUDED.node_count,
                    indexed_at = NOW()
                """,
                (novel, source_hash, hierarchy_hash or "", len(hierarchy_rows)),
            )
        check()
        conn.execute(
            """
            INSERT INTO index_manifest
                (novel, source_hash, pipeline_hash, chunk_count, quality_report, indexed_at)
            VALUES (%s, %s, %s, %s, %s::jsonb, NOW())
            ON CONFLICT (novel) DO UPDATE SET
                source_hash = EXCLUDED.source_hash,
                pipeline_hash = EXCLUDED.pipeline_hash,
                chunk_count = EXCLUDED.chunk_count,
                quality_report = EXCLUDED.quality_report,
                indexed_at = NOW()
            """,
            (novel, source_hash, pipeline_hash, len(rows), json.dumps(quality_report or {}, ensure_ascii=False)),
        )


def delete_novel_index(
    novel: str, cancel_check: Callable[[], None] | None = None
) -> None:
    """在一个短事务里删除一本已不存在的书及其清单记录。"""
    check = cancel_check or (lambda: None)
    with connect() as conn:
        check()
        conn.execute("DELETE FROM chunk_terms WHERE novel = %s", (novel,))
        conn.execute("DELETE FROM novel_chunks WHERE novel = %s", (novel,))
        conn.execute("DELETE FROM character_relations WHERE novel = %s", (novel,))
        conn.execute("DELETE FROM index_manifest WHERE novel = %s", (novel,))
        conn.execute("DELETE FROM hierarchy_summaries WHERE novel = %s", (novel,))
        conn.execute("DELETE FROM hierarchy_manifest WHERE novel = %s", (novel,))


def replace_novel_hierarchy(
    novel: str,
    rows: list[tuple],
    source_hash: str,
    pipeline_hash: str,
    cancel_check: Callable[[], None] | None = None,
) -> None:
    """只补建一本旧书的层级摘要，不触碰已经可用的片段向量/BM25。"""
    check = cancel_check or (lambda: None)
    with connect() as conn:
        check()
        conn.execute("DELETE FROM hierarchy_summaries WHERE novel = %s", (novel,))
        conn.execute("DELETE FROM hierarchy_manifest WHERE novel = %s", (novel,))
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO hierarchy_summaries "
                "(novel, level, node_id, title, node_order, start_chunk_id, "
                "end_chunk_id, summary, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)",
                rows,
            )
        check()
        conn.execute(
            """
            INSERT INTO hierarchy_manifest
                (novel, source_hash, pipeline_hash, node_count, indexed_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (novel) DO UPDATE SET
                source_hash = EXCLUDED.source_hash,
                pipeline_hash = EXCLUDED.pipeline_hash,
                node_count = EXCLUDED.node_count,
                indexed_at = NOW()
            """,
            (novel, source_hash, pipeline_hash, len(rows)),
        )


def search_hierarchy(
    query_vector: str,
    *,
    level: str,
    limit: int,
    novels: list[str] | None = None,
) -> list[dict]:
    """按摘要向量搜索章节或全书节点；返回范围用于映射回原文。"""
    if level not in {"chapter", "novel"}:
        raise ValueError("level must be chapter or novel")
    scope = "AND novel = ANY(%s)" if novels else ""
    params: list = [query_vector, level]
    if novels:
        params.append(novels)
    params.extend([query_vector, limit])
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT novel, level, node_id, title, node_order,
                   start_chunk_id, end_chunk_id, summary,
                   embedding <=> %s::vector AS distance
            FROM hierarchy_summaries
            WHERE level = %s {scope}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def index_chunk_count() -> int:
    """返回当前可检索片段总数。"""
    if not has_index():
        return 0
    with connect() as conn:
        return int(
            conn.execute("SELECT count(*) AS count FROM novel_chunks").fetchone()[
                "count"
            ]
        )


def hierarchy_node_count() -> int:
    """返回当前章节 + 全书摘要节点总数；尚未迁移时为 0。"""
    with connect() as conn:
        table = conn.execute(
            "SELECT to_regclass('public.hierarchy_summaries') AS table_name"
        ).fetchone()
        if not table or not table["table_name"]:
            return 0
        row = conn.execute("SELECT count(*) AS count FROM hierarchy_summaries").fetchone()
    return int(row["count"])


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

    **刻意和可重算的索引表分开**，理由和 chat_turns 一样但更关键：
    生成这些上下文说明要调 LLM，实测单条约 4.4 秒。如果跟着向量索引一起
    被索引替换事务删除，用户修改一本书就要重跑几十分钟到几小时。
    分开存之后，增量同步时能按内容哈希直接复用已有结果。

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

    刻意和小说派生索引分开：单书替换或清理不能影响对话历史——
    重新整理书架不该抹掉用户的聊天记录。

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
                -- Agent Lab 那条独立链路的步骤记录，形状和 trace 不一样（tool/
                -- reason/observation，不是 step/detail），所以单开一列，不往
                -- trace 里塞两种形状——那样 StoredTurn 的类型就没法同时对两边诚实。
                agent_steps JSONB,
                -- complete：正常生成完；interrupted：用户中断，content 是部分内容
                status      TEXT    NOT NULL DEFAULT 'complete',
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (session_id, turn_index)
            )
            """
        )
        conn.execute(
            "ALTER TABLE chat_turns ADD COLUMN IF NOT EXISTS agent_steps JSONB"
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
    agent_steps: list | None = None,
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
                (session_id, turn_index, role, content, sources, trace, agent_steps, status)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s)
            ON CONFLICT (session_id, turn_index) DO UPDATE SET
                role        = EXCLUDED.role,
                content     = EXCLUDED.content,
                sources     = EXCLUDED.sources,
                trace       = EXCLUDED.trace,
                agent_steps = EXCLUDED.agent_steps,
                status      = EXCLUDED.status
            """,
            (
                session_id,
                turn_index,
                role,
                content,
                json.dumps(sources, ensure_ascii=False) if sources is not None else None,
                json.dumps(trace, ensure_ascii=False) if trace is not None else None,
                json.dumps(agent_steps, ensure_ascii=False) if agent_steps is not None else None,
                status,
            ),
        )


def load_turns(session_id: str) -> list[dict]:
    """按顺序读回某个会话的全部对话轮次。"""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT turn_index, role, content, sources, trace, agent_steps, status
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
