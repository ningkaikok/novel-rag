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
from contextlib import AbstractContextManager

import psycopg
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from config import (
    DATABASE_URL,
    DB_POOL_MAX_SIZE,
    DB_POOL_MIN_SIZE,
    GRAPH_MIN_CONFIDENCE,
    GRAPH_REQUIRE_EXPLICIT,
)

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


def connect() -> AbstractContextManager[psycopg.Connection[DictRow]]:
    """返回一个「连接」的上下文管理器：`with connect() as conn:` 用法不变。

    已经调用过 init_pool() 时从池子里借用/归还一个连接（进程生命周期内
    复用，省去重复握手）；没有池子（独立脚本、测试、pytest）时退化为
    每次新建一个连接，行为和以前完全一样——13 处调用点都不用改。

    返回类型显式声明为 dict 行工厂的连接：池子本身不携带行类型信息，
    不加这层注解的话 pyright 会把所有 `row["列名"]` 推成 TupleRow 下标、
    全仓库报出上百个假阳性（本次工程化踩过的坑）。
    """
    if _pool is not None:
        return _pool.connection()  # type: ignore[no-any-return]  # kwargs 已注入 dict_row
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def vector_literal(values: Iterable[float]) -> str:
    """将 embedding 转成 pgvector 可解析的文本格式。"""
    return "[" + ",".join(str(float(value)) for value in values) + "]"


# --- 人物关系图（M4 schema v2）的 DDL 常量 -----------------------------------
# 抽成常量是因为同一段 DDL 要在两个入口复用：建索引事务（_ensure_index_schema）
# 和 FastAPI 启动时的独立升级入口（ensure_graph_review_schema）——审核界面
# 不应该要求用户先重建一次索引才能打开页面。
_CHARACTER_RELATIONS_DDL = """
CREATE TABLE IF NOT EXISTS character_relations (
    novel      TEXT    NOT NULL,
    person_a   TEXT    NOT NULL,
    person_b   TEXT    NOT NULL,
    relation   TEXT    NOT NULL,
    weight     INTEGER NOT NULL,
    -- M4 schema v2：以下五列是质量闭环的基础。老库通过幂等 ALTER 补列，
    -- 新库建表时直接带上，所以这里的形状就是最终形状。
    direction        TEXT,
    confidence       REAL,
    evidence_type    TEXT,
    source_chunk_ids JSONB,
    review_status    TEXT NOT NULL DEFAULT 'pending',
    PRIMARY KEY (novel, person_a, person_b, relation)
)
"""

_GRAPH_CHARACTERS_DDL = """
CREATE TABLE IF NOT EXISTS graph_characters (
    novel        TEXT NOT NULL,
    relation     TEXT NOT NULL,
    sample_hash  TEXT NOT NULL,
    -- 抽到的人名列表（JSON 数组）。存整个列表而不是一行一个名字：
    -- 它是「一次抽取的完整结果」，拆开存反而要额外判断是不是抽全了。
    names        JSONB NOT NULL,
    -- M4：LLM 抽出的关系记录（含方向/置信度/来源片段）。NULL 表示
    -- 「还没用 LLM 抽过」，与「抽过了但没抽到」（空数组）区分开：
    -- 前者下次要重试，后者直接复用。老库经幂等 ALTER 补列。
    relations    JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (novel, relation, sample_hash)
)
"""


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
    conn.execute("ALTER TABLE novel_chunks ADD COLUMN IF NOT EXISTS chapter_title TEXT")
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
    # character_relations 的建表与补列统一在 _ensure_graph_schema 里做；
    # index_manifest 是文件清单，不和某次任务绑定。只有一本书的两套索引
    # 在同一事务里完整写入后，才更新这里的哈希；任务失败或取消时事务回滚，
    # 下次会安全地再次识别为待处理。
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
    conn.execute("CREATE INDEX IF NOT EXISTS chunk_terms_term_idx ON chunk_terms (term)")
    # BM25 两阶段聚合（M3.4 性能优化，见 retrieval_mixins.keyword_retrieve）
    # 的每词 Top-N 依赖这个复合索引：候选子查询的 ORDER BY (tf DESC, novel,
    # chunk_id) 与索引键序一致，常见词取前 N 行时可以提前终止扫描，而不是
    # 读完并排序全部命中行。缺了它查询结果不变，只是退化为每词排序。
    conn.execute(
        "CREATE INDEX IF NOT EXISTS chunk_terms_term_tf_idx "
        "ON chunk_terms (term, tf DESC, novel, chunk_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS novel_chunks_novel_chunk_idx "
        "ON novel_chunks (novel, chunk_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS novel_chunks_embedding_hnsw_idx "
        "ON novel_chunks USING hnsw (embedding vector_cosine_ops)"
    )
    _ensure_graph_schema(conn)
    _ensure_hierarchy_schema(conn, dimension)


def _ensure_graph_schema(conn) -> None:
    """幂等准备人物关系图两张表的 schema v2（M4 质量闭环）。

    沿用全项目统一的 ``ADD COLUMN IF NOT EXISTS`` 迁移模式：新库由
    CREATE TABLE 直接带全列（ALTER 全部空跑），老库逐条补列、已有数据
    不受影响。补列之后做一次**有条件的回填**——把 v1 时代留下的旧行如实
    标注为共现推断边（它们本来就是共现统计的产物），否则这些行的
    evidence_type 是 NULL，质量门槛无法区分对待。WHERE 条件保证重复执行
    是零成本的。

    刻意做成自包含：不依赖 _ensure_index_schema 先建好基础表，这样 FastAPI
    启动时单独调用它（ensure_graph_review_schema）也不会因为表不存在而失败。
    """
    conn.execute(_CHARACTER_RELATIONS_DDL)
    conn.execute("ALTER TABLE character_relations ADD COLUMN IF NOT EXISTS direction TEXT")
    conn.execute("ALTER TABLE character_relations ADD COLUMN IF NOT EXISTS confidence REAL")
    conn.execute("ALTER TABLE character_relations ADD COLUMN IF NOT EXISTS evidence_type TEXT")
    conn.execute(
        "ALTER TABLE character_relations ADD COLUMN IF NOT EXISTS source_chunk_ids JSONB"
    )
    conn.execute(
        "ALTER TABLE character_relations ADD COLUMN IF NOT EXISTS "
        "review_status TEXT NOT NULL DEFAULT 'pending'"
    )
    # v1 旧行回填：低置信度的共现推断。只动 evidence_type IS NULL 的行，
    # 因此幂等且几乎零成本。
    conn.execute(
        "UPDATE character_relations SET evidence_type = 'co_occurrence', confidence = 0.3 "
        "WHERE evidence_type IS NULL"
    )
    # 关系边的查询索引放在这里而不是 _ensure_index_schema：建表也在这一个函数里，
    # 顺序保证了「先表后索引」。此前放在 _ensure_index_schema 时会先于建表执行，
    # 全新数据库（临时实验库、夜间 CI 库）会直接 UndefinedTable 失败。
    conn.execute(
        "CREATE INDEX IF NOT EXISTS character_relations_a_idx "
        "ON character_relations (person_a, relation)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS character_relations_b_idx "
        "ON character_relations (person_b, relation)"
    )
    # 人名抽取缓存表：M4 起 LLM 抽出的关系记录一并缓存，否则"人名命中缓存"
    # 会让抽取静默降级成纯共现，质量悄悄倒退。
    conn.execute(_GRAPH_CHARACTERS_DDL)
    conn.execute("ALTER TABLE graph_characters ADD COLUMN IF NOT EXISTS relations JSONB")


def ensure_graph_review_schema() -> None:
    """FastAPI 启动时的独立入口：确保审核端点依赖的表和列存在（幂等）。

    审核界面读写的只是 character_relations / graph_characters 两张表，
    不应该要求用户先重建一次索引才能打开页面。
    """
    with connect() as conn:
        _ensure_graph_schema(conn)


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
            "SELECT novel, source_hash, pipeline_hash, node_count FROM hierarchy_manifest"
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
    relations: list[dict] | None = None,
    cancel_check: Callable[[], None] | None = None,
    hierarchy_rows: list[tuple] | None = None,
    hierarchy_hash: str | None = None,
    quality_report: dict | None = None,
) -> None:
    """在一个事务里原子替换单本书的全部派生数据。

    embedding 和分词在进入这里之前已经准备好。删除旧数据、写向量、COPY BM25、
    写关系边和更新 manifest 共享同一事务；任何异常（包括用户取消）都会回滚，
    因此检索永远只会看到旧版或完整新版，不会看到“向量已换、BM25 只写一半”。

    ``relations`` 是 graph.build_edge_records / extract_relations_llm 产出的
    边记录 dict（含 M4 的方向/置信度/证据类型/来源片段），review_status 入库时
    统一从 'pending' 起步——重建索引会重置审核结果，因为边本身已经是新抽的了。
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
        # BM25 的“一个片段 × 多个词”会产生大量行，COPY 比逐条 INSERT 快得多。
        # COPY 仍属于外层同一个事务，并不会削弱原子性。
        with (
            conn.cursor() as cursor,
            cursor.copy("COPY chunk_terms (novel, chunk_id, term, tf) FROM STDIN") as copy,
        ):
            for index, (row, freqs) in enumerate(zip(rows, per_chunk_terms, strict=True)):
                if index % 50 == 0:
                    check()
                chunk_id = row[1]
                for term, tf in freqs.items():
                    copy.write_row((novel, chunk_id, term, tf))
        if relations:
            with conn.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO character_relations "
                    "(novel, person_a, person_b, relation, weight, direction, "
                    "confidence, evidence_type, source_chunk_ids, review_status) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)",
                    [
                        (
                            record["novel"],
                            record["person_a"],
                            record["person_b"],
                            record["relation"],
                            int(record["weight"]),
                            record.get("direction"),
                            record.get("confidence"),
                            record.get("evidence_type"),
                            json.dumps(
                                record.get("source_chunk_ids") or [], ensure_ascii=False
                            ),
                            "pending",
                        )
                        for record in relations
                    ],
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
            (
                novel,
                source_hash,
                pipeline_hash,
                len(rows),
                json.dumps(quality_report or {}, ensure_ascii=False),
            ),
        )


def delete_novel_index(novel: str, cancel_check: Callable[[], None] | None = None) -> None:
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
        row = conn.execute("SELECT count(*) AS count FROM novel_chunks").fetchone()
        return int(row["count"]) if row else 0


def hierarchy_node_count() -> int:
    """返回当前章节 + 全书摘要节点总数；尚未迁移时为 0。"""
    with connect() as conn:
        table = conn.execute(
            "SELECT to_regclass('public.hierarchy_summaries') AS table_name"
        ).fetchone()
        if not table or not table["table_name"]:
            return 0
        row = conn.execute("SELECT count(*) AS count FROM hierarchy_summaries").fetchone()
        return int(row["count"]) if row else 0
    return 0


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
                "ALTER TABLE novel_chunks ADD COLUMN IF NOT EXISTS chapter_title TEXT"
            )


def _relation_visibility_clause(
    require_explicit: bool | None = None, min_confidence: float | None = None
) -> tuple[str, list]:
    """生成在线查询的可见性过滤 SQL 片段和参数（M4 质量门槛）。纯函数便于单测。

    两条规则，按严格程度递进：
    - **被人工拒绝的边永远不可见**——审核结果是最高优先级，任何开关都不能
      让 rejected 的边复活；
    - GRAPH_REQUIRE_EXPLICIT 开启时（默认），只有「明确关系陈述」且置信度
      达标的边才进入在线结果；co_occurrence 边保留在库里供审核界面处理，
      只是不再自动展示给问答模型。
    COALESCE 兜底是为了 v1 旧行：理论上迁移已回填，但防御 NULL 永远不亏。
    """
    if require_explicit is None:
        require_explicit = GRAPH_REQUIRE_EXPLICIT
    if min_confidence is None:
        min_confidence = GRAPH_MIN_CONFIDENCE
    clause = "COALESCE(review_status, 'pending') <> 'rejected'"
    params: list = []
    if require_explicit:
        clause += " AND evidence_type = 'explicit' AND COALESCE(confidence, 0) >= %s"
        params.append(min_confidence)
    return clause, params


def query_relations(
    person: str, relation: str, limit: int = 8, min_weight: int = 2
) -> list[tuple[str, int]]:
    """查某个人物在某种关系下的所有对手方，按权重从高到低。

    权重的含义随边的来源不同：共现边是共现次数，LLM 边是来源片段数——
    都是"支持这条关系的证据量"，排序语义一致。

    min_weight 默认 2：只出现一次的边大概率是偶然同框（两个人碰巧出现在
    同一段提到「师父」的话里），过滤掉能显著降噪。

    人名可能存在于 person_a 或 person_b 任一侧（person_a/person_b 按字典序存，
    关系方向只写在 direction 里），所以两侧都要查，用 UNION ALL 合并；
    可见性过滤（审核状态 + M4 质量门槛）对两侧同样生效。
    """
    clause, gate_params = _relation_visibility_clause()
    # psycopg 的存根把查询参数声明成 LiteralString，运行时拼好的门槛片段
    # 过不了静态检查；片段来自上面的纯函数、不含任何外部输入，安全。
    gated_sql = f"""
        SELECT other, SUM(weight) AS weight FROM (
            SELECT person_b AS other, weight FROM character_relations
            WHERE person_a = %s AND relation = %s AND {clause}
            UNION ALL
            SELECT person_a AS other, weight FROM character_relations
            WHERE person_b = %s AND relation = %s AND {clause}
        ) AS both_sides
        GROUP BY other
        HAVING SUM(weight) >= %s
        ORDER BY weight DESC
        LIMIT %s
        """
    with connect() as conn:
        rows = conn.execute(
            gated_sql,  # type: ignore[reportArgumentType]
            (
                person,
                relation,
                *gate_params,
                person,
                relation,
                *gate_params,
                min_weight,
                limit,
            ),
        ).fetchall()
    return [(r["other"], int(r["weight"])) for r in rows]


def known_characters(novel: str | None = None) -> list[str]:
    """图里出现过的所有人物名（用于在问题里识别"用户问的是谁"）。

    只统计未被拒绝的边上的名字：一条边被拒绝后，它两端的人名不应再因为
    这条死边被当成图检索的查询对象。
    """
    not_rejected = "COALESCE(review_status, 'pending') <> 'rejected'"
    if novel:
        scope_a = f"WHERE novel = %s AND {not_rejected}"
        scope_b = f"WHERE novel = %s AND {not_rejected}"
    else:
        scope_a = f"WHERE {not_rejected}"
        scope_b = f"WHERE {not_rejected}"
    params = (novel,) if novel else ()
    with connect() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT person_a AS name FROM character_relations {scope_a} "
            f"UNION SELECT DISTINCT person_b FROM character_relations {scope_b}",
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
        conn.execute(_GRAPH_CHARACTERS_DDL)
        # M4：老库补 relations 列（新库建表语句已带）。DDL 与 _ensure_graph_schema
        # 共用同一个常量，两处不可能漂移。
        conn.execute("ALTER TABLE graph_characters ADD COLUMN IF NOT EXISTS relations JSONB")


def load_cached_graph_characters(novel: str, relation: str, sample_hash: str) -> dict | None:
    """读回缓存的抽取结果；没缓存返回 None（和"缓存了空结果"区分开）。

    返回 {"names": [...], "relations": [...] | None}：
    - names 是人名列表；
    - relations 是 M4 的 LLM 关系记录列表，**None 表示还没用 LLM 抽过**
      （老缓存行、或上次抽取失败）——调用方据此决定要不要重试抽取，
      而不是误以为"LLM 确认过没有关系"。
    """
    with connect() as conn:
        row = conn.execute(
            "SELECT names, relations FROM graph_characters "
            "WHERE novel = %s AND relation = %s AND sample_hash = %s",
            (novel, relation, sample_hash),
        ).fetchone()
    if not row:
        return None
    return {
        "names": list(row["names"]),
        # psycopg 会把 JSONB 解成 list；显式转一遍让"没抽过"(NULL) 保持 None
        "relations": list(row["relations"]) if row["relations"] is not None else None,
    }


def save_graph_characters(
    novel: str,
    relation: str,
    sample_hash: str,
    names: list[str],
    relations: list[dict] | None = None,
) -> None:
    """写入抽取结果（幂等）。空列表也会写——"这批确实没抽到"本身就是有效结果。

    ``relations`` 只在 LLM 抽取**成功跑完**时传（哪怕空列表，也代表"模型确认过
    这些片段里没有关系"）；纯共现路径不传，保持 NULL 以便后端可用时重试。
    """
    with connect() as conn:
        conn.execute(
            "INSERT INTO graph_characters (novel, relation, sample_hash, names, relations) "
            "VALUES (%s, %s, %s, %s::jsonb, %s::jsonb) "
            "ON CONFLICT (novel, relation, sample_hash) "
            "DO UPDATE SET names = EXCLUDED.names, relations = EXCLUDED.relations",
            (
                novel,
                relation,
                sample_hash,
                json.dumps(names, ensure_ascii=False),
                json.dumps(relations, ensure_ascii=False) if relations is not None else None,
            ),
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
    with connect() as conn, conn.cursor() as cursor:
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
        conn.execute("ALTER TABLE chat_turns ADD COLUMN IF NOT EXISTS agent_steps JSONB")
        # M3.5-④：在线配置快照列（幂等补列，模式与上面 agent_steps 一致）。
        conn.execute("ALTER TABLE chat_turns ADD COLUMN IF NOT EXISTS run_config JSONB")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS chat_turns_session_idx "
            "ON chat_turns (session_id, turn_index)"
        )
        # M3.6：滚动会话摘要。一个会话一行，覆盖到哪一轮记在 covers_through，
        # 靠它判断"哪些轮次还没进摘要"——不记的话每次都得重新摘要全部历史，
        # 那就不叫滚动了。摘要是派生数据，丢了只是回到"只有最近几轮原文"。
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_session_summaries (
                session_id     UUID PRIMARY KEY,
                summary        TEXT    NOT NULL,
                covers_through INTEGER NOT NULL,
                model          TEXT,
                updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
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
    run_config: dict | None = None,
) -> None:
    """写入或覆盖一轮对话（幂等）。

    用 ON CONFLICT DO UPDATE 而非 INSERT：中断保存可能被重复调用，
    同一 (session_id, turn_index) 直接覆盖，不会主键冲突。

    ``run_config``：本轮问答的在线配置快照（M3.5-④，见 backend/main.py 的
    _build_run_config）。**隐私红线**：快照里不得出现 API Key、完整版权原文、
    或用户问题原文之外的隐私——调用方负责保证内容，这里只做透明落库。
    """
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO chat_turns
                (session_id, turn_index, role, content, sources, trace,
                 agent_steps, status, run_config)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s::jsonb)
            ON CONFLICT (session_id, turn_index) DO UPDATE SET
                role        = EXCLUDED.role,
                content     = EXCLUDED.content,
                sources     = EXCLUDED.sources,
                trace       = EXCLUDED.trace,
                agent_steps = EXCLUDED.agent_steps,
                status      = EXCLUDED.status,
                run_config  = EXCLUDED.run_config
            """,
            (
                session_id,
                turn_index,
                role,
                content,
                json.dumps(sources, ensure_ascii=False) if sources is not None else None,
                json.dumps(trace, ensure_ascii=False) if trace is not None else None,
                json.dumps(agent_steps, ensure_ascii=False)
                if agent_steps is not None
                else None,
                status,
                json.dumps(run_config, ensure_ascii=False) if run_config is not None else None,
            ),
        )


def load_turns(session_id: str) -> list[dict]:
    """按顺序读回某个会话的全部对话轮次。"""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT turn_index, role, content, sources, trace, agent_steps,
                   status, run_config
            FROM chat_turns
            WHERE session_id = %s
            ORDER BY turn_index
            """,
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def load_session_summary(session_id: str) -> dict | None:
    """读这个会话的滚动摘要；没有就返回 None（第一次或还没到阈值）。"""
    with connect() as conn:
        row = conn.execute(
            "SELECT summary, covers_through, model FROM chat_session_summaries "
            "WHERE session_id = %s",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def save_session_summary(
    session_id: str, summary: str, covers_through: int, model: str | None = None
) -> None:
    """写入或覆盖会话摘要（幂等，一个会话只有一行）。"""
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO chat_session_summaries
                (session_id, summary, covers_through, model, updated_at)
            VALUES (%s, %s, %s, %s, now())
            ON CONFLICT (session_id) DO UPDATE SET
                summary        = EXCLUDED.summary,
                covers_through = EXCLUDED.covers_through,
                model          = EXCLUDED.model,
                updated_at     = now()
            """,
            (session_id, summary, covers_through, model),
        )


def next_turn_index(session_id: str) -> int:
    """该会话下一轮的序号；空会话返回 0。"""
    with connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(turn_index) + 1, 0) AS next FROM chat_turns "
            "WHERE session_id = %s",
            (session_id,),
        ).fetchone()
    return int(row["next"]) if row else 0


# --------------------------------------------------------------- 索引任务记录
# 任务卡片的状态落库（路线图 M3.3.5）。这张表和 chat_turns 一样属于「与小说派生
# 索引无关」的独立数据：只记录后台任务的可见状态，数据一致性仍由单书事务保证，
# 所以这里不需要任何复杂结构——一行 = 一个任务快照，UPSERT 即可。

_INDEX_TASK_ACTIVE_SQL = "('queued', 'running', 'cancelling')"


def ensure_index_task_schema() -> None:
    """建索引任务运行记录表（幂等，用 IF NOT EXISTS）。

    时间列用 TIMESTAMPTZ 而不是 TEXT：manager 传进来的本来就是 ISO 字符串，
    psycopg 能直接解析；落库成真正的时间类型后，以后想按时间清理旧记录或算
    任务耗时都是 SQL 一句话的事。
    """
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS index_task_runs (
                id          TEXT PRIMARY KEY,
                force       BOOLEAN NOT NULL DEFAULT FALSE,
                retry_of    TEXT,
                status      TEXT NOT NULL,
                stage       TEXT NOT NULL DEFAULT '',
                progress    INTEGER NOT NULL DEFAULT 0,
                message     TEXT NOT NULL DEFAULT '',
                error       TEXT,
                result      JSONB,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                started_at  TIMESTAMPTZ,
                finished_at TIMESTAMPTZ
            )
            """
        )


def save_index_task_run(snapshot: dict) -> None:
    """写入/覆盖一个任务快照（id 是主键，重复写同一条就是覆盖）。

    进度类字段会随任务推进被反复 UPSERT，这正好符合「重启后恢复显示」的语义：
    表里永远保留每个任务最后已知的状态。
    """
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO index_task_runs
                (id, force, retry_of, status, stage, progress,
                 message, error, result, created_at, started_at, finished_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                force       = EXCLUDED.force,
                retry_of    = EXCLUDED.retry_of,
                status      = EXCLUDED.status,
                stage       = EXCLUDED.stage,
                progress    = EXCLUDED.progress,
                message     = EXCLUDED.message,
                error       = EXCLUDED.error,
                result      = EXCLUDED.result,
                created_at  = EXCLUDED.created_at,
                started_at  = EXCLUDED.started_at,
                finished_at = EXCLUDED.finished_at
            """,
            (
                snapshot["id"],
                bool(snapshot.get("force")),
                snapshot.get("retry_of"),
                snapshot["status"],
                snapshot.get("stage", ""),
                int(snapshot.get("progress", 0)),
                snapshot.get("message", ""),
                snapshot.get("error"),
                json.dumps(snapshot["result"], ensure_ascii=False)
                if snapshot.get("result") is not None
                else None,
                snapshot.get("created_at"),
                snapshot.get("started_at"),
                snapshot.get("finished_at"),
            ),
        )


def _task_row_to_snapshot(row: dict) -> dict:
    """把表行还原成 API 层认识的快照字典（时间转回 ISO 字符串）。"""

    def _iso(value):
        return value.isoformat() if value is not None else None

    return {
        "id": row["id"],
        "status": row["status"],
        "stage": row["stage"],
        "progress": int(row["progress"]),
        "message": row["message"],
        "error": row["error"],
        "force": bool(row["force"]),
        "retry_of": row["retry_of"],
        "result": row["result"],
        "created_at": _iso(row["created_at"]),
        "started_at": _iso(row["started_at"]),
        "finished_at": _iso(row["finished_at"]),
    }


def load_latest_index_task_runs(limit: int = 1) -> list[dict]:
    """按创建时间倒序读回最近的任务快照（默认只要最新一条，供重启恢复）。"""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM index_task_runs ORDER BY created_at DESC LIMIT %s
            """,
            (limit,),
        ).fetchall()
    return [_task_row_to_snapshot(row) for row in rows]


def load_index_task_run(task_id: str) -> dict | None:
    """按 id 读回单个任务快照；不存在返回 None。"""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM index_task_runs WHERE id = %s",
            (task_id,),
        ).fetchone()
    return _task_row_to_snapshot(row) if row else None


def mark_interrupted_index_task_runs() -> int:
    """把上次进程遗留的 active 状态改成 failed（服务重启时调用一次）。

    重启意味着旧线程已经没了、进行中的事务已随连接断开而回滚——这些任务既不
    可能继续跑也不可能变成 completed，如实地标成 failed 并提示重试，比留着
    一个永远不会更新的"running"卡片诚实。返回受影响的行数，仅供启动日志。
    """
    with connect() as conn:
        rows = conn.execute(
            f"""
            UPDATE index_task_runs
            SET status = 'failed',
                stage = 'failed',
                message = '后端曾重启，任务中断；已完成的书保持可用，可安全重试',
                finished_at = COALESCE(finished_at, now())
            WHERE status IN {_INDEX_TASK_ACTIVE_SQL}
            RETURNING id
            """
        ).fetchall()
    return len(rows)


# ------------------------------------------------------- 关系边审核（M4 质量闭环）
# 共现推断必然产生假边，LLM 抽取也只能降低而不是消灭错误率。与其让门槛"一刀切"，
# 不如把被门槛挡住的边留在库里、交给人工逐条通过/拒绝——机器判断和人工复核
# 各管一段，这就是"质量闭环"的含义。

VALID_REVIEW_STATUSES = ("pending", "approved", "rejected")


def list_relation_edges(
    status: str | None = "pending", limit: int = 50, offset: int = 0
) -> tuple[list[dict], int]:
    """按审核状态分页列出关系边，附证据摘录；返回 (边列表, 符合条件的总数)。

    这是审核界面的数据源。status=None 列全部状态（默认只看 pending——审核员
    打开页面最关心的是待处理队列）。v1 旧行的 review_status 经迁移回填不会是
    NULL，但查询仍用 COALESCE 防御。

    证据摘录取第一个来源片段的原文前 80 字：摘录只是帮审核员快速定位，
    完整原文永远以 novel_chunks 为准，不在关系表里复制第二份。
    """
    where = "WHERE COALESCE(r.review_status, 'pending') = %s" if status else ""
    params: list = [status] if status else []
    with connect() as conn:
        total_row = conn.execute(
            f"SELECT count(*) AS count FROM character_relations r {where}",
            params,
        ).fetchone()
        rows = conn.execute(
            f"""
            SELECT r.novel, r.person_a, r.person_b, r.relation, r.weight,
                   r.direction, r.confidence, r.evidence_type,
                   COALESCE(r.source_chunk_ids, '[]'::jsonb) AS source_chunk_ids,
                   COALESCE(r.review_status, 'pending') AS review_status,
                   LEFT(nc.text, 81) AS evidence_head
            FROM character_relations r
            LEFT JOIN novel_chunks nc
                ON nc.novel = r.novel
                AND nc.chunk_id = (COALESCE(r.source_chunk_ids, '[]'::jsonb) ->> 0)::int
            {where}
            ORDER BY r.weight DESC, r.novel, r.person_a, r.person_b
            LIMIT %s OFFSET %s
            """,
            [*params, limit, offset],
        ).fetchall()

    edges = []
    for row in rows:
        edge = dict(row)
        # 摘录截到 80 字（LEFT 取 81 是为了知道"还有更多"时能补省略号）
        head = edge.pop("evidence_head") or ""
        edge["evidence_excerpt"] = head[:80] + ("…" if len(head) > 80 else "")
        ids = edge["source_chunk_ids"]
        edge["source_chunk_ids"] = [int(i) for i in ids] if isinstance(ids, list) else []
        edges.append(edge)
    total = int(total_row["count"]) if total_row else 0
    return edges, total


def set_relation_review(
    novel: str, person_a: str, person_b: str, relation: str, review_status: str
) -> int:
    """写入一条边的审核结论，返回更新的行数（0 = 边不存在）。

    只接受 pending/approved/rejected 三种状态；调用方（端点层）负责校验并把
    非法值转成 400。写入即生效：rejected 的边在下一个查询里就不可见了。
    """
    if review_status not in VALID_REVIEW_STATUSES:
        raise ValueError(f"非法的审核状态：{review_status!r}")
    with connect() as conn:
        cursor = conn.execute(
            """
            UPDATE character_relations SET review_status = %s
            WHERE novel = %s AND person_a = %s AND person_b = %s AND relation = %s
            """,
            (review_status, novel, person_a, person_b, relation),
        )
        return cursor.rowcount
