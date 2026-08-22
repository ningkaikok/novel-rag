"""M4 关系图存储层测试：schema v2 幂等迁移、质量门槛 SQL、审核读写。

全部用假连接/假游标，不连真实数据库。迁移测试只断言「发出了正确的
幂等 DDL」，与 test_postgres_metadata.py 的既有模式保持一致。
"""

import postgres


class _FakeConn:
    """记录每条执行的 SQL 和参数，fetch 行为按需注入。"""

    def __init__(self, fetch_rows=None):
        self.sql = []
        self.params = []
        self.fetch_rows = fetch_rows or []
        # psycopg 的 execute 返回 cursor，UPDATE 后靠它读受影响行数
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sql.append(" ".join(sql.split()))
        self.params.append(params)
        return self

    def fetchall(self):
        return self.fetch_rows

    def fetchone(self):
        return self.fetch_rows[0] if self.fetch_rows else None


# ------------------------------------------------------------- schema v2 迁移


def test_graph_schema_migration_is_idempotent_ddl(monkeypatch):
    """v2 迁移必须全部是 IF NOT EXISTS 形态：老库升级可重复执行。"""
    conn = _FakeConn()
    monkeypatch.setattr(postgres, "connect", lambda: conn)

    postgres.ensure_graph_review_schema()

    alters = [sql for sql in conn.sql if "ADD COLUMN IF NOT EXISTS" in sql]
    for column in ("direction", "confidence", "evidence_type", "source_chunk_ids"):
        assert any(
            f"character_relations ADD COLUMN IF NOT EXISTS {column}" in sql for sql in alters
        ), f"缺 {column} 列的幂等补列"
    assert any(
        "character_relations ADD COLUMN IF NOT EXISTS review_status" in sql
        and "DEFAULT 'pending'" in sql
        for sql in alters
    )
    # 人名缓存表也要补 relations 列：否则"人名命中缓存"会让 LLM 抽取结果丢失
    assert any("graph_characters ADD COLUMN IF NOT EXISTS relations" in sql for sql in alters)


def test_graph_schema_backfills_legacy_rows_as_co_occurrence(monkeypatch):
    """v1 旧行没有证据类型，回填成低置信度共现边；WHERE 条件保证可重复执行。"""
    conn = _FakeConn()
    monkeypatch.setattr(postgres, "connect", lambda: conn)

    postgres.ensure_graph_review_schema()

    backfills = [sql for sql in conn.sql if sql.startswith("UPDATE character_relations")]
    assert len(backfills) == 1
    assert "evidence_type IS NULL" in backfills[0], "只动 NULL 行，重复执行零成本"
    assert "co_occurrence" in backfills[0]


def test_index_schema_migration_includes_graph_v2(monkeypatch):
    """建索引事务里的迁移入口同样带全 v2 列（一条路径漏了就会出现两套 schema）。"""
    executed = []

    class _Conn(_FakeConn):
        def execute(self, sql, params=None):  # hierarchy 表需要 dimension 插值
            executed.append(sql if isinstance(sql, str) else "")
            return self

    conn = _Conn()
    monkeypatch.setattr(postgres, "connect", lambda: conn)

    from postgres import _ensure_index_schema

    _ensure_index_schema(conn, dimension=4)

    assert any("ADD COLUMN IF NOT EXISTS direction" in sql for sql in executed)


# ------------------------------------------------------------- 在线质量门槛


def test_visibility_clause_off_keeps_pending_and_approved_only():
    """门槛关闭时只排除 rejected——审核结论优先于一切配置。"""
    clause, params = postgres._relation_visibility_clause(require_explicit=False)
    assert clause == "COALESCE(review_status, 'pending') <> 'rejected'"
    assert params == []


def test_visibility_clause_on_requires_explicit_and_confidence():
    clause, params = postgres._relation_visibility_clause(
        require_explicit=True, min_confidence=0.7
    )
    assert "review_status" in clause
    assert "evidence_type = 'explicit'" in clause
    assert "COALESCE(confidence, 0) >= %s" in clause
    assert params == [0.7]


def test_visibility_clause_reads_config_by_default(monkeypatch):
    """不传参数时读 config 注入的默认值（monkeypatch 模块属性模拟环境）。"""
    monkeypatch.setattr(postgres, "GRAPH_REQUIRE_EXPLICIT", True)
    monkeypatch.setattr(postgres, "GRAPH_MIN_CONFIDENCE", 0.55)
    _, params = postgres._relation_visibility_clause()
    assert params == [0.55]


def test_query_relations_applies_gate_to_both_sides(monkeypatch):
    """UNION ALL 两侧都要带门槛子句，参数顺序与占位符一一对应。"""
    conn = _FakeConn(fetch_rows=[{"other": "南宫婉", "weight": 9}])
    monkeypatch.setattr(postgres, "connect", lambda: conn)
    monkeypatch.setattr(postgres, "GRAPH_REQUIRE_EXPLICIT", True)
    monkeypatch.setattr(postgres, "GRAPH_MIN_CONFIDENCE", 0.7)

    result = postgres.query_relations("韩立", "伴侣", limit=5, min_weight=2)

    assert result == [("南宫婉", 9)]
    sql = conn.sql[-1]
    assert sql.count("<> 'rejected'") == 2, "person_a / person_b 两侧各过滤一次"
    assert sql.count("evidence_type = 'explicit'") == 2
    # 参数顺序：A侧(人,关系,置信度) + B侧(人,关系,置信度) + min_weight + limit
    assert conn.params[-1] == ("韩立", "伴侣", 0.7, "韩立", "伴侣", 0.7, 2, 5)


def test_known_characters_excludes_rejected_edges(monkeypatch):
    """被拒绝的边两端的人名不应再出现在图检索的反查里。"""
    conn = _FakeConn(fetch_rows=[{"name": "韩立"}])
    monkeypatch.setattr(postgres, "connect", lambda: conn)

    postgres.known_characters()

    sql = conn.sql[-1]
    assert sql.count("COALESCE(review_status") == 2


# ------------------------------------------------------------- 审核列表与写入


def test_list_relation_edges_paginates_with_status_filter(monkeypatch):
    row = {
        "novel": "青梧镇异闻",
        "person_a": "小顺",
        "person_b": "沈砚秋",
        "relation": "师徒",
        "weight": 3,
        "direction": "沈砚秋→小顺",
        "confidence": 0.85,
        "evidence_type": "explicit",
        "source_chunk_ids": [4, 9],
        "review_status": "pending",
        # LEFT(...,81) 取回 81 字符 → 截断后应带省略号
        "evidence_head": "字" * 81,
    }

    class _SeqConn:
        """list_relation_edges 先发 count、再发分页查询，按序返回两组结果。"""

        def __init__(self):
            self.sql = []
            self.params = []
            self._responses = [[{"count": 27}], [row]]
            self._current: list = []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params=None):
            self.sql.append(" ".join(sql.split()))
            self.params.append(params)
            self._current = self._responses.pop(0) if self._responses else []
            return self

        def fetchone(self):
            return self._current[0] if self._current else None

        def fetchall(self):
            return self._current

    seq_conn = _SeqConn()
    monkeypatch.setattr(postgres, "connect", lambda: seq_conn)
    edges, total = postgres.list_relation_edges(status="pending", limit=20, offset=40)

    assert total == 27
    assert len(edges) == 1
    edge = edges[0]
    assert edge["novel"] == "青梧镇异闻"
    assert edge["evidence_excerpt"].endswith("…"), "超过 80 字的摘录要截断加省略号"
    assert len(edge["evidence_excerpt"]) == 81
    assert edge["source_chunk_ids"] == [4, 9]
    # 分页查询的参数：状态过滤 + limit + offset
    assert seq_conn.params[-1] == ["pending", 20, 40]


def test_set_relation_review_updates_by_full_primary_key(monkeypatch):
    conn = _FakeConn()
    monkeypatch.setattr(postgres, "connect", lambda: conn)

    updated = postgres.set_relation_review("书", "小顺", "沈砚秋", "师徒", "approved")

    assert updated == 1  # 假连接固定 rowcount=1，只关心 SQL 形状与参数
    sql = conn.sql[-1]
    assert sql.startswith("UPDATE character_relations SET review_status")
    assert "person_a = %s AND person_b = %s AND relation = %s" in sql
    assert conn.params[-1] == ("approved", "书", "小顺", "沈砚秋", "师徒")


def test_set_relation_review_rejects_unknown_status(monkeypatch):
    """非法状态在进数据库之前就拦下（端点层负责转成 400）。"""
    try:
        postgres.set_relation_review("书", "甲", "乙", "同伴", "maybe")
    except ValueError as exc:
        assert "maybe" in str(exc)
    else:
        raise AssertionError("非法状态必须抛 ValueError")
