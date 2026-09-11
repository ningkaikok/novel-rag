import postgres


class _FakeConn:
    def __init__(self, table_exists=True):
        self.table_exists = table_exists
        self.sql = []
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sql.append(sql)
        self.calls.append((sql, params))
        return self

    def fetchone(self):
        return {"table_name": "novel_chunks" if self.table_exists else None}


def test_old_index_gets_chapter_column_without_rebuild(monkeypatch):
    conn = _FakeConn(table_exists=True)
    monkeypatch.setattr(postgres, "connect", lambda: conn)

    postgres.ensure_novel_metadata_schema()

    assert any("ADD COLUMN IF NOT EXISTS chapter_title" in sql for sql in conn.sql)


def test_metadata_upgrade_is_safe_before_first_index(monkeypatch):
    conn = _FakeConn(table_exists=False)
    monkeypatch.setattr(postgres, "connect", lambda: conn)

    postgres.ensure_novel_metadata_schema()

    assert not any("ALTER TABLE" in sql for sql in conn.sql)


def test_citation_feedback_stores_only_answer_hash(monkeypatch):
    conn = _FakeConn()
    monkeypatch.setattr(postgres, "connect", lambda: conn)

    postgres.save_citation_feedback(
        answer="顾长风中了蚀骨散[1]。",
        citation=1,
        novel="雾隐山庄",
        chunk_id=3,
        feedback="incorrect",
    )

    sql, params = next((sql, params) for sql, params in conn.calls)
    assert "INSERT INTO citation_feedback" in sql
    assert params[0] is None
    assert params[3] == "雾隐山庄"
    assert params[5] == "incorrect"
    assert params[6] != "顾长风中了蚀骨散[1]。"
    assert len(params[6]) == 64


def test_citation_judgment_stores_only_hash_and_aggregates(monkeypatch):
    conn = _FakeConn()
    monkeypatch.setattr(postgres, "connect", lambda: conn)

    postgres.save_citation_judgment(
        answer="顾长风中了蚀骨散[1]。",
        citation=1,
        novel="雾隐山庄",
        chunk_id=3,
        label="partial",
        method="shadow_two_step",
        model="glm:glm-4-flash",
        claim_count=2,
        supported_count=1,
        not_found_count=1,
    )

    sql, params = conn.calls[-1]
    assert "INSERT INTO citation_judgments" in sql
    assert params[5:9] == ("partial", "shadow_two_step", "glm:glm-4-flash", 2)
    assert params[9:12] == (1, 0, 1)
    assert params[-1] != "顾长风中了蚀骨散[1]。"
    assert len(params[-1]) == 64


def test_load_citation_calibration_rows_returns_only_labels(monkeypatch):
    class _RowsConn(_FakeConn):
        def fetchall(self):
            return [{"feedback": "helpful", "label": "supported", "method": "m", "model": "x"}]

    conn = _RowsConn()
    monkeypatch.setattr(postgres, "connect", lambda: conn)
    rows = postgres.load_citation_calibration_rows(limit=3)
    assert rows == [{"feedback": "helpful", "label": "supported", "method": "m", "model": "x"}]
    assert "JOIN LATERAL" in conn.calls[-1][0]
    assert conn.calls[-1][1] == (3,)
