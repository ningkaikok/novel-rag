import postgres


class _FakeConn:
    def __init__(self, table_exists=True):
        self.table_exists = table_exists
        self.sql = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sql.append(sql)
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
