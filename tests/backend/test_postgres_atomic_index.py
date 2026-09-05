import pytest

import postgres


class _FailingCopy:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write_row(self, row):
        raise RuntimeError("COPY interrupted")


class _Cursor:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def executemany(self, sql, rows):
        self.rows = rows

    def copy(self, sql):
        return _FailingCopy()


class _Connection:
    def __init__(self):
        self.sql = []
        self.exit_exception = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.exit_exception = exc_type
        return False

    def execute(self, sql, params=None):
        self.sql.append(" ".join(sql.split()))
        return self

    def cursor(self):
        return _Cursor()


def test_replace_book_uses_one_transaction_and_does_not_advance_manifest_on_failure(
    monkeypatch,
):
    connection = _Connection()
    monkeypatch.setattr(postgres, "connect", lambda: connection)
    rows = [("书", 0, None, "正文", "[0.1,0.2]", 1, "")]

    with pytest.raises(RuntimeError, match="COPY interrupted"):
        postgres.replace_novel_index("书", rows, [{"正文": 1}], "source", "pipeline")

    assert connection.exit_exception is RuntimeError
    assert any("DELETE FROM novel_chunks" in sql for sql in connection.sql)
    assert not any("INSERT INTO index_manifest" in sql for sql in connection.sql)


def test_cancel_before_replace_leaves_transaction_without_delete(monkeypatch):
    connection = _Connection()
    monkeypatch.setattr(postgres, "connect", lambda: connection)

    with pytest.raises(RuntimeError, match="cancelled"):
        postgres.replace_novel_index(
            "书",
            [],
            [],
            "source",
            "pipeline",
            cancel_check=lambda: (_ for _ in ()).throw(RuntimeError("cancelled")),
        )

    assert connection.sql == []


def test_quality_report_is_written_in_same_transaction(monkeypatch):
    connection = _Connection()
    monkeypatch.setattr(postgres, "connect", lambda: connection)
    rows = [("书", 0, None, "正文", "[0.1,0.2]", 1, "")]

    postgres.replace_novel_index(
        "书", rows, [{}], "source", "pipeline", quality_report={"passed": True}
    )

    manifest_sql = [sql for sql in connection.sql if "INSERT INTO index_manifest" in sql]
    assert manifest_sql
    assert "quality_report" in manifest_sql[0]


def test_row_and_term_length_mismatch_is_rejected_before_touching_the_database(
    monkeypatch,
):
    """rows 和 per_chunk_terms 必须等长，且要在连数据库之前就拦住。

    这两个列表是按下标一一对应的（第 i 个片段的向量行 ↔ 第 i 个片段的词频）。
    长度不一致意味着上游生成逻辑已经错位，此时若照常写库，向量和 BM25 会
    对不上号——检索结果会静默错乱，而不是报错。所以这里必须是"提前失败"，
    而且不能已经把旧数据 DELETE 掉了才发现。
    """
    touched = []
    monkeypatch.setattr(
        postgres, "connect", lambda: touched.append("connected") or _Connection()
    )

    with pytest.raises(ValueError):
        postgres.replace_novel_index(
            "雾隐山庄",
            rows=[("雾隐山庄", 0, None, "正文", "[0.1]", 2, "")],
            per_chunk_terms=[{"正文": 1}, {"多出来的": 1}],
            source_hash="h",
            pipeline_hash="p",
        )

    assert touched == [], "长度校验必须发生在连接数据库之前，不能先删旧数据再报错"
