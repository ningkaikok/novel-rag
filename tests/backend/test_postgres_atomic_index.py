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
        postgres.replace_novel_index(
            "书", rows, [{"正文": 1}], "source", "pipeline"
        )

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
