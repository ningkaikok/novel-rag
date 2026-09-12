import v2_ingest
from parsers import MarkdownParser


class _Embedder:
    model_name = "test-embedding"

    def encode(self, texts, *, normalize_embeddings, show_progress_bar):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _Cursor:
    def __init__(self, rows):
        self.rows = iter(rows)
        self.params = []

    def execute(self, query, params=()):
        self.params.append((query, tuple(params)))
        return self

    def fetchone(self):
        return next(self.rows)


class _Connection:
    def __init__(self, rows):
        self.cursor = _Cursor(rows)

    def execute(self, query, params=()):
        return self.cursor.execute(query, params)

    def fetchone(self):
        return self.cursor.fetchone()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_index_v2_document_reuses_existing_version_number(monkeypatch):
    connection = _Connection([{"version_no": 4}])
    monkeypatch.setattr(v2_ingest, "connect", lambda: connection)
    monkeypatch.setattr(v2_ingest, "publish_v2_index", lambda *_args, **_kwargs: None)

    result = v2_ingest.index_v2_document(
        "# 标题\n\n正文".encode(),
        title="说明.md",
        collection_name="资料",
        embedder=_Embedder(),
        embedding_dimension=3,
        parser=MarkdownParser(),
    )

    assert result.version_id
    assert connection.cursor.params[0][0].lstrip().startswith("SELECT version_no")


def test_index_v2_document_assigns_next_version_number_for_new_content(monkeypatch):
    connection = _Connection([None, {"version_no": 4}])
    captured = {}
    monkeypatch.setattr(v2_ingest, "connect", lambda: connection)
    monkeypatch.setattr(
        v2_ingest,
        "publish_v2_index",
        lambda _conn, plan, **_kwargs: captured.setdefault(
            "version", plan.documents[0].version.version_no
        ),
    )

    v2_ingest.index_v2_document(
        "# 标题\n\n新正文".encode(),
        title="说明.md",
        collection_name="资料",
        embedder=_Embedder(),
        embedding_dimension=3,
        parser=MarkdownParser(),
    )

    assert captured["version"] == 5
