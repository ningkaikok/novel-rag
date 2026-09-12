from domain_models import RetrievalScope
from v2_retrieval import V2KnowledgeRetriever, V2ReadRepository


class _Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.query = ""
        self.params = []

    def execute(self, query, params=()):
        self.query = query
        self.params = list(params)
        return self

    def fetchall(self):
        return self.rows


class _Connection:
    def __init__(self, rows):
        self.cursor = _Cursor(rows)

    def execute(self, query, params=()):
        return self.cursor.execute(query, params)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _row(*, distance=0.2):
    return {
        "collection_id": "c-1",
        "collection_name": "资料库",
        "collection_description": "测试",
        "document_id": "d-1",
        "document_title": "设计文档",
        "source_type": "markdown",
        "document_metadata": {"team": "ai"},
        "version_id": "v-1",
        "version_no": 1,
        "source_hash": "sha-1",
        "parser_name": "markdown",
        "parser_version": "1",
        "chunk_id": "ch-1",
        "ordinal": 0,
        "section_path": ["概览"],
        "page_number": None,
        "text": "一段知识库内容",
        "context": "文档概览",
        "chunk_metadata": {"source": "test"},
        "distance": distance,
    }


def test_vector_search_returns_generic_identity_and_applies_scope():
    connection = _Connection([_row()])
    repository = V2ReadRepository(lambda: connection)

    hits = repository.vector_search(
        [0.1, 0.2],
        top_k=3,
        scope=RetrievalScope(collection_id="c-1", document_id="d-1"),
    )

    assert len(hits) == 1
    assert hits[0].document.title == "设计文档"
    assert hits[0].chunk.section_path == ("概览",)
    assert hits[0].distance == 0.2
    assert "knowledge_v2.document_chunks" in connection.cursor.query
    assert connection.cursor.params == ["[0.1,0.2]", "c-1", "d-1", "[0.1,0.2]", 3]


def test_keyword_search_builds_scoped_bm25_query():
    connection = _Connection(
        [_row(distance=-1.5), {**_row(distance=-0.5), "chunk_id": "ch-2"}]
    )
    repository = V2ReadRepository(lambda: connection)

    hits = repository.keyword_search(
        ["检索", "检索"],
        top_k=1,
        scope=RetrievalScope(version_id="v-1", document_id="d-1"),
    )

    assert [hit.chunk.id for hit in hits] == ["ch-1"]
    assert "knowledge_v2.chunk_terms" in connection.cursor.query
    assert "LN(" in connection.cursor.query
    assert connection.cursor.params[:1] == ["检索"]


def test_knowledge_retriever_uses_embedder_and_returns_chunks():
    class _Embedder:
        def encode(self, questions, *, normalize_embeddings):
            assert questions == ["问题"]
            assert normalize_embeddings is True
            return [[0.3, 0.4]]

    connection = _Connection([_row()])
    retriever = V2KnowledgeRetriever(_Embedder(), V2ReadRepository(lambda: connection))

    chunks = retriever.retrieve("问题", top_k=1)

    assert [chunk.id for chunk in chunks] == ["ch-1"]
