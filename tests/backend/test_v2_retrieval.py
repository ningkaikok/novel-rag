from domain_models import (
    Collection,
    Document,
    DocumentChunk,
    DocumentVersion,
    RetrievalScope,
)
from v2_retrieval import V2KnowledgeRetriever, V2ReadRepository, V2SearchHit


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


def test_vector_search_can_exclude_legacy_novel_shadow_copies():
    """search_documents（Agent Lab）需要排掉小说经 LegacyNovelAdapter 影子发布
    进 V2 的那部分——小说语料体量大得多，混在一起会把真正的 V2 文档挤出候选榜。
    """
    connection = _Connection([_row()])
    repository = V2ReadRepository(lambda: connection)

    repository.vector_search([0.1, 0.2], top_k=3, exclude_legacy_novels=True)

    assert "metadata->>'legacy_novel' IS NULL" in connection.cursor.query


def test_vector_search_without_exclusion_flag_does_not_filter_legacy_novels():
    """默认行为不变：V2ShadowReader 等既有调用方仍要能看到小说的影子副本。"""
    connection = _Connection([_row()])
    repository = V2ReadRepository(lambda: connection)

    repository.vector_search([0.1, 0.2], top_k=3)

    assert "legacy_novel" not in connection.cursor.query


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


class _StubEmbedder:
    def encode(self, questions, *, normalize_embeddings, show_progress_bar=False):
        assert normalize_embeddings is True
        return [[0.1, 0.2]]


def test_v2_native_retrieve_returns_empty_when_disabled(monkeypatch):
    """开关默认关闭：零开销直接返回空列表，不该碰任何 repository/DB。"""
    import retrieval_mixins

    def _boom():
        raise AssertionError("关闭时不应该构造 V2ReadRepository")

    monkeypatch.setattr(retrieval_mixins, "V2_NATIVE_RETRIEVAL_ENABLED", False)
    monkeypatch.setattr(retrieval_mixins, "V2ReadRepository", _boom)
    mixin = retrieval_mixins.RetrievalMixin()
    mixin.embedder = _StubEmbedder()

    assert mixin.v2_native_retrieve("问题", top_k=5) == []


def test_v2_native_retrieve_supports_deterministic_gray_rollout(monkeypatch):
    """灰度 100%/0% 必须分别等价于开启/关闭，且不依赖随机数。"""
    import retrieval_mixins

    monkeypatch.setattr(retrieval_mixins, "V2_NATIVE_RETRIEVAL_ENABLED", False)
    monkeypatch.setattr(retrieval_mixins, "V2_NATIVE_RETRIEVAL_MODE", "gray")
    monkeypatch.setattr(retrieval_mixins, "V2_NATIVE_RETRIEVAL_GRAY_PERCENT", 0)
    mixin = retrieval_mixins.RetrievalMixin()
    mixin.embedder = _StubEmbedder()
    monkeypatch.setattr(
        retrieval_mixins,
        "V2ReadRepository",
        lambda: (_ for _ in ()).throw(AssertionError("0% 灰度不应访问 V2")),
    )
    assert mixin.v2_native_retrieve("问题", rollout_key="session-a") == []

    monkeypatch.setattr(retrieval_mixins, "V2_NATIVE_RETRIEVAL_GRAY_PERCENT", 100)
    monkeypatch.setattr(retrieval_mixins, "V2ReadRepository", lambda: _Connection([]))
    assert mixin.v2_native_retrieve("问题", rollout_key="session-a") == []


def test_v2_native_retrieve_degrades_to_empty_on_repository_error(monkeypatch):
    """V2 只是锦上添花的一路召回：DB/embedding 异常绝不能拖垮 V1 主链路。"""
    import retrieval_mixins

    class _BrokenRepository:
        def vector_search(self, *_args, **_kwargs):
            raise RuntimeError("db down")

    monkeypatch.setattr(retrieval_mixins, "V2_NATIVE_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(retrieval_mixins, "V2ReadRepository", _BrokenRepository)
    mixin = retrieval_mixins.RetrievalMixin()
    mixin.embedder = _StubEmbedder()

    assert mixin.v2_native_retrieve("问题", top_k=5) == []


def test_v2_native_retrieve_merges_dedups_and_sorts_by_fused_rank(monkeypatch):
    """向量和关键词可能命中同一个片段：按两路排名融合、按 chunk id 去重、截到 top_k。"""
    import retrieval_mixins

    connection = _Connection(
        [_row(distance=0.5), {**_row(distance=-3.0), "chunk_id": "ch-2", "ordinal": 1}]
    )
    keyword_connection = _Connection(
        [{**_row(distance=-3.0), "chunk_id": "ch-2", "ordinal": 1}]
    )

    class _FakeRepository:
        def vector_search(self, *_args, **_kwargs):
            return V2ReadRepository(lambda: connection).vector_search([0.1, 0.2], top_k=5)

        def keyword_search(self, *_args, **_kwargs):
            return V2ReadRepository(lambda: keyword_connection).keyword_search(
                ["检索"], top_k=5
            )

    monkeypatch.setattr(retrieval_mixins, "V2_NATIVE_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(retrieval_mixins, "V2ReadRepository", _FakeRepository)
    mixin = retrieval_mixins.RetrievalMixin()
    mixin.embedder = _StubEmbedder()

    sources = mixin.v2_native_retrieve("检索", top_k=5)

    # ch-2（distance -3.0）在向量和关键词两路都命中，去重后只保留一份，且排在最前面。
    assert [s.chunk_id for s in sources] == [1, 0]
    assert all(s.origin == "v2_document" for s in sources)


# ------------------------------------------------- V2 双路排名融合（RRF）


def _hit(chunk_id: str, ordinal: int, text: str, distance: float) -> V2SearchHit:
    collection = Collection(id="c-1", name="知识库")
    document = Document(
        id="d-1", collection_id="c-1", title="设计文档", source_type="markdown"
    )
    version = DocumentVersion(
        id="v-1", document_id="d-1", version_no=1, parser_name="markdown", parser_version="1"
    )
    chunk = DocumentChunk(id=chunk_id, document_version_id="v-1", ordinal=ordinal, text=text)
    return V2SearchHit(
        collection=collection,
        document=document,
        version=version,
        chunk=chunk,
        distance=distance,
    )


class _RankedRepository:
    """两路各返回一份已排好序的候选（顺序就是各自的排名）。"""

    def __init__(self, vector_hits=(), keyword_hits=()):
        self.vector_hits = list(vector_hits)
        self.keyword_hits = list(keyword_hits)

    def vector_search(self, *_args, **_kwargs):
        return list(self.vector_hits)

    def keyword_search(self, *_args, **_kwargs):
        return list(self.keyword_hits)


def _fusion_mixin(repository, monkeypatch):
    import retrieval_mixins

    monkeypatch.setattr(retrieval_mixins, "V2_NATIVE_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(retrieval_mixins, "V2ReadRepository", lambda: repository)
    mixin = retrieval_mixins.RetrievalMixin()
    mixin.embedder = _StubEmbedder()
    return mixin


def test_v2_native_retrieve_fuses_by_rank_not_raw_score(monkeypatch):
    """融合必须看"在各路排第几"，不能拿负 BM25 和余弦距离直接比大小。

    场景：向量路 A 第 1、B 第 2；关键词路 B 第 1、C 第 2（A 没被关键词命中）。
    旧实现把两个列表拼起来按 ``distance`` 排序，负 BM25 的 B（-8.0）和 C（-1.0）
    无条件排在余弦距离的 A（0.1）前面——即使 A 是向量路第一名。
    RRF 正确结果：B 两路都命中得两份贡献排第一，A（向量第 1）第二，C 第三。
    """
    repository = _RankedRepository(
        vector_hits=[_hit("A", 0, "向量第一", 0.1), _hit("B", 1, "向量第二", 0.15)],
        keyword_hits=[_hit("B", 1, "关键词第一", -8.0), _hit("C", 2, "关键词第二", -1.0)],
    )
    mixin = _fusion_mixin(repository, monkeypatch)

    sources = mixin.v2_native_retrieve("检索", top_k=5)

    assert [s.chunk_uid for s in sources] == ["B", "A", "C"]


def test_v2_native_retrieve_is_invariant_to_bm25_magnitude(monkeypatch):
    """改变不影响排名的 BM25 分数绝对值，最终排名不能变。

    关键是分两次取结果：如果先把 baseline 和 scaled 两个仓库都造好再分别调用，
    后一次 monkeypatch 会覆盖模块里的 ``V2ReadRepository``，两次调用实际都走
    scaled 仓库——那就成了自己和自己比（永远相等，什么也没验证）。
    """
    vector_hits = [_hit("A", 0, "向量第一", 0.1), _hit("B", 1, "向量第二", 0.15)]
    baseline = _fusion_mixin(
        _RankedRepository(
            vector_hits=vector_hits,
            keyword_hits=[_hit("B", 1, "关键词第一", -8.0), _hit("C", 2, "关键词第二", -1.0)],
        ),
        monkeypatch,
    )
    baseline_order = [s.chunk_uid for s in baseline.v2_native_retrieve("检索", top_k=5)]

    scaled = _fusion_mixin(
        _RankedRepository(
            vector_hits=vector_hits,
            keyword_hits=[
                _hit("B", 1, "关键词第一", -800.0),
                _hit("C", 2, "关键词第二", -0.01),
            ],
        ),
        monkeypatch,
    )
    scaled_order = [s.chunk_uid for s in scaled.v2_native_retrieve("检索", top_k=5)]

    assert baseline_order == ["B", "A", "C"], "两路名次决定的融合顺序"
    assert scaled_order == baseline_order, "BM25 绝对值缩放不改变最终排名"


def test_v2_native_retrieve_boosts_hits_found_by_both_routes(monkeypatch):
    """双路共同命中的候选要拿到融合加分，能反超单路第一名。"""
    repository = _RankedRepository(
        vector_hits=[
            _hit("X", 0, "只在向量命中且排名第一", 0.1),
            _hit("S", 1, "共同命中", 0.2),
        ],
        keyword_hits=[_hit("S", 1, "共同命中", -0.5)],
    )
    mixin = _fusion_mixin(repository, monkeypatch)

    sources = mixin.v2_native_retrieve("检索", top_k=5)

    assert [s.chunk_uid for s in sources] == ["S", "X"]
    assert [s.text for s in sources] == ["共同命中", "只在向量命中且排名第一"]


def test_v2_native_retrieve_handles_single_empty_route(monkeypatch):
    """单路为空时另一路的排名原样保留；两路都空才返回空。"""
    keyword_only = _fusion_mixin(
        _RankedRepository(
            keyword_hits=[_hit("K1", 0, "关键词1", -5.0), _hit("K2", 1, "关键词2", -1.0)]
        ),
        monkeypatch,
    )
    assert [s.chunk_uid for s in keyword_only.v2_native_retrieve("检索", top_k=5)] == [
        "K1",
        "K2",
    ]

    vector_only = _fusion_mixin(
        _RankedRepository(
            vector_hits=[_hit("V1", 0, "向量1", 0.1), _hit("V2", 1, "向量2", 0.2)]
        ),
        monkeypatch,
    )
    assert [s.chunk_uid for s in vector_only.v2_native_retrieve("检索", top_k=5)] == [
        "V1",
        "V2",
    ]

    empty = _fusion_mixin(_RankedRepository(), monkeypatch)
    assert empty.v2_native_retrieve("检索", top_k=5) == []


def test_v2_native_retrieve_dedups_stably_and_applies_top_k(monkeypatch):
    """融合后先稳定去重，再应用 top_k；重复调用结果一致。"""
    repository = _RankedRepository(
        vector_hits=[_hit("A", 0, "A", 0.1), _hit("B", 1, "B", 0.2)],
        keyword_hits=[_hit("A", 0, "A", -3.0), _hit("B", 1, "B", -1.0)],
    )
    mixin = _fusion_mixin(repository, monkeypatch)

    first = mixin.v2_native_retrieve("检索", top_k=5)
    second = mixin.v2_native_retrieve("检索", top_k=5)
    limited = mixin.v2_native_retrieve("检索", top_k=1)

    assert [s.chunk_uid for s in first] == ["A", "B"]
    assert [s.chunk_uid for s in second] == ["A", "B"], "同一输入必须给出同一排序"
    assert [s.chunk_uid for s in limited] == ["A"]


def test_v2_native_retrieve_keeps_surviving_route_when_one_route_fails(monkeypatch):
    """单路下游失败不应把另一路已经拿到的候选一起丢掉；两路都失败才降级为空。"""

    class _KeywordBroken:
        def vector_search(self, *_args, **_kwargs):
            return [_hit("V", 0, "向量命中", 0.1)]

        def keyword_search(self, *_args, **_kwargs):
            raise RuntimeError("bm25 down")

    class _BothBroken:
        def vector_search(self, *_args, **_kwargs):
            raise RuntimeError("vector down")

        def keyword_search(self, *_args, **_kwargs):
            raise RuntimeError("bm25 down")

    survivor = _fusion_mixin(_KeywordBroken(), monkeypatch)
    assert [s.chunk_uid for s in survivor.v2_native_retrieve("检索", top_k=5)] == ["V"]

    broken = _fusion_mixin(_BothBroken(), monkeypatch)
    assert broken.v2_native_retrieve("检索", top_k=5) == []
