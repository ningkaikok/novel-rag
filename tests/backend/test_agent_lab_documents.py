"""Agent Lab 的 search_documents 工具：检索 V2 知识库里的通用文档（非小说）。

不连真实 PostgreSQL、不调真实 embedding 模型——V2KnowledgeRetriever 和
postgres.connect 都用假对象替换，只验证：RRF 融合去重、V2 命中转换成
SourceChunk 的字段映射、文档标题模糊解析、以及 execute() 的分发。
"""

import pytest

import agent_lab
from domain_models import Collection, Document, DocumentChunk, DocumentVersion, RetrievalScope
from v2_retrieval import V2SearchHit
from v2_source_adapter import fuse_v2_hits, source_chunk_from_v2_hit


def _hit(
    chunk_id: str, ordinal: int, text: str, distance: float, section_path=()
) -> V2SearchHit:
    collection = Collection(id="c1", name="我的知识库")
    document = Document(
        id="d1", collection_id="c1", title="方案C历史推演", source_type="markdown"
    )
    version = DocumentVersion(
        id="v1", document_id="d1", version_no=1, parser_name="markdown", parser_version="1"
    )
    chunk = DocumentChunk(
        id=chunk_id,
        document_version_id="v1",
        ordinal=ordinal,
        text=text,
        section_path=section_path,
    )
    return V2SearchHit(
        collection=collection,
        document=document,
        version=version,
        chunk=chunk,
        distance=distance,
    )


def test_source_chunk_from_v2_hit_maps_fields():
    """document 标题当 novel、ordinal 当 chunk_id、章节路径拼成标题。"""
    hit = _hit(
        "chunk-1", 3, "2022 年决定 2023 年入学的原因", 0.1, section_path=("背景", "决策")
    )

    source = source_chunk_from_v2_hit(hit)

    assert source.novel == "方案C历史推演"
    assert source.chunk_id == 3
    assert source.text == "2022 年决定 2023 年入学的原因"
    assert source.chapter_title == "背景/决策"
    assert source.distance == 0.1


def test_source_chunk_from_v2_hit_empty_section_path_is_none():
    hit = _hit("chunk-1", 0, "正文", 0.2, section_path=())
    assert source_chunk_from_v2_hit(hit).chapter_title is None


def test_fuse_v2_hits_dedupes_by_chunk_id_and_boosts_agreement():
    """同一 chunk 同时出现在向量和关键词候选里时，RRF 分数应该更高、排在前面。"""
    shared = _hit("shared", 0, "共同命中", 0.05)
    vector_only = _hit("vec-only", 1, "只有向量命中", 0.2)
    keyword_only = _hit("kw-only", 2, "只有关键词命中", 0.3)

    fused = fuse_v2_hits([shared, vector_only], [shared, keyword_only])

    ids = [hit.chunk.id for hit in fused]
    assert ids[0] == "shared", "两路都命中的候选应该排第一"
    assert set(ids) == {"shared", "vec-only", "kw-only"}, "去重后不应该出现重复的 chunk"


class _FakeRetriever:
    """替身 V2KnowledgeRetriever：不连数据库/embedding，返回预设候选。"""

    def __init__(self, *, vector_hits, keyword_hits):
        self._vector_hits = vector_hits
        self._keyword_hits = keyword_hits
        self.calls: list[tuple[str, RetrievalScope | None, int]] = []

    def __call__(self, embedder):  # 模拟 V2KnowledgeRetriever(embedder=...) 的构造签名
        return self

    def search(self, query, *, scope=None, top_k=5, exclude_legacy_novels=False):
        self.calls.append(("search", scope, top_k, exclude_legacy_novels))
        return self._vector_hits[:top_k]

    def keyword_search(self, query, *, scope=None, top_k=5, exclude_legacy_novels=False):
        self.calls.append(("keyword_search", scope, top_k, exclude_legacy_novels))
        return self._keyword_hits[:top_k]


def test_search_documents_fuses_vector_and_keyword_results(monkeypatch):
    vector_hits = [_hit("a", 0, "向量命中片段", 0.1)]
    keyword_hits = [_hit("b", 1, "关键词命中片段", 0.2)]
    fake = _FakeRetriever(vector_hits=vector_hits, keyword_hits=keyword_hits)
    monkeypatch.setattr(agent_lab, "V2KnowledgeRetriever", fake)

    class _FakeRag:
        embedder = object()

    toolbox = agent_lab.AgentToolbox(rag=_FakeRag())
    result = toolbox.search_documents("2022 年的决定")

    assert result.facts["kind"] == "document_passages"
    assert result.facts["coverage"] == "partial"
    assert {source.text for source in result.sources} == {"向量命中片段", "关键词命中片段"}
    # 没传 document 参数时不应该限定检索范围，但必须排除小说的影子副本——
    # 否则体量大得多的小说语料会把真正的文档挤出候选榜（实测复现过）。
    assert all(scope is None and exclude for _name, scope, _k, exclude in fake.calls)


def test_search_documents_respects_limit(monkeypatch):
    hits = [_hit(str(i), i, f"片段{i}", float(i)) for i in range(6)]
    fake = _FakeRetriever(vector_hits=hits, keyword_hits=[])
    monkeypatch.setattr(agent_lab, "V2KnowledgeRetriever", fake)

    class _FakeRag:
        embedder = object()

    toolbox = agent_lab.AgentToolbox(rag=_FakeRag())
    result = toolbox.search_documents("问题", limit=3)

    assert len(result.sources) == 3


class _FakeDocConn:
    """假的 postgres 连接：只回一份固定的文档标题列表。"""

    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, *_args):
        return self

    def fetchall(self):
        return self._rows


def test_resolve_document_scope_exact_match(monkeypatch):
    rows = [{"id": "d1", "title": "方案C历史推演"}, {"id": "d2", "title": "读书笔记"}]
    monkeypatch.setattr(agent_lab, "connect", lambda: _FakeDocConn(rows))
    toolbox = agent_lab.AgentToolbox(rag=None)

    scope = toolbox._resolve_document_scope("方案C历史推演")

    assert scope == RetrievalScope(document_id="d1")


def test_resolve_document_scope_fuzzy_single_match(monkeypatch):
    rows = [{"id": "d1", "title": "方案C历史推演_2022决定2023入学"}]
    monkeypatch.setattr(agent_lab, "connect", lambda: _FakeDocConn(rows))
    toolbox = agent_lab.AgentToolbox(rag=None)

    scope = toolbox._resolve_document_scope("方案C历史推演")

    assert scope == RetrievalScope(document_id="d1")


def test_resolve_document_scope_ambiguous_raises(monkeypatch):
    rows = [{"id": "d1", "title": "方案C历史推演"}, {"id": "d2", "title": "方案C补充材料"}]
    monkeypatch.setattr(agent_lab, "connect", lambda: _FakeDocConn(rows))
    toolbox = agent_lab.AgentToolbox(rag=None)

    with pytest.raises(ValueError, match="无法唯一确定文档"):
        toolbox._resolve_document_scope("方案C")


def test_execute_dispatches_search_documents(monkeypatch):
    calls = []

    class _Toolbox(agent_lab.AgentToolbox):
        def search_documents(self, query, document=None, limit=5):
            calls.append((query, document, limit))
            return agent_lab.ToolResult("ok")

    toolbox = _Toolbox(rag=None)
    toolbox.execute("search_documents", {"query": "问题", "document": "方案C", "limit": 3})

    assert calls == [("问题", "方案C", 3)]
