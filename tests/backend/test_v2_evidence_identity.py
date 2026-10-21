"""V2 文档证据的身份与扩展边界（#1 修复的回归测试）。

背景：V2 命中被投影成 ``SourceChunk`` 时，``novel``/``chunk_id`` 只是**展示投影**
（文档标题 + 片段序号），并不是真正的主键。V1 小说的主键是
``novel_chunks(novel, chunk_id)``，V2 文档的主键是 ``knowledge_v2`` 的 UUID——
两套 ID 空间完全独立。

旧实现的三个下游都拿展示投影当身份键，于是只要出现"一份上传文档和一本小说同名"，
就会在完全静默的情况下出错：

1. RRF 融合：同名同序号的 V1/V2 候选被当成同一段证据，其中一个的正文被丢掉；
2. 邻居 / 整章扩展：V2 命中被拿去查 ``novel_chunks``——查到同名小说就**替换**
   正文（最终 prompt 里的证据变成另一本书的内容），查不到就整个丢掉；
3. Agent 证据注册：同名的两份来源共用一个 ``S<n>`` 编号，``answer_with_citations``
   只能引用到其中一份。

这里逐条覆盖：V1/V2 混合、仅 V2、V1/V2 同名同序号、多个 V2 命中，以及 neighbors /
chapter 两种扩展模式。全部 mock，不连数据库、不加载模型。
"""

import types
from unittest.mock import patch

import agent_lab
import rag
import retrieval_mixins
from chunk_model import SourceChunk, _trace_candidates
from domain_models import Collection, Document, DocumentChunk, DocumentVersion
from v2_retrieval import V2SearchHit

SAME_TITLE = "雾隐山庄.txt"


# ---------------------------------------------------------------- 构造工具
def _v2_hit(
    *,
    text: str,
    ordinal: int = 0,
    chunk_id: str = "ch-1",
    document_id: str = "doc-1",
    title: str = SAME_TITLE,
    section_path: tuple[str, ...] = (),
) -> V2SearchHit:
    document = Document(
        id=document_id,
        collection_id="c-1",
        title=title,
        source_type="markdown",
    )
    version = DocumentVersion(
        id=f"{document_id}-v1",
        document_id=document_id,
        version_no=1,
        parser_name="markdown",
        parser_version="1",
    )
    chunk = DocumentChunk(
        id=chunk_id,
        document_version_id=version.id,
        ordinal=ordinal,
        text=text,
        section_path=section_path,
    )
    return V2SearchHit(
        collection=Collection(id="c-1", name="知识库"),
        document=document,
        version=version,
        chunk=chunk,
        distance=0.1,
    )


def _v1_source(chunk_id: int = 0, text: str = "V1小说正文", chapter="第一章") -> SourceChunk:
    return SourceChunk(
        novel=SAME_TITLE,
        chunk_id=chunk_id,
        text=text,
        distance=0.2,
        chapter_title=chapter,
    )


class _LegacyConn:
    """假的 ``novel_chunks`` 连接：只认邻居区间查询和整章查询两种 SQL。"""

    def __init__(self, rows):
        self.rows = rows
        self.queries: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        self.queries.append(normalized)
        if "chapter_title = %s" in normalized:
            novel, title = params
            matched = [
                row
                for row in self.rows
                if row["novel"] == novel and row["chapter_title"] == title
            ]
        elif "BETWEEN" in normalized:
            flat = list(params or [])
            matched = []
            seen: set[tuple[str, int]] = set()
            for index in range(0, len(flat), 3):
                novel, low, high = flat[index : index + 3]
                for row in self.rows:
                    key = (row["novel"], row["chunk_id"])
                    if (
                        row["novel"] == novel
                        and low <= row["chunk_id"] <= high
                        and key not in seen
                    ):
                        seen.add(key)
                        matched.append(row)
        else:
            raise AssertionError(f"未预期的 SQL：{normalized}")
        matched = sorted(matched, key=lambda row: (row["novel"], row["chunk_id"]))
        return types.SimpleNamespace(fetchall=lambda: [dict(row) for row in matched])


def _legacy_rows(ids, *, text_prefix="V1正文", chapter="第一章"):
    return [
        {
            "novel": SAME_TITLE,
            "chunk_id": chunk_id,
            "chapter_title": chapter,
            "text": f"{text_prefix}-{chunk_id}",
            "context": "",
        }
        for chunk_id in ids
    ]


def _mixin(embedder=None):
    service = object.__new__(retrieval_mixins.RetrievalMixin)
    service.embedder = embedder or _BareEmbedder()
    return service


class _BareEmbedder:
    """没有 tokenizer 的替身：整章档会退化成「闸门未执行」，不影响身份断言。"""


class _CharTokenizer:
    """假 tokenizer：1 个字符 = 1 个 token，让预算闸门有确定性计数。"""

    def __call__(self, text, padding=False, truncation=False, add_special_tokens=True):
        return {"input_ids": [1] * len(text)}


class _CharEmbedder:
    """带 tokenizer 的假 embedding 模型。"""

    tokenizer = _CharTokenizer()


# ---------------------------------------------------------------- 身份键
def test_v2_hit_identity_does_not_collide_with_same_titled_v1_chunk():
    """同名同序号的 V1/V2 证据必须是两条不同身份，且各自保留自己的正文与来源。"""
    v1 = _v1_source(0, "V1小说正文")
    v2 = SourceChunk.from_v2_hit(_v2_hit(text="V2文档正文", ordinal=0))

    assert v1.novel == v2.novel, "场景前提：展示标题/序号刻意相同"
    assert v1.chunk_id == v2.chunk_id
    assert v1.identity_key != v2.identity_key, "身份键必须带来源域，不能只用展示字段"

    assert v2.text == "V2文档正文"
    assert v2.origin == "v2_document"
    assert v2.document_id == "doc-1"
    assert v2.chunk_uid == "ch-1"
    assert v1.is_legacy_novel and not v2.is_legacy_novel


def test_identity_key_distinguishes_two_documents_with_same_title():
    """两份同名文档：document_id 不同就是不同证据，不能按标题合并。"""
    first = SourceChunk.from_v2_hit(
        _v2_hit(text="A 文档", ordinal=0, chunk_id="ch-a", document_id="doc-a")
    )
    second = SourceChunk.from_v2_hit(
        _v2_hit(text="B 文档", ordinal=0, chunk_id="ch-b", document_id="doc-b")
    )

    assert first.identity_key != second.identity_key


def test_trace_candidates_do_not_merge_same_titled_v1_and_v2():
    """trace 里的 previous_rank 也不能把同名 V1/V2 当成同一条候选。"""
    v1 = _v1_source(0)
    v2 = SourceChunk.from_v2_hit(_v2_hit(text="V2文档正文", ordinal=0))
    previous_ranks = {v1.identity_key: 1, v2.identity_key: 2}

    payload = _trace_candidates(
        [v1, v2],
        score_label="RRF",
        score_of=lambda _source, key: 1.0 if key == v1.identity_key else 2.0,
        previous_ranks=previous_ranks,
    )

    assert [item["origin"] for item in payload] == ["legacy_novel", "v2_document"]
    assert [item["previous_rank"] for item in payload] == [1, 2]


# ---------------------------------------------------------------- 融合层
def _service_with_routes(semantic, v2_sources, monkeypatch):
    service = object.__new__(rag.NovelRAG)
    service.embedder = object()
    monkeypatch.setattr(service, "_named_novels", lambda _question: [])
    monkeypatch.setattr(service, "_full_text_chunks", lambda _novels: None)
    monkeypatch.setattr(service, "retrieve", lambda _q, top_k, only_novels: semantic)
    monkeypatch.setattr(service, "keyword_retrieve", lambda _q, top_k, only_novels: [])
    monkeypatch.setattr(service, "positional_retrieve", lambda *a, **k: [])
    monkeypatch.setattr(service, "v2_native_retrieve", lambda _q, top_k: v2_sources)
    monkeypatch.setattr(rag, "HIERARCHY_ENABLED", False)
    monkeypatch.setattr(rag, "RERANK_ENABLED", False)
    monkeypatch.setattr(rag, "V2_NATIVE_RETRIEVAL_ENABLED", True)
    return service


def test_rrf_keeps_both_sides_when_v1_and_v2_share_title_and_ordinal(monkeypatch):
    """同名同序号不能让 RRF 把两条证据合并成一条。"""
    v1 = _v1_source(0, "V1小说正文")
    v2 = SourceChunk.from_v2_hit(_v2_hit(text="V2文档正文", ordinal=0))
    service = _service_with_routes([v1], [v2], monkeypatch)

    events = list(service.retrieve_hybrid_stream("庄主是谁", top_k=2))
    result = events[-1][1]
    rrf_step = next(
        payload
        for kind, payload in events
        if kind == "step" and payload.get("stage_key") == "rrf"
    )

    assert [source.text for source in result] == ["V1小说正文", "V2文档正文"]
    assert {source.origin for source in result} == {"legacy_novel", "v2_document"}
    assert len(rrf_step["candidates"]) == 2, "RRF 候选池里应该是两条，不是被合并后的一条"


def test_answer_context_and_prompt_keep_v2_text_with_same_named_v1_neighbors(monkeypatch):
    """最终进入 prompt 的证据必须仍含 V2 正文，不能被同名小说的 V1 邻居替换掉。

    这是 #1 里最严重的表现：扩展阶段把 V2 命中拿去查 ``novel_chunks``，
    查到同名小说就把 V1 正文写回去，prompt 里的"证据"变成另一本书的内容，
    而模型完全无从察觉。
    """
    v1 = _v1_source(1, "V1小说正文-1")
    v2 = SourceChunk.from_v2_hit(_v2_hit(text="V2文档正文", ordinal=1))
    service = _service_with_routes([v1], [v2], monkeypatch)
    conn = _LegacyConn(_legacy_rows([0, 1, 2]))

    events = list(service.retrieve_hybrid_stream("庄主是谁", top_k=2))
    sources = events[-1][1]

    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MODE", "neighbors")
    with patch.object(retrieval_mixins, "connect", lambda: conn):
        context_sources, step = service.build_answer_context(sources)
        prompt = service.build_prompt("庄主是谁", context_sources)

    assert step is None
    v2_entries = [s for s in context_sources if s.origin == "v2_document"]
    assert len(v2_entries) == 1, "V2 命中不能在扩展后消失"
    assert v2_entries[0].text == "V2文档正文", "V2 命中不能被同名小说的 V1 正文替换"
    assert v2_entries[0].chunk_uid == "ch-1"
    # V1 命中的正常邻居扩展仍然生效（0 和 2 是命中 1 的邻居）
    assert [s.chunk_id for s in context_sources if s.is_legacy_novel] == [0, 1, 2]
    assert "V2文档正文" in prompt, "最终 prompt 里必须还有 V2 原文"
    assert "V1正文-0" in prompt


def test_expand_neighbors_v2_only_sources_are_returned_untouched():
    """只有 V2 命中时，邻居扩展不查 V1 表，原样返回（正文/来源/顺序都不变）。"""
    service = _mixin()
    first = SourceChunk.from_v2_hit(_v2_hit(text="第一段", ordinal=0, chunk_id="ch-1"))
    second = SourceChunk.from_v2_hit(_v2_hit(text="第二段", ordinal=1, chunk_id="ch-2"))
    conn = _LegacyConn([])

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        expanded = service.expand_neighbors([first, second])

    assert expanded == [first, second]
    assert [s.text for s in expanded] == ["第一段", "第二段"]
    assert conn.queries == [], "纯 V2 结果不应该产生任何 novel_chunks 查询"


def test_expand_neighbors_still_expands_normal_v1_hits(monkeypatch):
    """回归：纯 V1 命中仍然按 CONTEXT_NEIGHBORS 正常扩邻居，顺序与正文不变。"""
    service = _mixin()
    conn = _LegacyConn(_legacy_rows([4, 5, 6, 7]))

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        expanded = service.expand_neighbors([_v1_source(5, "V1正文-5")])

    assert [s.chunk_id for s in expanded] == [4, 5, 6]
    assert [s.text for s in expanded] == ["V1正文-4", "V1正文-5", "V1正文-6"]
    assert len(conn.queries) == 1, "V1 命中仍应发起一次邻居区间查询"


def test_expand_neighbors_keeps_multiple_v2_hits_in_relevance_order():
    """同一文档的多个 V2 命中都要保留，且保持检索相关性顺序。"""
    service = _mixin()
    hits = [
        SourceChunk.from_v2_hit(_v2_hit(text=f"片段{i}", ordinal=i, chunk_id=f"ch-{i}"))
        for i in (2, 0, 1)
    ]
    conn = _LegacyConn([])

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        expanded = service.expand_neighbors(hits, neighbors=1)

    assert [s.text for s in expanded] == ["片段2", "片段0", "片段1"]


def test_chapter_mode_does_not_pull_v1_chapter_for_v2_hits(monkeypatch):
    """整章档：V2 命中（哪怕有同名章节）不能被换成一整章 V1 正文。"""
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MODE", "chapter")
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MAX_TOKENS", 10_000)
    conn = _LegacyConn(_legacy_rows([0, 1, 2]))
    service = _mixin()
    v2 = SourceChunk.from_v2_hit(
        _v2_hit(text="V2文档正文", ordinal=1, section_path=("第一章",))
    )
    sources = [v2, _v1_source(1, "V1正文-1")]

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        result, step = service.build_answer_context(sources)

    assert step["expansion_mode"] == "chapter"
    v2_entries = [s for s in result if s.origin == "v2_document"]
    assert len(v2_entries) == 1
    assert v2_entries[0].text == "V2文档正文"
    # 组间顺序沿用相关性顺序：V2 命中在检索里排第一，就该排第一
    assert result[0].origin == "v2_document"
    # V1 命中照常按整章补齐
    assert [s.chunk_id for s in result if s.is_legacy_novel] == [0, 1, 2]


# ---------------------------------------------------------------- Agent 证据注册
class _NoDocsConn:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, *_args):
        self._sql = str(_args[0]) if _args else ""
        return self

    def fetchone(self):
        return {"name": "knowledge_v2.documents"}

    def fetchall(self):
        return []


class _MixedToolbox:
    """一次返回两条"同名同序号"的证据：一条 V1、一条 V2。"""

    def __init__(self, _rag):
        pass

    def execute(self, name, _args):
        assert name == "search_novels"
        return agent_lab.ToolResult(
            "找到两段证据",
            [
                _v1_source(0, "V1小说正文"),
                SourceChunk.from_v2_hit(_v2_hit(text="V2文档正文", ordinal=0)),
            ],
        )


class _FakeRag:
    def build_prompt(self, question, sources, history=None, summary=None, facts_text=None):
        return question + "|" + ",".join(source.text for source in sources)


def test_agent_registers_same_titled_v1_and_v2_as_distinct_evidence(monkeypatch):
    """Agent 证据注册表不能把同名同序号的 V1/V2 合成一个 S 编号。"""
    import json

    monkeypatch.setattr(agent_lab, "connect", lambda: _NoDocsConn())
    monkeypatch.setattr(agent_lab, "AgentToolbox", _MixedToolbox)
    actions = iter(
        [
            {"reason": "先找原文", "tool": "search_novels", "args": {"query": "庄主是谁"}},
            {
                "reason": "证据足够",
                "tool": "answer_with_citations",
                "args": {"source_ids": ["S1", "S2"]},
            },
        ]
    )

    events = list(
        agent_lab.run_agent(
            "庄主是谁",
            rag=_FakeRag(),
            planner=lambda _prompt: json.dumps(next(actions), ensure_ascii=False),
            answerer=lambda _prompt: iter(["答案"]),
            max_steps=3,
        )
    )

    steps = [value for kind, value in events if kind == "agent_step"]
    sources = next(value for kind, value in events if kind == "sources")

    assert steps[0]["source_ids"] == ["S1", "S2"], "两条同名同序号证据应各自注册"
    assert [source.text for source in sources] == ["V1小说正文", "V2文档正文"]
    assert {source.origin for source in sources} == {"legacy_novel", "v2_document"}


# ------------------------------------------------- 上级复查发现的三个边界
def test_expand_neighbors_keeps_v1_hit_when_legacy_query_returns_nothing():
    """混合结果里 V1 邻居查询为空时，V1 命中本身不能消失。

    旧实现只在「整个 expanded 为空」时才退回原始列表，混合结果里只要有一路
    （比如 V2）能输出，这个全局兜底就永远不触发——查不到邻居的 V1 命中被静默丢掉。
    兜底必须按命中个体做：宁可没有邻居，也不能少一条证据。
    """
    service = _mixin()
    v1 = _v1_source(5, "V1小说正文-5")
    v2 = SourceChunk.from_v2_hit(_v2_hit(text="V2文档正文", ordinal=5))
    conn = _LegacyConn([])  # 索引正在重建：区间查询什么也没查到

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        expanded = service.expand_neighbors([v1, v2])

    assert [s.text for s in expanded] == ["V1小说正文-5", "V2文档正文"]
    assert [s.origin for s in expanded] == ["legacy_novel", "v2_document"]
    assert len(conn.queries) == 1, "V1 命中仍然发起了一次区间查询（只是没查到）"


def test_chapter_mode_empty_chapter_query_does_not_duplicate_same_titled_v2(monkeypatch):
    """章节查询为空时的退化列表只能包含 V1 命中，不能把同名 V2 也当章节成员。

    否则同一个 V2 片段会先作为"章节成员"输出一次、再作为透传单位输出一次，
    最终证据里出现两份一模一样的 V2 正文（提示词也被重复占用）。
    """
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MODE", "chapter")
    conn = _LegacyConn([])  # 章节查询什么都没查到 → 走退化兜底
    service = _mixin(embedder=_CharEmbedder())
    v1 = _v1_source(0, "V1小说正文", chapter="第一章")
    v2 = SourceChunk.from_v2_hit(
        _v2_hit(text="V2文档正文", ordinal=0, section_path=("第一章",))
    )

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        result, step = service.build_answer_context([v1, v2])

    assert [s.origin for s in result] == ["legacy_novel", "v2_document"]
    assert [s.text for s in result] == ["V1小说正文", "V2文档正文"]
    assert len(result) == 2, "同名 V1/V2 各一片，扩展后仍应是两片，不能变成三片"
    assert step["evidence_tokens"] == sum(len(s.text) for s in result)


def test_chapter_mode_missing_hit_fallback_accounts_for_actual_tokens(monkeypatch):
    """章节成员里找不到命中时的兜底分支：证据要真计费，超预算要如实报告。

    旧实现把这个分支的原文直接塞进结果、却完全不进 token 计数，trace 于是写成
    ``evidence_tokens=0, truncated=False``——明明带了 60 tokens 的原文进 prompt，
    界面和评测复盘看到的却是"零证据、没截断"。
    """
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MODE", "chapter")
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MAX_TOKENS", 20)
    # 章节存在，但查回来的成员里没有命中片段 99（索引与命中不一致）
    conn = _LegacyConn(_legacy_rows([10, 11]))
    service = _mixin(embedder=_CharEmbedder())
    hit = retrieval_mixins.SourceChunk(
        novel=SAME_TITLE,
        chunk_id=99,
        text="字" * 60,
        distance=0.0,
        chapter_title="第一章",
    )

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        result, step = service.build_answer_context([hit])

    assert [s.chunk_id for s in result] == [99]
    assert result[0].text == "字" * 60, "兜底必须原样保留命中正文"
    assert step["evidence_tokens"] == 60, "实际送进 prompt 的 60 tokens 必须计入"
    assert step["truncated"] is True
    assert "超预算" in (step["truncation_reason"] or "")


def test_chapter_mode_partial_chapter_refresh_keeps_every_original_hit(monkeypatch):
    """章节查询只回来一部分片段时，没回来的命中也要保留并统一计费。

    场景：命中 0 和 9，但章节查询只返回 0、1（增量重建中 / 片段被删改）。
    旧实现只在「整章一条都没查到」时兜底，部分缺失时 9 号命中会被静默丢掉——
    而它是检索真正找到的证据。现在按 identity 把该组未查到的命中补回成员列表，
    再按 chunk_id 排序后统一参与预算与输出。
    """
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MODE", "chapter")
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MAX_TOKENS", 30)
    rows = [
        {
            "novel": SAME_TITLE,
            "chunk_id": chunk_id,
            "chapter_title": "第一章",
            "text": "字" * 10,
            "context": "",
        }
        for chunk_id in (0, 1)
    ]
    conn = _LegacyConn(rows)
    service = _mixin(embedder=_CharEmbedder())
    sources = [
        retrieval_mixins.SourceChunk(
            novel=SAME_TITLE,
            chunk_id=0,
            text="字" * 10,
            distance=0.0,
            chapter_title="第一章",
        ),
        retrieval_mixins.SourceChunk(
            novel=SAME_TITLE,
            chunk_id=9,
            text="字" * 10,
            distance=0.1,
            chapter_title="第一章",
        ),
    ]

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        result, step = service.build_answer_context(sources)

    assert [s.chunk_id for s in result] == [0, 1, 9], "9 号命中不能被章节查询吞掉"
    assert step["evidence_tokens"] == sum(len(s.text) for s in result) == 30, (
        "trace 里的证据 token 必须等于实际输出的证据总量"
    )
    assert step["evidence_tokens"] <= 30
    # 该章全部成员（0、1、9）都在预算内，没有东西被丢掉 → 不应报假截断
    assert step["truncated"] is False
    assert step["truncation_reason"] is None


def test_expand_neighbors_partial_query_still_keeps_original_hit():
    """邻居区间查询只回来一部分时，连命中自己都要保留，且组内顺序按原文。

    场景：命中片段 1，但区间查询只返回了邻居 0 和 2（索引增量重建中）。
    旧实现只在「整组一条都没查到」时回填命中，组非空时命中 1 会被静默丢掉——
    而这正是检索真正找到的那一段。现在原始命中先按 (novel, chunk_id) 补进
    by_key（真实 DB 行优先），再按 chunk_id 顺序组装。
    """
    service = _mixin()
    hit = _v1_source(1, "V1命中正文")
    conn = _LegacyConn(
        [
            {
                "novel": SAME_TITLE,
                "chunk_id": chunk_id,
                "chapter_title": "第一章",
                "text": f"V1邻居-{chunk_id}",
                "context": "",
            }
            for chunk_id in (0, 2)
        ]
    )

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        expanded = service.expand_neighbors([hit], neighbors=1)

    assert [s.chunk_id for s in expanded] == [0, 1, 2], "命中 1 不能被部分结果吞掉"
    assert [s.text for s in expanded] == ["V1邻居-0", "V1命中正文", "V1邻居-2"]
    assert len(conn.queries) == 1
