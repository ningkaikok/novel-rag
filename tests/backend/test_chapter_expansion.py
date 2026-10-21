"""M3.4 整章扩展对照实验的单元测试（全部 mock，不连真实数据库）。

覆盖四类行为：
1. off / neighbors 档与旧 expand_neighbors 行为逐字节一致，且不产生 trace 步骤；
2. chapter 档把同章命中聚合为整章、跨书各自独立；
3. token 预算截断的方向（从命中向两侧丢最远的片段）与 trace 记录；
4. tokenizer 不可用时不假装闸门存在，trace 如实记录。
"""

import types
from unittest.mock import patch

import retrieval_mixins


# ---------------------------------------------------------------- 假模型
class _FakeTokenizer:
    """假 tokenizer：1 个字符 = 1 个 token。

    满足 index_quality._token_count 的调用协议（truncation=False、
    add_special_tokens=True），让预算闸门在测试里有确定性的计数结果。
    """

    def __call__(self, text, padding=False, truncation=False, add_special_tokens=True):
        return {"input_ids": [1] * len(text)}


class _FakeEmbedder:
    """带 tokenizer 的假 embedding 模型。"""

    tokenizer = _FakeTokenizer()


class _BareEmbedder:
    """没有 tokenizer 属性的替身：模拟 tokenizer 不可用的环境。"""


# ---------------------------------------------------------------- 假连接
class _FakeConn:
    """假的 ``with connect() as conn:``，按 SQL 形状分发两种查询：

    - 整章查询：``WHERE novel = %s AND chapter_title = %s ORDER BY chunk_id``
    - 邻居扩展查询：若干 ``(novel, lo, hi)`` 的 BETWEEN 区间（off 档回归用）
    """

    def __init__(self, chapters):
        # {(novel, chapter_title): [{"chunk_id": int, "text": str, ...}, ...]}
        self.chapters = chapters
        self.calls: list[tuple[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.calls.append((" ".join(sql.split()), params))
        result = types.SimpleNamespace(fetchall=lambda: [])

        if "chapter_title = %s" in sql:
            novel, title = params
            rows = sorted(
                self.chapters.get((novel, title), []),
                key=lambda row: row["chunk_id"],
            )
            # 真实 SQL 会按 chunk_id 返回；这里同样保证原文顺序
            result.fetchall = lambda: [
                dict(row, novel=novel, chapter_title=title) for row in rows
            ]
        elif "BETWEEN" in sql:
            flat = list(params)
            collected: list[dict] = []
            seen: set[tuple[str, int]] = set()
            for index in range(0, len(flat), 3):
                novel, low, high = flat[index : index + 3]
                for (book, title), rows in self.chapters.items():
                    if book != novel:
                        continue
                    for row in rows:
                        key = (novel, row["chunk_id"])
                        if low <= row["chunk_id"] <= high and key not in seen:
                            seen.add(key)
                            collected.append(dict(row, novel=novel, chapter_title=title))
            result.fetchall = lambda: collected
        return result


# ---------------------------------------------------------------- 工具函数
def _rows(ids, width=4):
    """造一章的数据库行：每个片段 width 个字（= width 个 token）。"""
    return [{"chunk_id": cid, "text": "字" * width, "context": ""} for cid in ids]


def _source(novel="甲书", chunk_id=11, chapter="第一章", width=4):
    return retrieval_mixins.SourceChunk(
        novel=novel,
        chunk_id=chunk_id,
        text="字" * width,
        distance=0.0,
        chapter_title=chapter,
    )


def _service(embedder=None):
    """不跑 __init__（避免连库/加载模型），只注入 embedder。"""
    service = object.__new__(retrieval_mixins.RetrievalMixin)
    service.embedder = embedder or _FakeEmbedder()
    return service


# ---------------------------------------------------------------- 测试
def test_off_and_neighbors_modes_are_identical_to_legacy_expand(monkeypatch):
    """off / neighbors 两档必须与旧 expand_neighbors 完全一致且无 trace 步骤。"""
    chapters = {("雾隐山庄", "第一章"): _rows(range(3, 10))}
    conn = _FakeConn(chapters)
    service = _service()
    sources = [_source(novel="雾隐山庄", chunk_id=6)]

    for mode in ("off", "neighbors"):
        monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MODE", mode)
        with patch.object(retrieval_mixins, "connect", lambda: conn):
            via_entry, step = service.build_answer_context(sources)
            legacy = service.expand_neighbors(sources)

        assert step is None, f"{mode} 档不应产生任何新的 trace 步骤"
        assert [(c.novel, c.chunk_id) for c in via_entry] == [
            (c.novel, c.chunk_id) for c in legacy
        ], f"{mode} 档的最终证据列表必须与旧邻居扩展逐条一致"
        assert [(c.novel, c.chunk_id) for c in via_entry] == [
            ("雾隐山庄", cid) for cid in (5, 6, 7)
        ], "命中 6 号片段应按 CONTEXT_NEIGHBORS 补齐前后各 1 个相邻片段"


def test_chapter_mode_aggregates_whole_chapters_per_book(monkeypatch):
    """chapter 档：同章命中聚合成整章原文顺序，跨书的组各自独立。"""
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MODE", "chapter")
    chapters = {
        ("甲书", "第一章"): _rows([10, 11, 12]),
        ("乙书", "序章"): _rows([0, 1]),
    }
    conn = _FakeConn(chapters)
    service = _service()
    # 两个命中分属两本书：甲书的整章绝不能混进乙书的片段，反之亦然
    sources = [_source("甲书", 11, "第一章"), _source("乙书", 0, "序章")]

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        result, step = service.build_answer_context(sources)

    assert [(c.novel, c.chunk_id) for c in result] == [
        ("甲书", 10),
        ("甲书", 11),
        ("甲书", 12),
        ("乙书", 0),
        ("乙书", 1),
    ], "每本书都应带入命中所在章节的全部片段，组内按原文顺序、组间按相关性顺序"
    assert step["expansion_mode"] == "chapter"
    assert step["truncated"] is False
    assert step["truncation_reason"] is None
    # 5 个片段 × 每段 4 字 = 20 tokens（假 tokenizer 按 1 字 1 token 计）
    assert step["evidence_tokens"] == 20
    assert step["ms"] >= 0

    # trace 步骤必须能通过后端的 TraceStep 校验（字段名/类型不会手滑写错）
    from backend.schemas import TraceStep

    TraceStep(**step).model_dump()


def test_chapter_mode_truncates_outward_from_hits_within_budget(monkeypatch):
    """超预算时从命中片段向两侧对称生长到放不下为止，并记录截断原因。

    场景：一章 10 个片段、每段 10 tokens，命中第 5 片段，预算 35：
    命中本身占 10，剩余 25 → 先左纳入 4 号（余 15）→ 再右纳入 6 号（余 5）→
    两侧下一个都是 10 tokens 放不下，停止。最终保留 {4,5,6}。
    """
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MODE", "chapter")
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MAX_TOKENS", 35)
    chapters = {("长章之书", "第二章"): _rows(range(10), width=10)}
    conn = _FakeConn(chapters)
    service = _service()
    sources = [_source("长章之书", 5, "第二章", width=10)]

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        result, step = service.build_answer_context(sources)

    assert [c.chunk_id for c in result] == [4, 5, 6], (
        "截断方向必须从命中向两侧对称生长：离命中最近的片段优先保留，离得最远的先被丢掉"
    )
    assert step["evidence_tokens"] == 30
    assert step["evidence_tokens"] <= 35, "硬性闸门：拼入 prompt 的证据不得超预算"
    assert step["truncated"] is True
    assert step["truncation_reason"] and "截断" in step["truncation_reason"]
    assert "30" in step["detail"], "detail 里应能看到实际使用的证据 token 数"


def test_chapter_mode_hits_alone_over_budget_keeps_hits_and_records(monkeypatch):
    """光命中片段就超预算的极端情况：证据不能被挤掉，全保留并如实记录。"""
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MODE", "chapter")
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MAX_TOKENS", 5)
    chapters = {("巨章之书", "第三章"): _rows([100], width=50)}
    conn = _FakeConn(chapters)
    service = _service()
    sources = [_source("巨章之书", 100, "第三章", width=50)]

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        result, step = service.build_answer_context(sources)

    assert [c.chunk_id for c in result] == [100]
    assert step["evidence_tokens"] == 50
    assert step["truncated"] is True
    assert "超预算" in step["truncation_reason"]


def test_chapter_mode_records_gate_skip_without_tokenizer(monkeypatch):
    """tokenizer 不可用时不能静默跳过闸门：全量带入但 trace 明确说明。"""
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MODE", "chapter")
    chapters = {("甲书", "第一章"): _rows([10, 11, 12])}
    conn = _FakeConn(chapters)
    service = _service(embedder=_BareEmbedder())
    sources = [_source("甲书", 11, "第一章")]

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        result, step = service.build_answer_context(sources)

    assert [c.chunk_id for c in result] == [10, 11, 12]
    assert step["evidence_tokens"] is None
    assert step["truncated"] is False
    assert "tokenizer" in step["truncation_reason"]


def test_chapter_mode_keeps_untitled_hits_without_guessing_bounds(monkeypatch):
    """没有章节标题的命中无从谈「章」：只保留命中本身，不猜边界。"""
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MODE", "chapter")
    conn = _FakeConn({})
    service = _service()
    sources = [_source("旧索引书", 7, chapter=None)]

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        result, step = service.build_answer_context(sources)

    assert [(c.novel, c.chunk_id) for c in result] == [("旧索引书", 7)]
    assert step["expansion_mode"] == "chapter"
    assert step["evidence_tokens"] == 4


def test_chapter_mode_counts_chunks_between_discontiguous_hits(monkeypatch):
    """不连续命中之间的中间片段也必须占预算，不能为了「维持连续区间」白送。

    场景（计划指定的回归场景）：一章 10 段、每段 10 tokens，命中第 1 段和第 10 段，
    预算 35。两个命中本身已经 20 tokens，只剩 15；把它们之间的 8 段全带进来
    需要 80 tokens——旧实现把这 8 段当成「连续区间的一部分」免费塞入，最终
    证据总量 100 tokens，比预算 35 超出近两倍，闸门形同虚设。
    """
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MODE", "chapter")
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MAX_TOKENS", 35)
    chapters = {("长章之书", "第二章"): _rows(range(10), width=10)}
    conn = _FakeConn(chapters)
    service = _service()
    sources = [
        _source("长章之书", 0, "第二章", width=10),
        _source("长章之书", 9, "第二章", width=10),
    ]

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        result, step = service.build_answer_context(sources)

    assert [c.chunk_id for c in result] == [0, 9], "付不起中间片段时只保留命中本身"
    assert step["evidence_tokens"] == 20
    assert step["evidence_tokens"] <= 35, "硬性闸门：不允许因中间片段突破预算"
    assert step["truncated"] is True
    assert "中间片段" in (step["truncation_reason"] or "")


def test_chapter_mode_pays_for_interior_chunks_when_budget_allows(monkeypatch):
    """预算充足时中间片段要真的带上，且它们的 token 也要计入证据总量。"""
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MODE", "chapter")
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MAX_TOKENS", 50)
    chapters = {("长章之书", "第二章"): _rows(range(10), width=10)}
    conn = _FakeConn(chapters)
    service = _service()
    sources = [
        _source("长章之书", 3, "第二章", width=10),
        _source("长章之书", 5, "第二章", width=10),
    ]

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        result, step = service.build_answer_context(sources)

    # 命中 {3, 5} 本身 20 tokens，中间 4 号 10 tokens → 30；剩 20 再向两侧各延一段 → 50
    assert [c.chunk_id for c in result] == [2, 3, 4, 5, 6]
    assert step["evidence_tokens"] == 50
    assert step["evidence_tokens"] <= 50


def test_chapter_mode_shares_one_budget_across_chapters(monkeypatch):
    """多章节共用同一份预算：靠前的相关章节先拿到余量，总证据不得超预算。"""
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MODE", "chapter")
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MAX_TOKENS", 40)
    chapters = {
        ("甲书", "第一章"): _rows(range(5), width=10),
        ("乙书", "第二章"): _rows(range(10, 15), width=10),
    }
    conn = _FakeConn(chapters)
    service = _service()
    sources = [
        _source("甲书", 2, "第一章", width=10),
        _source("乙书", 12, "第二章", width=10),
    ]

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        result, step = service.build_answer_context(sources)

    assert step["evidence_tokens"] <= 40, "两个章节加起来的证据也要守同一个预算"
    assert [c.novel for c in result][:3] == ["甲书", "甲书", "甲书"]
    assert ("乙书", 12) in [(c.novel, c.chunk_id) for c in result]
    assert step["truncated"] is True, "乙书只剩命中、没能延展，应如实记为截断"


def test_chapter_mode_single_hit_in_one_chunk_chapter_is_not_truncated(monkeypatch):
    """单命中且章节只有一个片段：正好在预算内，不截断也不报假截断。"""
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MODE", "chapter")
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MAX_TOKENS", 10)
    conn = _FakeConn({("短章之书", "第一章"): _rows([0], width=10)})
    service = _service()
    sources = [_source("短章之书", 0, "第一章", width=10)]

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        result, step = service.build_answer_context(sources)

    assert [c.chunk_id for c in result] == [0]
    assert step["evidence_tokens"] == 10
    assert step["truncated"] is False
    assert step["truncation_reason"] is None


def test_chapter_mode_empty_input_reports_no_evidence_without_fake_gate(monkeypatch):
    """空输入：返回空证据，但不能谎称"tokenizer 不可用导致闸门没执行"。"""
    monkeypatch.setattr(retrieval_mixins, "CHAPTER_EXPANSION_MODE", "chapter")
    conn = _FakeConn({})
    service = _service()

    with patch.object(retrieval_mixins, "connect", lambda: conn):
        result, step = service.build_answer_context([])

    assert result == []
    assert step["evidence_tokens"] == 0
    assert step["truncated"] is False
    assert step["truncation_reason"] is None
    assert conn.calls == [], "没有命中就不该发起任何章节查询"

    from backend.schemas import TraceStep

    TraceStep(**step).model_dump()
