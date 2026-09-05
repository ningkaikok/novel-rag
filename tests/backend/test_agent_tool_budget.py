"""Agent Lab 工具输出的体积闸门（M3.6 独立项）。

步数上限（3~5 步）管的是**次数**，参数上限（radius≤3 / limit≤12）管的是**条数**，
两者都拦不住**体积**：大部头的一章、或几段特别长的原文，条数完全合法、字数照样
能撑爆 prompt。这里测的是补上的第二道闸。

不连数据库、不调模型。
"""

import agent_lab
from agent_lab import _apply_output_budget, _summarize_items
from chunk_model import SourceChunk


def _chunk(chunk_id: int, chars: int = 100) -> SourceChunk:
    return SourceChunk("雾隐山庄", chunk_id, "原" * chars, 0.0, "第一章")


def test_output_within_budget_is_untouched():
    """没超预算就什么都不该发生——闸门不能变成常态截断。"""
    sources = [_chunk(i, 100) for i in range(3)]
    kept, trace = _apply_output_budget(sources, max_chars=1000)

    assert kept == sources
    assert trace["truncated"] is False
    assert trace["dropped"] == 0


def test_chapter_output_is_cut_from_the_tail():
    """按原文顺序读的场景（get_chapter）：从末尾开始丢，前面的先保住。"""
    sources = [_chunk(i, 100) for i in range(12)]
    kept, trace = _apply_output_budget(sources, max_chars=450)

    assert [s.chunk_id for s in kept] == [0, 1, 2, 3]
    assert trace["truncated"] is True
    assert trace["dropped"] == 8
    assert "6000" not in trace["reason"] and "450" in trace["reason"]


def test_neighbors_keep_the_center_and_grow_outwards():
    """read_neighbors 的中心片段无条件保留——它正是用户点名要看的那一段。

    截断从离中心最远处开始，和整章扩展是同一套策略：最远端的邻居本来就最可有可无。
    """
    sources = [_chunk(i, 100) for i in range(7)]  # chunk_id 0..6，中心 3
    kept, trace = _apply_output_budget(sources, max_chars=350, center_chunk_id=3)

    ids = [s.chunk_id for s in kept]
    assert 3 in ids, "中心片段绝不能被预算挤掉"
    assert ids == sorted(ids), "保留下来的仍要按原文顺序"
    assert 0 not in ids and 6 not in ids, "该丢的是离中心最远的两端"
    assert trace["truncated"] is True


def test_a_single_oversized_chunk_still_comes_back():
    """哪怕一段自己就超预算，也不能返回空——那等于这次工具调用白跑，
    模型会拿着"什么都没读到"继续瞎猜，比截断更糟。"""
    kept, trace = _apply_output_budget([_chunk(1, 5000)], max_chars=100)

    assert len(kept) == 1
    assert trace["truncated"] is True


def test_truncation_is_written_into_the_observation(monkeypatch):
    """截断不写进可见轨迹就是静默丢证据。规划器要能看见"这里被截了"。"""

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, *_a):
            return self

        def fetchall(self):
            return [
                {
                    "novel": "雾隐山庄",
                    "chunk_id": i,
                    "chapter_title": "第一章",
                    "text": "原" * 3000,
                    "context": "",
                }
                for i in range(5)
            ]

    monkeypatch.setattr(agent_lab, "connect", lambda: _Conn())
    toolbox = agent_lab.AgentToolbox(rag=None)
    monkeypatch.setattr(toolbox, "_resolve_novel", lambda n: n)

    result = toolbox.get_chapter("雾隐山庄", "第一章")

    assert "预算" in result.summary, "截断必须出现在规划器读得到的 observation 里"
    assert result.facts["budget"]["truncated"] is True
    assert result.facts["returned_count"] == len(result.sources)


# ------------------------------------------------------------------ 目录类输出


def test_long_catalog_line_is_shortened_but_counts_stay_exact():
    """目录只压给规划器看的那一行，facts 里的完整清单不能动——
    "一共有几部小说"这类确定性回答依赖后者。"""
    items = [f"小说{i:03d}" for i in range(100)]
    text = _summarize_items(items, max_chars=100)

    assert len(text) < 200
    assert "共 100 项" in text
    assert "小说000" in text


def test_short_catalog_is_listed_in_full():
    assert _summarize_items(["甲", "乙", "丙"]) == "甲、乙、丙"
    assert _summarize_items([]) == "（空）"
