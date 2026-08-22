"""GraphRAG 的单元测试。不调真实 LLM、不连数据库。"""

from dataclasses import dataclass

from graph import (
    _parse_names,
    build_edges,
    chunks_with_relation,
    detect_relation_question,
    extract_characters_from_chunks,
    format_graph_hint,
)


@dataclass
class FakeChunk:
    novel: str
    text: str


def test_detects_relation_questions():
    assert detect_relation_question("韩立有哪些伴侣？") == "伴侣"
    assert detect_relation_question("韩立的师父是谁") == "师徒"
    assert detect_relation_question("韩立的仇人有哪些") == "敌对"


def test_ignores_non_relation_questions():
    """图检索是补充不是替代——普通问题不该多查一次图。"""
    assert detect_relation_question("凡人修仙传的结局是什么") is None
    assert detect_relation_question("韩立修炼的功法叫什么") is None


def test_parse_names_filters_non_names():
    """模型可能输出解释性短语或标点，要清洗掉。"""
    names = _parse_names("韩立、南宫婉，紫灵。这些是主要人物")
    assert "韩立" in names and "南宫婉" in names and "紫灵" in names
    assert "这些是主要人物" not in names  # 超过 4 个字，不当人名


def test_parse_names_rejects_non_chinese():
    assert _parse_names("Han Li, 韩立") == ["韩立"]


def test_chunks_with_relation_filters_by_keyword():
    chunks = [
        FakeChunk("书", "韩立与南宫婉结为双修伴侣"),
        FakeChunk("书", "韩立在打坐修炼"),
    ]
    matched = chunks_with_relation(chunks, "伴侣")
    assert len(matched) == 1
    assert "双修伴侣" in matched[0].text


def test_build_edges_counts_cooccurrence():
    """边的权重就是共现次数——这是唯一的置信度信号。"""
    chunks = [
        FakeChunk("书", "韩立与南宫婉相识"),
        FakeChunk("书", "南宫婉看着韩立"),
        FakeChunk("书", "韩立独自一人"),  # 只有一个人，不产生边
    ]
    edges = build_edges(chunks, {"韩立", "南宫婉"}, "伴侣")

    assert len(edges) == 1
    novel, a, b, rel, weight = edges[0]
    assert {a, b} == {"韩立", "南宫婉"}
    assert weight == 2, "两个片段都共现，权重应为 2"


def test_edges_are_undirected_and_sorted():
    """关系无向：同一对人不能因为出现顺序不同被记成两条边。"""
    chunks = [
        FakeChunk("书", "韩立和南宫婉"),
        FakeChunk("书", "南宫婉和韩立"),  # 顺序相反
    ]
    edges = build_edges(chunks, {"韩立", "南宫婉"}, "伴侣")
    assert len(edges) == 1, "顺序相反的共现应该合并成一条边"
    assert edges[0][4] == 2


def test_extraction_failure_skips_batch_not_whole_build():
    """某批抽取失败不该让整个建图挂掉，但原因要保留下来。"""

    def boom(prompt):
        raise RuntimeError("模型限流")

    errors: list[str] = []
    result = extract_characters_from_chunks([FakeChunk("书", "内容")], boom, errors=errors)

    assert result == {}  # 降级：这批没抽到
    assert len(errors) == 1 and "限流" in errors[0]  # 但原因保留了


def test_hint_marks_itself_as_inference():
    """图线索必须标明是统计推断，不能让模型当成确定事实照抄。"""
    hint = format_graph_hint("韩立", "伴侣", [("南宫婉", 11), ("紫灵", 7)])

    assert "南宫婉" in hint and "11" in hint
    assert "推断" in hint, "必须标注这是推断而非事实"


def test_empty_neighbors_produce_no_hint():
    """查不到关系时不要往 prompt 里塞空提示。"""
    assert format_graph_hint("韩立", "伴侣", []) == ""
