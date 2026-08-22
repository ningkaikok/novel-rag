"""GraphRAG 的单元测试。不调真实 LLM、不连数据库。"""

from dataclasses import dataclass

from graph import (
    CO_OCCURRENCE_CONFIDENCE,
    EVIDENCE_CO_OCCURRENCE,
    EVIDENCE_EXPLICIT,
    _parse_names,
    _parse_relation_payload,
    build_edge_records,
    build_edges,
    chunks_with_relation,
    detect_relation_question,
    extract_characters_from_chunks,
    extract_relations_llm,
    format_graph_hint,
)


@dataclass
class FakeChunk:
    novel: str
    text: str
    chunk_id: int | None = None


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


# ------------------------------------------------------- M4：边记录与质量字段


def test_build_edge_records_carries_quality_fields():
    """共现边的记录要带全 M4 质量字段：低置信度、co_occurrence、来源片段。"""
    chunks = [
        FakeChunk("书", "韩立与南宫婉相识", chunk_id=7),
        FakeChunk("书", "南宫婉看着韩立", chunk_id=9),
        FakeChunk("书", "韩立独自一人", chunk_id=10),  # 只有一人，不产生边
    ]
    records = build_edge_records(chunks, {"韩立", "南宫婉"}, "伴侣")

    assert len(records) == 1
    record = records[0]
    assert (record["person_a"], record["person_b"]) == ("南宫婉", "韩立") or (
        record["person_a"],
        record["person_b"],
    ) == ("韩立", "南宫婉")
    assert record["weight"] == 2
    assert record["confidence"] == CO_OCCURRENCE_CONFIDENCE == 0.3, "共现推断必须如实给低分"
    assert record["evidence_type"] == EVIDENCE_CO_OCCURRENCE
    assert record["source_chunk_ids"] == [7, 9], "来源片段用于审核界面回溯原文"
    assert record["direction"] is None, "共现判断不出方向"


def test_build_edges_keeps_legacy_tuple_shape():
    """旧调用方拿到的仍是 5 元组；没有 chunk_id 的假片段也不能崩。

    A/B 按字典序存（南 < 韩），同一对人不会因出现顺序不同记成两条边。
    """
    chunks = [FakeChunk("书", "韩立与南宫婉相识")]  # 没有 chunk_id 属性
    edges = build_edges(chunks, {"韩立", "南宫婉"}, "伴侣")
    assert edges == [("书", "南宫婉", "韩立", "伴侣", 1)]


def test_parse_relation_payload_happy_path():
    """合法 JSON 数组解析成边记录：方向归一化、置信度截断、编号映射。"""
    raw = """前面是一些废话。
    [{"a": "沈砚秋", "b": "小顺", "direction": "沈砚秋→小顺",
      "kind": "explicit", "confidence": 1.7, "chunks": [2]}]
    后面也是废话。"""
    records = _parse_relation_payload(
        raw,
        {"沈砚秋", "小顺"},
        [11, 12, 13],  # 片段2 → batch_chunk_ids[1] = 12
    )

    assert len(records) == 1
    record = records[0]
    # person_a/person_b 按字典序归一（与共现边同一种存储约定），方向单独立字段
    assert (record["person_a"], record["person_b"]) == ("小顺", "沈砚秋")
    assert record["evidence_type"] == EVIDENCE_EXPLICIT
    assert record["confidence"] == 1.0, "超出 [0,1] 的置信度必须截断"
    assert record["direction"] == "沈砚秋→小顺", "语义方向不受存储顺序影响"
    assert record["source_chunk_ids"] == [12]


def test_parse_relation_payload_tolerates_echoed_markers():
    """模型复读「[片段1]」时不能把它当成 JSON 数组起点（实测踩过的坑）。"""
    raw = (
        '[片段1] 沙海航灯\n[片段2] 三块石板\n\n[{"a": "陆知微", "b": "巴特尔", '
        '"direction": "none", "kind": "co_occurrence", "confidence": 0.4, "chunks": []}]'
    )
    records = _parse_relation_payload(raw, {"陆知微", "巴特尔"}, [5, 6])

    assert len(records) == 1
    assert records[0]["evidence_type"] == EVIDENCE_CO_OCCURRENCE
    # chunks 为空 → 整批都是来源，宁可冗余也不丢追溯
    assert records[0]["source_chunk_ids"] == [5, 6]


def test_parse_relation_payload_rejects_garbage_and_hallucinated_names():
    """解析不出 JSON 返回 None（触发降级）；名单外的"人名"直接丢弃。"""
    assert _parse_relation_payload("完全不是 JSON", set(), []) is None

    raw = '[{"a": "韩立", "b": "老者", "kind": "explicit", "confidence": 0.9}]'
    assert _parse_relation_payload(raw, {"韩立"}, []) == [], "泛称不在候选名单，丢弃"

    same = '[{"a": "韩立", "b": "韩立", "kind": "explicit", "confidence": 0.9}]'
    assert _parse_relation_payload(same, {"韩立"}, []) == [], "a==b 是幻觉，丢弃"


def test_extract_relations_llm_merges_batches_and_fails_gracefully():
    """多批结果按人物对合并（保留高置信度、并集来源）；全部失败返回 None。"""

    def good_fn(prompt):
        return (
            '[{"a": "甲", "b": "乙", "direction": "甲→乙", "kind": "explicit", '
            '"confidence": 0.8, "chunks": [1]}]'
        )

    chunks = [FakeChunk("书", "甲和乙在一起", chunk_id=i) for i in range(4)]
    records = extract_relations_llm(chunks[:2], {"甲", "乙"}, "同伴", good_fn)
    assert records[0]["weight"] == 1, "weight 取独立来源片段数"
    assert records[0]["novel"] == "书"

    calls = {"n": 0}

    def flaky_fn(prompt):
        calls["n"] += 1
        if calls["n"] == 1:  # 第一批输出坏 JSON
            return "这不是 JSON"
        return (
            '[{"a": "甲", "b": "乙", "direction": "none", "kind": "explicit", '
            '"confidence": 0.6, "chunks": [1]}]'
        )

    errors: list[str] = []
    records = extract_relations_llm(
        chunks, {"甲", "乙"}, "同伴", flaky_fn, batch_size=2, errors=errors
    )
    assert len(errors) == 1 and "JSON" in errors[0], "失败原因必须保留"
    assert len(records) == 1 and records[0]["confidence"] == 0.6, (
        "只要有一批成功就继续，不整体降级"
    )

    def boom(prompt):
        raise RuntimeError("限流")

    records = extract_relations_llm(chunks, {"甲", "乙"}, "同伴", boom, errors=errors)
    assert records is None, "所有批次都失败才整体降级为 None"


def test_merge_relation_records_dedupes_and_unions_sources():
    """跨批次同一对人只留一条：置信度取高、来源取并、weight 重算。"""
    from graph import _merge_relation_records

    merged = _merge_relation_records(
        [
            {
                "novel": "书",
                "person_a": "甲",
                "person_b": "乙",
                "relation": "同伴",
                "direction": "甲→乙",
                "confidence": 0.6,
                "evidence_type": EVIDENCE_EXPLICIT,
                "source_chunk_ids": [1],
            },
            {
                "novel": "书",
                "person_a": "甲",
                "person_b": "乙",
                "relation": "同伴",
                "direction": "甲→乙",
                "confidence": 0.9,
                "evidence_type": EVIDENCE_EXPLICIT,
                "source_chunk_ids": [2],
            },
        ]
    )
    assert len(merged) == 1
    assert merged[0]["confidence"] == 0.9
    assert sorted(merged[0]["source_chunk_ids"]) == [1, 2], "来源片段做并集"
    assert merged[0]["weight"] == 2
