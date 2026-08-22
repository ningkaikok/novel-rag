"""自适应查询扩展（M3.4）的单元测试：expander 解析逻辑 + rag 流水线挂钩。

全部用假生成函数和假检索，不调真实 LLM、不连数据库。重点验证四件事：
1. 变体解析对模型输出的各种花式格式（编号/引号/重复/与原问题相同）的清洗；
2. 扩展开关关闭时主链路行为与从前完全一致（不多一次检索、无 expand trace）；
3. 触发时补救恰好发生一次（变体检索不会再触发扩展，绝不循环）；
4. trace 里 stage/reasons/variants/ms/still_no_evidence 字段齐全。
"""
import rag
from chunk_model import SourceChunk
from query_expander import expand_query_variants


def _chunk(novel: str, chunk_id: int, text: str) -> SourceChunk:
    return SourceChunk(
        novel=novel, chunk_id=chunk_id, text=text, distance=0.0, chapter_title="第一章"
    )


# --------------------------------------------------------------- expander 纯函数

def test_prompt_contains_question_and_limits():
    prompts = []

    def fake_generate(prompt):
        prompts.append(prompt)
        yield "变体甲\n变体乙"

    expand_query_variants("庄主的病是怎么好的", fake_generate, max_variants=2)
    # 只调一次 LLM（不是每个变体各调一次），且问题与变体上限都要写进提示词
    assert len(prompts) == 1
    assert "庄主的病是怎么好的" in prompts[0]
    assert "2 个改写变体" in prompts[0]


def test_parse_strips_numbering_quotes_and_dedupes():
    raw = (
        "1. 庄主所患的疾病是什么\n"
        "「庄主所患的疾病是什么」\n"  # 与上一行实质相同（仅引号包装）→ 去重
        "- 好的\n"  # 客套话残渣，清洗后太短 → 丢弃
    )
    variants = expand_query_variants(
        "雾隐山庄的庄主得了什么病", lambda _p: iter([raw]), max_variants=3
    )
    assert variants == ["庄主所患的疾病是什么"]


def test_variant_identical_to_original_is_dropped():
    raw = "雾隐山庄的庄主得了什么病。\n庄主患了什么疾病"
    variants = expand_query_variants(
        "雾隐山庄的庄主得了什么病", lambda _p: iter([raw])
    )
    # 第一个变体去掉标点后与原问题实质相同 → 丢弃；只剩第二个
    assert variants == ["庄主患了什么疾病"]


def test_max_variants_caps_output():
    raw = "变体一号问题\n变体二号问题\n变体三号问题\n变体四号问题"
    variants = expand_query_variants(
        "原始的问题是什么呢", lambda _p: iter([raw]), max_variants=3
    )
    assert len(variants) == 3


def test_generation_failure_degrades_to_empty_list_and_records_error():
    def boom(_prompt):
        raise RuntimeError("限流")

    errors: list[str] = []
    assert expand_query_variants("任何问题", boom, errors=errors) == []
    assert "RuntimeError: 限流" in errors[0]


# --------------------------------------------------------------- rag 挂钩点

def _stub_pipeline(monkeypatch, service, semantic, keyword, rerank_scored):
    """把 retrieve_hybrid_stream 依赖的召回/重排全部打桩（仿 test_retrieval_trace）。

    一律走 monkeypatch.setattr：直接给 rag 模块属性赋值会泄漏到后续用例。
    """
    calls = {"retrieve": 0}

    def retrieve(_question, top_k, only_novels):
        calls["retrieve"] += 1
        return list(semantic)

    monkeypatch.setattr(service, "_named_novels", lambda _q: [])
    monkeypatch.setattr(service, "_full_text_chunks", lambda _n: None)
    monkeypatch.setattr(service, "retrieve", retrieve)
    monkeypatch.setattr(
        service, "keyword_retrieve", lambda _q, top_k, only_novels: list(keyword)
    )
    monkeypatch.setattr(
        service, "positional_retrieve", lambda _q, top_k, hint_novels: []
    )
    monkeypatch.setattr(rag, "HIERARCHY_ENABLED", False)
    monkeypatch.setattr(rag, "RERANK_ENABLED", True)

    def stub_rerank(_question, candidates, _k):
        return [
            (
                next(
                    c
                    for c in candidates
                    if (c.novel, c.chunk_id) == (s.novel, s.chunk_id)
                ),
                score,
            )
            for s, score in rerank_scored
        ]

    monkeypatch.setattr(rag, "rerank_with_scores", stub_rerank)
    service._retrieve_calls = calls


def _low_confidence_fixture():
    """三段同书候选、分数咬得极紧且完全没覆盖问题词——必然判低置信。

    候选必须多于 top_k=2，否则流水线的重排分支不会执行、拿不到归一化分数。
    """
    semantic = [
        _chunk("雾隐山庄", 1, "庄主在练剑"),
        _chunk("雾隐山庄", 2, "仆人扫地"),
        _chunk("雾隐山庄", 3, "账房算账"),
    ]
    return "韩立的师父是谁", semantic, [], [
        (semantic[0], 5.0),
        (semantic[1], 4.99),
        (semantic[2], 4.9),
    ]


def test_expand_disabled_keeps_main_path_untouched(monkeypatch):
    monkeypatch.setattr(rag, "QUERY_EXPAND_ENABLED", False)
    service = object.__new__(rag.NovelRAG)
    question, semantic, keyword, scored = _low_confidence_fixture()
    _stub_pipeline(monkeypatch, service, semantic, keyword, scored)

    events = list(service.retrieve_hybrid_stream(question, top_k=2))
    steps = [payload for kind, payload in events if kind == "step"]
    assert not any(payload.get("stage") == "expand" for payload in steps)
    # 只召回了一轮，没有任何补救检索；结果就是普通重排后的 top-k
    assert service._retrieve_calls["retrieve"] == 1
    result = events[-1][1]
    assert [c.chunk_id for c in result] == [1, 2]


def test_expand_skipped_when_reranker_scores_absent(monkeypatch):
    monkeypatch.setattr(rag, "QUERY_EXPAND_ENABLED", True)
    service = object.__new__(rag.NovelRAG)
    _, semantic, keyword, scored = _low_confidence_fixture()
    _stub_pipeline(monkeypatch, service, semantic, keyword, scored)

    # 重排失败 → 拿不到归一化分数 → 绝不允许拿向量/BM25 原始分凑合触发扩展
    def broken_rerank(*_args):
        raise RuntimeError("模型加载失败")

    monkeypatch.setattr(rag, "rerank_with_scores", broken_rerank)
    events = list(service.retrieve_hybrid_stream("韩立的师父是谁", top_k=2))
    assert not any(p.get("stage") == "expand" for k, p in events if k == "step")


def test_expand_triggers_once_and_merges_reranks(monkeypatch):
    monkeypatch.setattr(rag, "QUERY_EXPAND_ENABLED", True)
    service = object.__new__(rag.NovelRAG)
    question, semantic, keyword, scored = _low_confidence_fixture()
    _stub_pipeline(monkeypatch, service, semantic, keyword, scored)

    generated = {"prompts": []}

    def fake_expand_generate(prompt):
        generated["prompts"].append(prompt)
        yield "墨大夫的下落\n韩立师父的结局"

    service.expand_generate_fn = fake_expand_generate

    events = list(service.retrieve_hybrid_stream(question, top_k=2))
    assert [kind for kind, _ in events][-1] == "result"

    expand_steps = [p for k, p in events if k == "step" and p.get("stage") == "expand"]
    # 恰好两条补救 trace：「查询扩展」（触发+变体）+「扩展重排」（收尾）
    assert [s["step"] for s in expand_steps] == ["查询扩展", "扩展重排"]

    trigger = expand_steps[0]
    assert {"score_gap", "cross_book_dispersion"} <= set(trigger["reasons"])
    assert trigger["variants"] == ["墨大夫的下落", "韩立师父的结局"]
    assert isinstance(trigger["ms"], int) and trigger["ms"] >= 0

    final_step = expand_steps[1]
    assert isinstance(final_step["still_no_evidence"], bool)
    assert isinstance(final_step["ms"], int) and final_step["ms"] >= 0

    # 原始 1 轮召回 + 2 个变体各 1 轮；生成函数只在补救时调过一次
    assert service._retrieve_calls["retrieve"] == 3
    assert len(generated["prompts"]) == 1

    result = events[-1][1]
    assert 1 <= len(result) <= 2


def test_expand_not_triggered_when_confidence_high(monkeypatch):
    monkeypatch.setattr(rag, "QUERY_EXPAND_ENABLED", True)
    service = object.__new__(rag.NovelRAG)
    # 分差悬殊 + 关键词全覆盖 + 单书 → 高置信，不应花任何补救开销
    text_a = "韩立的师父是墨大夫，他传授了长春功"
    text_b = "墨大夫另有图谋的相关段落原文"
    semantic = [_chunk("凡人修仙传", 1, text_a), _chunk("凡人修仙传", 2, text_b)]
    _stub_pipeline(
        monkeypatch, service, semantic, [], [(semantic[0], 9.0), (semantic[1], -3.0)]
    )

    def must_not_call(_prompt):  # 高置信时绝不该碰生成函数
        raise AssertionError("高置信度不应触发生成")

    service.expand_generate_fn = must_not_call
    events = list(service.retrieve_hybrid_stream("韩立的师父是谁", top_k=2))
    assert not any(p.get("stage") == "expand" for k, p in events if k == "step")
    assert service._retrieve_calls["retrieve"] == 1


def test_variant_retrieval_cannot_retrigger_expansion(monkeypatch):
    """补救只发生一次：即使变体的检索结果同样低置信，也不许再扩展。"""
    monkeypatch.setattr(rag, "QUERY_EXPAND_ENABLED", True)
    service = object.__new__(rag.NovelRAG)
    question, semantic, keyword, scored = _low_confidence_fixture()
    _stub_pipeline(monkeypatch, service, semantic, keyword, scored)

    service.expand_generate_fn = lambda _p: iter(["墨大夫的下落"])
    events = list(service.retrieve_hybrid_stream(question, top_k=2))

    expand_steps = [p for k, p in events if k == "step" and p.get("stage") == "expand"]
    assert len(expand_steps) == 2  # 触发 + 重排收尾，之后没有新的补救步骤
    assert service._retrieve_calls["retrieve"] == 2  # 原始 1 轮 + 变体 1 轮


def test_missing_generate_fn_is_reported_in_trace(monkeypatch):
    """开了开关但没注入生成函数（如独立脚本环境）：trace 里要有明确说明。"""
    monkeypatch.setattr(rag, "QUERY_EXPAND_ENABLED", True)
    service = object.__new__(rag.NovelRAG)
    question, semantic, keyword, scored = _low_confidence_fixture()
    _stub_pipeline(monkeypatch, service, semantic, keyword, scored)

    events = list(service.retrieve_hybrid_stream(question, top_k=2))
    expand_steps = [p for k, p in events if k == "step" and p.get("stage") == "expand"]
    assert len(expand_steps) == 1
    assert "没有可用的生成后端" in expand_steps[0]["detail"]
    assert expand_steps[0]["variants"] == []
