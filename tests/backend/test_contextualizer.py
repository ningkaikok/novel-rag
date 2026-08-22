"""Contextual Retrieval 的单元测试。

不调真实 LLM——用假的生成函数验证调度逻辑：判据、缓存键、窗口、失败降级。
"""

from dataclasses import dataclass

from contextualizer import (
    build_window,
    extract_main_characters,
    generate_context,
    generate_contexts_parallel,
    is_context_poor,
    text_hash,
)


@dataclass
class FakeChunk:
    text: str


def test_context_poor_when_pronouns_without_names():
    """满篇代词、看不出在讲谁——正是需要补上下文的情况。"""
    assert is_context_poor("他们既然来过，又数日未回，或许是出了什么意外？", {"王慎"})


def test_not_poor_when_main_character_present():
    """含主要人物名，读者和检索系统都知道在讲谁，不需要额外补。"""
    assert not is_context_poor("王慎摇摇头，他并不这么想。", {"王慎"})


def test_not_poor_without_pronouns():
    """纯景物/设定描写没有指代问题，补上下文价值不大。"""
    assert not is_context_poor("那年秋天，连着下了半个月的雨。", {"王慎"})


def test_min_count_adapts_to_book_size():
    """小书也要能提取出人物名。

    踩过的坑：门槛写死为 20 时，只有 3 个片段的《雾隐山庄》里没有任何名字
    能出现 20 次，名单成了空集合，于是「不含任何人物名」恒为真——所有片段
    都被误判成缺上下文，判据在小书上静默失效。改成按片段数取比例后修复。

    注意用词要足够"像真的句子"：jieba 的人名识别依赖上下文，在
    「顾长风病了。」这种三字玩具句上会把人名切碎（顾/v + 长风/n），
    测不出真实行为。
    """
    tiny_book = [
        "沈砚之背着药箱走进山庄，向青黛说明来意。",
        "青黛请沈砚之进屋，屋里的药材已经快要用尽了。",
        "沈砚之连夜采药，青黛在一旁照看着。",
    ]
    names = extract_main_characters(tiny_book)
    assert names, "小书也应该能提取出人物名，不能返回空集合"


def test_hash_is_content_based_not_position_based():
    """缓存键必须跟着内容走。

    如果用 (书名, chunk_id) 做键，切分参数一变，同一个 chunk_id 对应的文本
    就变了，会取到过期的上下文说明。用内容哈希则天然正确。
    """
    assert text_hash("同样的文本") == text_hash("同样的文本")
    assert text_hash("文本 A") != text_hash("文本 B")


def test_window_includes_neighbors():
    """窗口要带上前后文——这是对 Anthropic 原方案（给整篇文档）的降级，
    长篇小说塞不进上下文窗口，只能给一个窗口。
    """
    chunks = [FakeChunk(f"第{i}段") for i in range(5)]
    window = build_window(chunks, index=2, neighbors=1)
    assert "第1段" in window and "第2段" in window and "第3段" in window
    assert "第0段" not in window and "第4段" not in window


def test_window_clamps_at_boundaries():
    """首尾片段不能越界。"""
    chunks = [FakeChunk("A"), FakeChunk("B")]
    assert "A" in build_window(chunks, index=0, neighbors=3)
    assert "B" in build_window(chunks, index=1, neighbors=3)


def test_generation_failure_degrades_to_empty_string():
    """生成失败必须降级而不是抛异常。

    某个片段超时/限流不该让整个入库流程卡住——直接索引原文即可，
    那只是回到没有 Contextual Retrieval 的状态，不是故障。
    """

    def boom(prompt):
        raise RuntimeError("模型限流了")

    assert generate_context("某书", "某片段", "某窗口", boom) == ""


def test_parallel_generation_preserves_order():
    """并发生成后结果顺序必须和输入一一对应，否则上下文会张冠李戴。"""
    tasks = [("书", f"片段{i}", "窗口") for i in range(6)]

    def fake_gen(prompt):
        # 从 prompt 里回读片段编号，模拟"每个片段生成对应的说明"
        marker = prompt.split("要说明的片段：\n")[-1]
        yield f"说明-{marker}"

    results, errors = generate_contexts_parallel(
        tasks, fake_gen, max_workers=4, progress_every=0
    )

    assert results == [f"说明-片段{i}" for i in range(6)]
    assert errors == []


def test_failures_are_reported_not_swallowed():
    """降级可以，但不能把失败原因也吞掉。

    踩过的坑：第一版只 return ""，结果 451 个片段全部失败，日志里只有一句
    "451 条生成失败"——真实原因（ingest 独立运行时没加载 .env、拿不到 API key）
    完全看不出来，只能自己去复现才查得到。
    """

    def always_fails(prompt):
        raise RuntimeError("未设置环境变量 ZHIPU_API_KEY")

    results, errors = generate_contexts_parallel(
        [("书", "片段", "窗口")] * 3, always_fails, max_workers=2, progress_every=0
    )

    assert results == ["", "", ""]  # 降级：返回空串，不抛异常
    assert len(errors) == 3
    assert "ZHIPU_API_KEY" in errors[0]  # 但原因保留下来了
