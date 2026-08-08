"""重排逻辑的单元测试。

不加载真实的交叉编码器模型（1.1GB，CI 里也没有），用假模型验证调度逻辑：
输入怎么组装成「问题-文档」对、分数怎么排序、top_k 怎么截断。
"""
from dataclasses import dataclass

from reranker import rerank


@dataclass
class FakeChunk:
    text: str


class FakeCrossEncoder:
    """假的交叉编码器：按预设好的分数表返回，并记录收到的输入。"""

    def __init__(self, scores):
        self._scores = scores
        self.received_pairs = None

    def predict(self, pairs):
        self.received_pairs = pairs
        return self._scores


def test_reorders_by_score():
    """核心行为：按交叉编码器给的分数重新排序，分高的在前。"""
    chunks = [FakeChunk("低相关"), FakeChunk("高相关"), FakeChunk("中相关")]
    model = FakeCrossEncoder([0.1, 0.9, 0.5])

    result = rerank("随便问问", chunks, top_k=3, model=model)

    assert [c.text for c in result] == ["高相关", "中相关", "低相关"]


def test_truncates_to_top_k():
    """重排的意义之一就是从多个候选里精挑少数几个送进 prompt。"""
    chunks = [FakeChunk(f"片段{i}") for i in range(5)]
    model = FakeCrossEncoder([0.1, 0.2, 0.3, 0.4, 0.5])

    result = rerank("问题", chunks, top_k=2, model=model)

    assert len(result) == 2
    assert [c.text for c in result] == ["片段4", "片段3"]


def test_builds_question_document_pairs():
    """交叉编码器的输入必须是「问题-文档」成对的——这正是它区别于双编码器的
    地方：两者拼在一起送进模型才能做注意力交互。
    """
    chunks = [FakeChunk("文本A"), FakeChunk("文本B")]
    model = FakeCrossEncoder([0.5, 0.5])

    rerank("我的问题", chunks, top_k=2, model=model)

    assert model.received_pairs == [("我的问题", "文本A"), ("我的问题", "文本B")]


def test_empty_candidates_short_circuits():
    """没有候选时直接返回，不该去加载模型（加载要 1.1GB + 好几秒）。"""
    assert rerank("问题", [], top_k=5) == []


@dataclass
class FakeContextualChunk:
    """带上下文说明的片段，模拟 SourceChunk 的 indexed_text 行为。"""

    text: str
    context: str = ""

    @property
    def indexed_text(self) -> str:
        return f"{self.context}\n{self.text}" if self.context else self.text


def test_rerank_sees_contextual_enhancement():
    """重排必须看到 Contextual Retrieval 加的上下文说明，不能只看原文。

    踩过的坑：上下文说明只进了索引（所以召回变好了），但重排拿 text 列的原文
    重新打分、看不到说明，等于把增强效果整个抵消。实测同一个片段：
        重排给【原文】       0.0055
        重排给【说明+原文】  0.9990    ← 差 180 倍
    结果正确片段被从第 5 名压到 top-20 之外。
    """
    chunk = FakeContextualChunk(text="那汉子来到门前", context="王慎在铁匠铺教训牛三")
    model = FakeCrossEncoder([0.9])

    rerank("王慎在铁匠铺遇到谁", [chunk], top_k=1, model=model)

    scored_doc = model.received_pairs[0][1]
    assert "王慎在铁匠铺教训牛三" in scored_doc, "重排必须看到上下文说明"
    assert "那汉子来到门前" in scored_doc, "原文也要保留"


def test_rerank_falls_back_to_text_without_context():
    """没做上下文增强的片段，行为和以前完全一样。"""
    chunk = FakeContextualChunk(text="纯原文", context="")
    model = FakeCrossEncoder([0.5])

    rerank("问题", [chunk], top_k=1, model=model)

    assert model.received_pairs[0][1] == "纯原文"


def test_returns_original_objects():
    """返回的必须是原对象本身，不是副本——调用方还要用上面的
    novel/chunk_id 等字段去查相邻片段、渲染出处卡片。
    """
    a, b = FakeChunk("A"), FakeChunk("B")
    model = FakeCrossEncoder([0.1, 0.9])

    result = rerank("问题", [a, b], top_k=2, model=model)

    assert result[0] is b
    assert result[1] is a
