"""交叉编码器重排（reranker）：给召回的候选片段做精细的相关性打分。

双编码器 vs 交叉编码器：检索系统最核心的一个架构权衡
--------------------------------------------------------
这两种模型都是在算"这个问题和这段文本有多相关"，但算法完全不同，
各自的优缺点正好互补——所以真实系统一般两个都用，分两个阶段。

**双编码器（bi-encoder，就是本项目 embedder.py 用的那种）**

    问题 ──→ [编码器] ──→ 向量 A  ┐
                                   ├─→ 算余弦相似度
    文档 ──→ [编码器] ──→ 向量 B  ┘

    问题和文档**各自独立**编码，编码完了才做一次简单的向量比较。

    ✅ 文档向量可以**预先算好存起来**（本项目 32730 个片段的向量都在
       novel_chunks.embedding 里）。查询时只需编码问题这一次，剩下的是
       纯向量运算，几万个片段几毫秒就扫完。
    ❌ 精度有限。因为编码文档的时候**根本不知道用户会问什么**，只能把整段
       文本压成一个固定的向量，压缩过程中必然丢信息。

**交叉编码器（cross-encoder，本模块用的）**

    [问题 + 文档] 拼在一起 ──→ [编码器] ──→ 相关性分数

    问题和文档**拼成一段**送进模型，模型内部的注意力机制能让问题里的每个词
    和文档里的每个词直接交互——"绰号"这个词可以直接"看到"文档里的"二愣子"。

    ✅ 精度高得多。它能判断"这段话是不是真的在回答这个问题"，而不只是
       "这段话和这个问题主题相近"。
    ❌ **没法预计算**。每换一个问题，所有文档都要重新算一遍。对 32730 个片段
       跑一遍要几分钟——完全不可能用来做全库检索。

**所以标准做法是两阶段**：

    32730 个片段
        ↓ 双编码器（快，粗）+ BM25
    20 个候选          ← 召回阶段：宁可多召回，别漏
        ↓ 交叉编码器（慢，精）
    5 个最相关          ← 重排阶段：从候选里精挑

只对 20 个候选跑交叉编码器，耗时可以接受；而这 20 个里如果本来就没有正确
答案，重排也救不回来——**重排解决的是"排序不准"，不是"召回不到"**。
所以评测时 Recall@20 不变、MRR 上升，才是重排生效的典型信号。
"""
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL

_model: CrossEncoder | None = None


def load_reranker(model_name: str = RERANKER_MODEL) -> CrossEncoder:
    """加载交叉编码器，进程内只加载一次。

    和 embedder.py 同样的理由：优先用本地缓存，避免每次都去 huggingface.co
    校验版本——网络不通时那会白等 20 秒以上，即使模型已经完整缓存在本地。
    """
    global _model
    if _model is None:
        try:
            _model = CrossEncoder(model_name, max_length=512, local_files_only=True)
        except Exception:
            # 本地没有（或缓存不完整）：联网下载一次，之后就走上面的快路径
            _model = CrossEncoder(model_name, max_length=512)
    return _model


def rerank(
    question: str,
    candidates: list,
    top_k: int,
    model: CrossEncoder | None = None,
) -> list:
    """用交叉编码器给候选片段重新打分排序，返回最相关的 top_k 个。

    candidates 是 SourceChunk 列表（只要有 .text 属性即可，不依赖具体类型，
    方便测试时传假对象）。返回的是原对象，顺序按相关性重排。
    """
    return [
        chunk
        for chunk, _ in rerank_with_scores(question, candidates, top_k, model)
    ]


def rerank_with_scores(
    question: str,
    candidates: list,
    top_k: int,
    model: CrossEncoder | None = None,
) -> list[tuple[object, float]]:
    """与 ``rerank`` 相同，但保留分数，供可视化评测比较重排前后名次。"""
    if not candidates:
        return []
    model = model or load_reranker()

    # 交叉编码器的输入是「问题-文档」对，模型内部会把两者拼起来做注意力交互。
    #
    # **用 indexed_text 而不是 text**——这是踩过的坑：Contextual Retrieval
    # 把上下文说明加进了索引（所以召回变好了），但重排如果拿原文重新打分，
    # 就看不到那些说明，等于把增强的效果整个抵消。实测同一个片段：
    #     重排给【原文】       0.0055
    #     重排给【说明+原文】  0.9990    ← 差 180 倍
    # 结果就是正确片段被重排从第 5 名压到 top-20 之外。
    #
    # 没有 context 的片段（没做增强、或没开这个功能），indexed_text 就等于 text，
    # 行为和以前完全一样。
    pairs = [(question, getattr(c, "indexed_text", c.text)) for c in candidates]
    scores = model.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [(chunk, float(score)) for chunk, score in ranked[:top_k]]
