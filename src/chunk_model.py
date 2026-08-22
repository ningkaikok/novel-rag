"""检索结果的数据类与候选 trace 记录。

初学者可以把这里看成 RAG 的“统一数据语言”：无论哪一路召回（向量、BM25、
结构性、层级摘要），最终都折算成 ``SourceChunk`` 才能进入融合和重排；

    novel        书名（文件名式全名，用于定位数据库行）
    chunk_id     片段编号，和 novel 一起构成主键
    text         原文片段——唯一允许作为“证据”交给模型的内容
    distance     各路分数的统一出口：约定“越小越好”（BM25 分数取负号存进来）
    chapter_title / context   可选的增强元数据，见字段注释

``_trace_candidates`` 则负责把内部候选压成轻量的排名记录，供前端“思考过程”
展示和评测复盘使用，避免把大段原文塞进 SSE/JSONB。
"""
from dataclasses import dataclass


@dataclass
class SourceChunk:
    novel: str
    chunk_id: int
    text: str
    distance: float
    # 章节识别是增强元数据：旧索引和无规范标题的 txt 都可能为空。
    chapter_title: str | None = None
    # Contextual Retrieval 生成的上下文说明（没做增强时是空串）。
    # 重排要用它（见 reranker.rerank 里 indexed_text 的说明），
    # 但 build_prompt 只用 text——不把 AI 生成的说明当原文依据给模型。
    context: str = ""

    @property
    def indexed_text(self) -> str:
        """建索引时用的文本，也是重排该看到的文本。

        必须和索引保持一致：索引的是「说明 + 原文」，重排如果只看原文，
        就会把上下文增强的效果整个抵消掉。
        """
        return f"{self.context}\n{self.text}" if self.context else self.text


_TRACE_CANDIDATE_LIMIT = 10


def _trace_candidates(
    sources: list[SourceChunk],
    *,
    score_label: str,
    score_of,
    previous_ranks: dict[tuple[str, int], int] | None = None,
    selected_count: int = 0,
) -> list[dict]:
    """把内部候选压成适合 SSE/JSONB 的轻量排名记录，避免保存大段原文。"""
    payload: list[dict] = []
    for rank, source in enumerate(sources[:_TRACE_CANDIDATE_LIMIT], start=1):
        key = (source.novel, source.chunk_id)
        score = score_of(source, key)
        payload.append(
            {
                "novel": source.novel,
                "chunk_id": source.chunk_id,
                "chapter_title": source.chapter_title,
                "rank": rank,
                "score": round(float(score), 6) if score is not None else None,
                "score_label": score_label,
                "previous_rank": (previous_ranks or {}).get(key),
                "selected": bool(selected_count and rank <= selected_count),
            }
        )
    return payload
