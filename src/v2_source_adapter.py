"""把 V2 检索命中转换成 Agent Lab 现有工具管线认识的 ``SourceChunk``。

只服务 Agent Lab 新增的 ``search_documents`` 工具（见 ``agent_lab.py``），不改动
V1 ``novel_chunks`` 的检索/融合/生成主链路——那条链路（以及 ``read_neighbors``/
``get_chapter``/``_resolve_novel``）继续依赖"``SourceChunk.novel``/``chunk_id``
能在 ``novel_chunks`` 表里唯一命中"这个假设。这里转换出来的 ``SourceChunk`` 只
用于 Agent Lab 的 ``answer_with_citations`` 展示和 ``build_prompt``，不会被那几个
只认 ``novel_chunks`` 的工具拿去再查一次库。
"""

from __future__ import annotations

from chunk_model import SourceChunk
from v2_retrieval import V2SearchHit


def source_chunk_from_v2_hit(hit: V2SearchHit) -> SourceChunk:
    """document 标题当 ``novel``、``ordinal`` 当 ``chunk_id``、章节路径拼成标题。"""
    return SourceChunk(
        novel=hit.document.title,
        chunk_id=hit.chunk.ordinal,
        text=hit.chunk.text,
        distance=hit.distance,
        chapter_title="/".join(hit.chunk.section_path) or None,
        context=hit.chunk.context,
    )


def fuse_v2_hits(*ranked_hit_lists: list[V2SearchHit], rrf_k: int = 60) -> list[V2SearchHit]:
    """通用 RRF：按 ``chunk.id``（V2 全局唯一）去重融合多路已排序候选。

    数学上和 ``rag.py`` 里 V1 的 RRF 融合是同一套（按名次取 ``1/(k+rank)`` 累加），
    只是这里用 V2 chunk 的全局唯一 id 当 key——V2 的 ``document.title`` 不保证唯一，
    直接拿转换后 SourceChunk 的 ``(novel, chunk_id)`` 当 key 融合会有误判重复的风险，
    所以融合放在转换成 SourceChunk 之前做。
    """
    scores: dict[str, float] = {}
    items: dict[str, V2SearchHit] = {}
    for ranked in ranked_hit_lists:
        for rank, hit in enumerate(ranked, start=1):
            key = hit.chunk.id
            scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + rank)
            items.setdefault(key, hit)
    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [items[key] for key in ranked_keys]
