"""检索可观测性回归：每一层排名都必须能在 trace 中被复盘。"""

import rag


def _source(chunk_id: int, distance: float) -> rag.SourceChunk:
    return rag.SourceChunk(
        novel="雾隐山庄",
        chunk_id=chunk_id,
        chapter_title="第一章",
        text=f"原文{chunk_id}",
        distance=distance,
    )


def test_trace_keeps_vector_bm25_rrf_and_rerank_rank_changes(monkeypatch):
    service = object.__new__(rag.NovelRAG)
    semantic = [_source(1, 0.1), _source(2, 0.2)]
    keyword = [_source(2, -8.0), _source(3, -6.0)]
    monkeypatch.setattr(service, "_named_novels", lambda _question: [])
    monkeypatch.setattr(service, "_full_text_chunks", lambda _novels: None)
    monkeypatch.setattr(
        service,
        "retrieve",
        lambda _question, top_k, only_novels: semantic,
    )
    monkeypatch.setattr(
        service,
        "keyword_retrieve",
        lambda _question, top_k, only_novels: keyword,
    )
    monkeypatch.setattr(
        service,
        "positional_retrieve",
        lambda _question, top_k, hint_novels: [],
    )
    monkeypatch.setattr(rag, "HIERARCHY_ENABLED", False)
    monkeypatch.setattr(rag, "RERANK_ENABLED", True)
    monkeypatch.setattr(
        rag,
        "rerank_with_scores",
        lambda _question, candidates, _top_k: [
            (next(c for c in candidates if c.chunk_id == 3), 0.95),
            (next(c for c in candidates if c.chunk_id == 2), 0.80),
            (next(c for c in candidates if c.chunk_id == 1), 0.20),
        ],
    )

    events = list(service.retrieve_hybrid_stream("庄主是谁", top_k=2))
    steps = {
        payload["stage_key"]: payload
        for kind, payload in events
        if kind == "step" and payload.get("stage_key")
    }

    assert {"vector", "bm25", "rrf", "rerank"} <= set(steps)
    assert [c["chunk_id"] for c in steps["vector"]["candidates"]] == [1, 2]
    assert steps["bm25"]["candidates"][0]["score"] == 8.0
    assert steps["rrf"]["candidates"][0]["chunk_id"] == 2

    reranked = steps["rerank"]["candidates"]
    assert reranked[0]["chunk_id"] == 3
    assert reranked[0]["previous_rank"] == 3
    assert reranked[0]["selected"] is True
    assert reranked[2]["selected"] is False
    assert all(step["ms"] >= 0 for step in steps.values())
