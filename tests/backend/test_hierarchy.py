"""M3 层级索引测试；数据库与 embedding 都用轻量替身。"""

from unittest.mock import MagicMock, patch

import rag
from hierarchy import build_hierarchy_nodes, extract_summary, is_global_question
from loader import Chunk


def _chunk(chunk_id: int, text: str, chapter: str | None) -> Chunk:
    return Chunk("雾隐山庄", chunk_id, text, chapter)


def test_builds_chapter_nodes_and_one_novel_node():
    chunks = [
        _chunk(0, "顾长风来到山庄。", "第一章 入庄"),
        _chunk(1, "他发现门锁有异样。", "第一章 入庄"),
        _chunk(2, "真相终于揭晓。", "第二章 真相"),
    ]

    nodes = build_hierarchy_nodes(chunks)

    assert [node.level for node in nodes] == ["chapter", "chapter", "novel"]
    assert (nodes[0].start_chunk_id, nodes[0].end_chunk_id) == (0, 1)
    assert (nodes[1].start_chunk_id, nodes[1].end_chunk_id) == (2, 2)
    assert (nodes[-1].start_chunk_id, nodes[-1].end_chunk_id) == (0, 2)
    assert "第一章 入庄" in nodes[-1].summary


def test_untitled_chunks_are_kept_as_virtual_chapter(monkeypatch):
    monkeypatch.setattr("hierarchy.HIERARCHY_UNTITLED_CHUNKS", 2)
    chunks = [_chunk(index, f"无标题正文{index}", None) for index in range(5)]

    chapters = [n for n in build_hierarchy_nodes(chunks) if n.level == "chapter"]

    assert len(chapters) == 3
    assert chapters[0].title == "未命名章节（片段 0–1）"
    assert (chapters[-1].start_chunk_id, chapters[-1].end_chunk_id) == (4, 4)


def test_summary_samples_the_end_instead_of_only_the_beginning():
    texts = [f"阶段{i}" for i in range(20)]

    summary = extract_summary("成长线", texts, max_chars=200)

    assert "阶段0" in summary
    assert "阶段19" in summary


def test_global_question_detection_is_conservative():
    assert is_global_question("主角在全书中的成长有什么变化？") is True
    assert is_global_question("比较两本书的主题") is True
    assert is_global_question("顾长风在哪一章进山庄？") is False


class _RowsConn:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, _sql, params=None):
        ids = params[1]
        rows = [
            {
                "novel": params[0],
                "chunk_id": chunk_id,
                "chapter_title": "第二章 成长",
                "text": f"原文证据{chunk_id}",
                "context": "",
            }
            for chunk_id in ids
        ]
        result = MagicMock()
        result.fetchall.return_value = rows
        return result


def test_hierarchy_hit_maps_back_to_original_evidence():
    """摘要只负责导航；最终交给回答模型的 SourceChunk 必须来自 novel_chunks。"""
    embedder = MagicMock()
    embedder.encode.return_value = [[0.1, 0.2]]
    with patch.object(rag, "has_index", return_value=True):
        service = rag.NovelRAG(embedder=embedder)
    chapter_hit = {
        "novel": "雾隐山庄",
        "level": "chapter",
        "title": "第二章 成长",
        "start_chunk_id": 10,
        "end_chunk_id": 14,
        "distance": 0.08,
    }

    with (
        patch.object(rag, "search_hierarchy", return_value=[chapter_hit]),
        patch.object(rag, "connect", return_value=_RowsConn()),
    ):
        sources, hits = service.hierarchy_retrieve(
            "主角在全书中的成长变化", named_novels=["雾隐山庄"]
        )

    assert [source.chunk_id for source in sources] == [10, 12, 14]
    assert all(source.text.startswith("原文证据") for source in sources)
    assert hits == [chapter_hit]
