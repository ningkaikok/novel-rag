"""小书全文短路的单元测试（不连数据库，mock 掉查询）。"""
from unittest.mock import MagicMock, patch

import rag


class _FakeConn:
    def __init__(self, chars, rows):
        self._chars = chars
        self._rows = rows
        self._calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._calls += 1
        result = MagicMock()
        if "SUM(LENGTH" in sql:
            result.fetchone.return_value = {"chars": self._chars}
        else:
            result.fetchall.return_value = self._rows
        return result


ROWS = [
    {"novel": "雾隐山庄", "chunk_id": 0, "text": "第一段", "context": ""},
    {"novel": "雾隐山庄", "chunk_id": 1, "text": "第二段", "context": ""},
]


def _rag():
    with patch.object(rag, "has_index", return_value=True):
        return rag.NovelRAG(embedder=MagicMock())


def test_small_book_returns_all_chunks_in_reading_order():
    """够小就返回全部片段，且按原文顺序——给模型一个连贯的故事。"""
    with patch.object(rag, "connect", lambda: _FakeConn(1229, ROWS)):
        result = _rag()._full_text_chunks(["雾隐山庄"])

    assert result is not None
    assert [c.chunk_id for c in result] == [0, 1], "必须按 chunk_id 顺序"


def test_large_book_returns_none():
    """超过阈值就老实走检索。"""
    with patch.object(rag, "connect", lambda: _FakeConn(9_000_000, ROWS)):
        assert _rag()._full_text_chunks(["凡人修仙传"]) is None


def test_multiple_books_never_short_circuit():
    """跨书问题不能短路：两本各自很小，合起来也可能超窗口，
    而且混在一起会干扰模型判断。
    """
    with patch.object(rag, "connect", lambda: _FakeConn(100, ROWS)):
        assert _rag()._full_text_chunks(["书A", "书B"]) is None


def test_no_named_book_never_short_circuit():
    """没点名书时无法确定范围，只能老实检索。"""
    with patch.object(rag, "connect", lambda: _FakeConn(100, ROWS)):
        assert _rag()._full_text_chunks([]) is None
