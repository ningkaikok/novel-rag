"""/api/ask 与结构化会话事实的接线（M3.6 唯一剩下的功能项）。

和滚动摘要不同，这里不需要开关测试——它默认随对话背景一起生效，不调用模型，
只在有历史时做一次轻量查询。这里只测"接线对不对"：extract_session_facts 的
结果真的传到了 build_prompt，并且在 trace 里可见。extract_session_facts 本身
的行为在 test_session_facts.py 里测。
"""

import backend.main as main
from tests.backend.test_endpoints import _FakeRag


def _turns(n: int) -> list[dict]:
    return [
        {
            "turn_index": i,
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"第{i}轮",
            "sources": [{"novel": "雾隐山庄", "chunk_id": 0, "chapter_title": None}]
            if i % 2 == 1
            else None,
        }
        for i in range(n)
    ]


def test_ask_passes_extracted_facts_into_the_prompt(monkeypatch):
    seen: dict = {}

    class _Rag(_FakeRag):
        def build_prompt(self, question, sources, history=None, summary=None, facts_text=None):
            seen["facts_text"] = facts_text
            return f"问题：{question}"

    main.state["rag"] = _Rag()
    main.state["model"] = "fake-model"
    monkeypatch.setattr(main, "QUERY_REWRITE_ENABLED", False)
    monkeypatch.setattr(main, "load_turns", lambda _sid: _turns(2))
    monkeypatch.setattr(
        main, "extract_session_facts", lambda turns: "__called_with_turns__" and object()
    )
    monkeypatch.setattr(main, "format_facts_line", lambda facts: "当前小说：《雾隐山庄》")
    monkeypatch.setattr(main, "generate_ollama_prompt_stream", lambda p, model: iter(["好"]))

    from fastapi.testclient import TestClient

    resp = TestClient(main.app).post(
        "/api/ask", json={"question": "他后来呢", "session_id": "s-facts"}
    )

    assert resp.status_code == 200
    assert seen["facts_text"] == "当前小说：《雾隐山庄》"
    assert "对话背景" in resp.text
    assert "当前小说" in resp.text


def test_ask_without_history_never_touches_session_facts(monkeypatch):
    """没有历史（第一轮、或没有 session_id）就不该调用提取逻辑——零开销。"""
    main.state["rag"] = _FakeRag()
    main.state["model"] = "fake-model"
    monkeypatch.setattr(main, "QUERY_REWRITE_ENABLED", False)
    monkeypatch.setattr(main, "load_turns", lambda _sid: [])
    monkeypatch.setattr(
        main,
        "extract_session_facts",
        lambda _turns: (_ for _ in ()).throw(AssertionError("没有历史就不该提取事实")),
    )
    monkeypatch.setattr(main, "generate_ollama_prompt_stream", lambda p, model: iter(["好"]))

    from fastapi.testclient import TestClient

    resp = TestClient(main.app).post(
        "/api/ask", json={"question": "顾长风是谁", "session_id": "s-empty"}
    )

    assert resp.status_code == 200


def test_ask_facts_extraction_failure_does_not_break_the_answer(monkeypatch):
    """提取失败（比如数据库暂时不可用）不该让回答本身失败——这是纯增强信息。"""
    main.state["rag"] = _FakeRag()
    main.state["model"] = "fake-model"
    monkeypatch.setattr(main, "QUERY_REWRITE_ENABLED", False)
    monkeypatch.setattr(main, "load_turns", lambda _sid: _turns(2))
    monkeypatch.setattr(
        main,
        "extract_session_facts",
        lambda _turns: (_ for _ in ()).throw(RuntimeError("坏了")),
    )
    monkeypatch.setattr(main, "generate_ollama_prompt_stream", lambda p, model: iter(["好"]))

    from fastapi.testclient import TestClient

    resp = TestClient(main.app).post(
        "/api/ask", json={"question": "他后来呢", "session_id": "s-broken"}
    )

    assert resp.status_code == 200, "提取失败不该让 SSE 流中途异常"
