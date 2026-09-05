"""M3.5-④ 在线配置快照（run_config）：落库内容、隐私红线、schema 幂等迁移。

全部 mock：save_turn / load_turns / generate_ollama_prompt_stream 都被替换，
不连真实数据库，也不调用任何模型。
"""

import contextlib
import json
import re

from fastapi.testclient import TestClient

import backend.main as main
import postgres


class _FakeRag:
    """最小 NovelRAG 替身，形状与 test_endpoints 的同名类一致。"""

    def retrieve_hybrid_stream(self, question, top_k):
        yield "step", {"step": "理解问题", "detail": "识别到你在问《雾隐山庄》", "ms": 5}
        yield (
            "result",
            [
                type(
                    "S",
                    (),
                    {
                        "novel": "雾隐山庄",
                        "chunk_id": 0,
                        "chapter_title": "第一章 夜雨访客",
                        # 刻意选一句"原文证据"，用于隐私断言：快照里绝不能出现它
                        "text": "顾长风所患的是奇毒蚀骨散，毒性极深",
                    },
                )()
            ],
        )

    def build_answer_context(self, sources):
        return sources, None

    def build_prompt(self, question, sources, history=None, summary=None):
        return f"[证据] 顾长风所患的是奇毒蚀骨散，毒性极深\n问题：{question}"


def _setup(monkeypatch, captured: list):
    main.state["rag"] = _FakeRag()
    main.state["model"] = "fake-model"
    monkeypatch.setattr(
        main, "generate_ollama_prompt_stream", lambda prompt, model: iter(["回答。"])
    )
    monkeypatch.setattr(main, "next_turn_index", lambda _sid: 0)
    monkeypatch.setattr(main, "save_turn", lambda *a, **k: captured.append((a, k)))


def test_ask_saves_run_config_snapshot(client, monkeypatch):
    captured: list = []
    _setup(monkeypatch, captured)

    resp = client.post(
        "/api/ask", json={"question": "顾长风得了什么病", "session_id": "s-run"}
    )

    assert resp.status_code == 200
    assistant_args, assistant_kwargs = captured[1]
    snapshot = assistant_kwargs["run_config"]
    assert snapshot is not None
    # 快照必须覆盖 M3.5-④ 要求的字段
    # 字面量而不是引用常量：模板文本改了就必须有人来改这一行，这正是它的作用。
    # v1 → v2：M3.6 引入带「对话背景」段的模板。
    assert snapshot["prompt_template_version"] == "v2"
    assert snapshot["answer_mode"] in ("auto", "grounded", "free")
    assert snapshot["route_reason"]
    assert snapshot["generate_model"] == "fake-model"
    assert isinstance(snapshot["rerank_enabled"], bool)
    assert "reranker_model" in snapshot
    # 最终状态并入快照且同步到 status 列
    assert snapshot["final_status"] == "complete"
    assert assistant_kwargs["status"] == "complete"


def test_run_config_snapshot_respects_privacy_red_lines(client, monkeypatch):
    """隐私红线：快照序列化后不含 API Key 痕迹、不含原文片段。"""
    captured: list = []
    _setup(monkeypatch, captured)

    client.post("/api/ask", json={"question": "顾长风得了什么病", "session_id": "s-privacy"})

    _, kwargs = captured[1]
    serialized = json.dumps(kwargs["run_config"], ensure_ascii=False)
    # 不含密钥痕迹（无论真实 key 长什么样，都不该出现在快照里）
    for marker in ("KEY", "key=", "sk-", "token"):
        assert marker not in serialized, f"快照泄漏了密钥痕迹：{marker}"
    # 不含检索到的原文片段（证据只应出现在 sources 列，不该复制进快照）
    assert "蚀骨散" not in serialized
    assert "顾长风所患的是奇毒蚀骨散" not in serialized
    # 也不含完整 prompt
    assert "[证据]" not in serialized


def test_free_mode_run_config_records_mode_and_reason(client, monkeypatch):
    captured: list = []
    main.state["rag"] = None  # 自由问答不依赖索引
    main.state["model"] = "fake-model"
    monkeypatch.setattr(
        main, "generate_ollama_prompt_stream", lambda prompt, model: iter(["你好！"])
    )
    monkeypatch.setattr(main, "next_turn_index", lambda _sid: 0)
    monkeypatch.setattr(main, "save_turn", lambda *a, **k: captured.append((a, k)))

    resp = client.post(
        "/api/ask", json={"question": "你好", "mode": "free", "session_id": "s-free"}
    )

    assert resp.status_code == 200
    _, kwargs = captured[1]
    snapshot = kwargs["run_config"]
    assert snapshot["answer_mode"] == "free"
    assert snapshot["route_reason"]


def test_error_status_recorded_in_snapshot(client, monkeypatch):
    """生成中途抛错时，落库状态必须是 error 而不是 complete。"""
    captured: list = []

    def broken_generate(prompt, model):
        raise RuntimeError("模型进程崩溃")
        yield  # pragma: no cover

    main.state["rag"] = _FakeRag()
    main.state["model"] = "fake-model"
    monkeypatch.setattr(main, "generate_ollama_prompt_stream", broken_generate)
    monkeypatch.setattr(main, "next_turn_index", lambda _sid: 0)
    monkeypatch.setattr(main, "save_turn", lambda *a, **k: captured.append((a, k)))

    # 异常穿过 StreamingResponse 会冒泡到测试客户端；finally 里的落库已经发生
    with contextlib.suppress(Exception):
        client.post("/api/ask", json={"question": "问", "session_id": "s-err"})

    assert captured, "异常路径也必须落库"
    _, kwargs = captured[1]
    assert kwargs["status"] == "error"
    assert kwargs["run_config"]["final_status"] == "error"
    assert "RuntimeError" in kwargs["run_config"]["error"]
    # 错误信息不携带用户问题原文
    assert "问：" not in json.dumps(kwargs["run_config"], ensure_ascii=False)


# ---------------------------------------------------------------- schema 迁移


class _SchemaConn:
    def __init__(self):
        self.sql = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sql.append(sql)
        return self


def test_ensure_chat_schema_adds_run_config_column_idempotently(monkeypatch):
    conn = _SchemaConn()
    monkeypatch.setattr(postgres, "connect", lambda: conn)

    postgres.ensure_chat_schema()

    alter_sql = [sql for sql in conn.sql if "ALTER TABLE chat_turns" in sql]
    assert any("ADD COLUMN IF NOT EXISTS run_config JSONB" in sql for sql in alter_sql), (
        "幂等迁移必须包含 run_config 补列"
    )


# ---------------------------------------------------------------- 历史透出


def test_session_history_exposes_run_config(client, monkeypatch):
    turns = [
        {"turn_index": 0, "role": "user", "content": "问", "status": "complete"},
        {
            "turn_index": 1,
            "role": "assistant",
            "content": "答",
            "status": "complete",
            "run_config": {"answer_mode": "grounded", "generate_model": "qwen2.5:7b"},
        },
    ]
    monkeypatch.setattr(main, "load_turns", lambda session_id: turns)

    body = TestClient(main.app).get("/api/sessions/s1").json()

    # 旧记录没有 run_config → 默认 None；新记录原样透出
    assert body["turns"][0]["run_config"] is None
    assert body["turns"][1]["run_config"]["answer_mode"] == "grounded"


def test_agent_steps_carry_shared_run_id(client, monkeypatch):
    """同一次 Agent 运行的所有步骤共享一个 run_id（M3.5-③ 轻量串联）。"""
    main.state["rag"] = object()
    main.state["model"] = "fake-model"

    def fake_run_agent(*_args, **_kwargs):
        yield (
            "agent_step",
            {
                "step": 1,
                "reason": "先搜索",
                "tool": "search_novels",
                "args": {},
                "observation": "找到 S1",
                "source_ids": ["S1"],
            },
        )
        yield (
            "agent_step",
            {
                "step": 2,
                "reason": "再读邻居",
                "tool": "read_neighbors",
                "args": {},
                "observation": "读到 S2",
                "source_ids": ["S2"],
            },
        )
        yield "sources", []
        yield "token", "答案"
        yield "done", {}

    monkeypatch.setattr(main, "run_agent", fake_run_agent)
    saved: list = []
    monkeypatch.setattr(main, "next_turn_index", lambda _sid: 0)
    monkeypatch.setattr(main, "save_turn", lambda *a, **k: saved.append((a, k)))

    body = (
        TestClient(main.app)
        .post(
            "/api/agent/ask", json={"question": "问", "max_steps": 3, "session_id": "s-agent"}
        )
        .text
    )

    run_ids = set(re.findall(r'"run_id": ?"(?P<rid>[0-9a-f]+)"', body))
    assert len(run_ids) == 1, "同一运行的所有步骤必须共享同一个 run_id"

    _, kwargs = saved[1]
    step_run_ids = {step["run_id"] for step in kwargs["agent_steps"]}
    assert step_run_ids == run_ids, "SSE 与落库的 run_id 必须一致"
