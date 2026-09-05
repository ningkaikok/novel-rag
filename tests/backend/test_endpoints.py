"""关键端点的响应形状/基本行为测试。

全部走 TestClient，不依赖真实数据库、Ollama 或云端 key——按需要 monkeypatch
掉会碰真实资源的函数（load_embedder、connect、_list_ollama_models 等）。
主要验证这一批 response_model 改造之后，真实返回值仍然符合声明的形状。
"""

import pytest

import backend.main as main


def _index_task(status="running", *, task_id="task-1", result=None):
    return {
        "id": task_id,
        "status": status,
        "stage": "embedding" if status == "running" else status,
        "progress": 35 if status == "running" else 100,
        "message": "正在建立索引" if status == "running" else "任务结束",
        "error": "数据库断开" if status == "failed" else None,
        "force": False,
        "retry_of": None,
        "result": result,
        "created_at": "2026-08-09T00:00:00+00:00",
        "started_at": "2026-08-09T00:00:00+00:00",
        "finished_at": None if status == "running" else "2026-08-09T00:00:01+00:00",
    }


def test_health_without_rag_loaded(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "ready": False}


def test_health_with_rag_loaded(client):
    main.state["rag"] = object()  # 只关心"有没有"，不需要真的是 NovelRAG
    resp = client.get("/api/health")
    assert resp.json() == {"ok": True, "ready": True}


def test_list_books_reads_novels_dir(client, tmp_path, monkeypatch):
    (tmp_path / "凡人修仙传.txt").write_text("……")
    (tmp_path / "诡秘之主.txt").write_text("……")
    (tmp_path / "not-a-novel.md").write_text("忽略非 txt 文件")
    monkeypatch.setattr(main, "NOVELS_DIR", tmp_path)

    resp = client.get("/api/books")

    assert resp.status_code == 200
    assert resp.json() == {"books": ["凡人修仙传", "诡秘之主"]}


def test_upload_saves_file_and_returns_background_task(client, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "NOVELS_DIR", tmp_path)

    def start_task(**kwargs):
        kwargs["prepare"]()
        return _index_task()

    monkeypatch.setattr(main, "_start_index_task", start_task)
    resp = client.post(
        "/api/books",
        files={"files": ("新小说.txt", "第一章 开始\n正文", "text/plain")},
    )

    assert resp.status_code == 200
    assert resp.json()["saved"] == ["新小说"]
    assert resp.json()["task"]["status"] == "running"
    assert (tmp_path / "新小说.txt").read_text() == "第一章 开始\n正文"


def test_delete_removes_file_then_returns_cleanup_task(client, tmp_path, monkeypatch):
    target = tmp_path / "要删除.txt"
    target.write_text("正文")
    monkeypatch.setattr(main, "NOVELS_DIR", tmp_path)

    def start_task(**kwargs):
        kwargs["prepare"]()
        return _index_task()

    monkeypatch.setattr(main, "_start_index_task", start_task)
    resp = client.delete("/api/books/要删除")

    assert resp.status_code == 200
    assert resp.json()["deleted"] == "要删除"
    assert not target.exists()


def test_reindex_starts_incremental_or_forced_task(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        main,
        "_start_index_task",
        lambda **kwargs: calls.append(kwargs) or _index_task(),
    )

    assert client.post("/api/reindex").status_code == 200
    assert client.post("/api/reindex?force=true").status_code == 200
    assert calls[0]["force"] is False
    assert calls[1]["force"] is True


def test_index_task_status_cancel_and_retry_endpoints(client, monkeypatch):
    failed = _index_task("failed")
    retried = _index_task("running", task_id="task-2")
    monkeypatch.setattr(main.index_tasks, "current", lambda: failed)
    monkeypatch.setattr(main.index_tasks, "get", lambda task_id: failed)
    monkeypatch.setattr(main.index_tasks, "cancel", lambda task_id: failed)
    monkeypatch.setattr(main, "_start_index_task", lambda **kwargs: retried)

    assert client.get("/api/index-tasks/current").json()["status"] == "failed"
    assert client.get("/api/index-tasks/task-1").status_code == 200
    assert client.post("/api/index-tasks/task-1/cancel").status_code == 200
    retry = client.post("/api/index-tasks/task-1/retry")
    assert retry.status_code == 200
    assert retry.json()["id"] == "task-2"


def test_models_shape(client, monkeypatch):
    monkeypatch.setattr(main, "_list_ollama_models", lambda: ["qwen2.5:7b"])
    monkeypatch.setattr(main.claude_cli, "claude_model_options", lambda: ["claude:sonnet"])
    monkeypatch.setattr(main.zhipu, "model_options", lambda: ["glm:glm-4-flash"])
    main.state["model"] = "qwen2.5:7b"

    resp = client.get("/api/models")

    assert resp.status_code == 200
    assert resp.json() == {
        "models": ["qwen2.5:7b", "claude:sonnet", "glm:glm-4-flash"],
        "current": "qwen2.5:7b",
    }


def test_set_model_rejects_unavailable_model(client, monkeypatch):
    monkeypatch.setattr(main, "_list_ollama_models", lambda: ["qwen2.5:7b"])
    monkeypatch.setattr(main.claude_cli, "claude_model_options", lambda: [])
    monkeypatch.setattr(main.zhipu, "model_options", lambda: [])
    main.state["model"] = "qwen2.5:7b"

    resp = client.post("/api/model", json={"model": "不存在的模型"})

    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "model_unavailable"
    assert "不可用" in body["error"]["message"]


def test_set_model_accepts_available_model(client, monkeypatch):
    monkeypatch.setattr(main, "_list_ollama_models", lambda: ["qwen2.5:3b", "qwen2.5:7b"])
    monkeypatch.setattr(main.claude_cli, "claude_model_options", lambda: [])
    monkeypatch.setattr(main.zhipu, "model_options", lambda: [])
    main.state["model"] = "qwen2.5:7b"

    resp = client.post("/api/model", json={"model": "qwen2.5:3b"})

    assert resp.status_code == 200
    assert resp.json() == {"current": "qwen2.5:3b"}
    assert main.state["model"] == "qwen2.5:3b"  # 真的切换了，不只是回显


class _FakeConn:
    """假的 psycopg 连接：execute().fetchall() 返回预设好的行。"""

    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        return self

    def fetchall(self):
        return self._rows


def test_search_shape(client, monkeypatch):
    monkeypatch.setattr(main, "has_index", lambda: True)
    rows = [
        {
            "novel": "雾隐山庄",
            "chunk_id": 0,
            "chapter_title": "第一章 风雪来客",
            "text": "顾长风是雾隐山庄的庄主",
        }
    ]
    monkeypatch.setattr(main, "connect", lambda: _FakeConn(rows))

    resp = client.get("/api/search", params={"q": "庄主"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "庄主"
    assert body["total"] == 1
    assert body["results"] == [
        {
            "novel": "雾隐山庄",
            "chunk_id": 0,
            "chapter_title": "第一章 风雪来客",
            "text": "顾长风是雾隐山庄的庄主",
            "match_count": 1,
        }
    ]


def test_search_without_index_returns_409(client, monkeypatch):
    monkeypatch.setattr(main, "has_index", lambda: False)

    resp = client.get("/api/search", params={"q": "庄主"})

    assert resp.status_code == 409


def test_get_session_shape(client, monkeypatch):
    turns = [
        {
            "turn_index": 0,
            "role": "user",
            "content": "雾隐山庄的庄主是谁",
            "sources": None,
            "trace": None,
            "status": "complete",
        },
        {
            "turn_index": 1,
            "role": "assistant",
            "content": "顾长风。",
            "sources": [{"novel": "雾隐山庄", "chunk_id": 0, "text": "……"}],
            "trace": [{"step": "理解问题", "detail": "……"}],
            "status": "complete",
        },
    ]
    monkeypatch.setattr(main, "load_turns", lambda session_id: turns)

    resp = client.get("/api/sessions/some-session-id")

    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "some-session-id"
    assert len(body["turns"]) == 2
    assert body["turns"][1]["content"] == "顾长风。"


def _agent_turn(agent_steps):
    """一条 Agent Lab 的历史记录（结构化字段由参数决定合法与否）。"""
    return {
        "turn_index": 1,
        "role": "assistant",
        "content": "顾长风[1]。",
        "sources": None,
        "trace": None,
        "agent_steps": agent_steps,
        "status": "complete",
    }


def test_get_session_restores_agent_lab_steps(client, monkeypatch):
    """Agent Lab 的历史轮次要能原样读回，步骤卡片才能在刷新后恢复。"""
    steps = [
        {
            "step": 1,
            "reason": "先搜索",
            "tool": "search_novels",
            "args": {"query": "庄主"},
            "observation": "找到 3 段",
            "source_ids": ["S1"],
        }
    ]
    monkeypatch.setattr(main, "load_turns", lambda _sid: [_agent_turn(steps)])

    body = client.get("/api/sessions/s1").json()

    assert body["turns"][0]["agent_steps"][0]["tool"] == "search_novels"


def test_broken_structured_payload_degrades_that_turn_not_the_whole_session(
    client, monkeypatch
):
    """一条坏记录只丢自己的结构化字段，不能让整段对话都读不出来。

    `sources`/`trace`/`agent_steps` 是 jsonb 列，形状随代码版本演进（TraceStep
    先后加过 ms/stage_key/candidates，chat_turns 后来才加 agent_steps）。旧记录
    一旦不满足新模型，整个会话历史就会 500——用户丢的不是某一轮的步骤卡片，
    而是**整段对话**。这里固定住"降级而不是全灭"的行为。
    """
    good = {
        "turn_index": 0,
        "role": "user",
        "content": "庄主是谁",
        "sources": None,
        "trace": None,
        "agent_steps": None,
        "status": "complete",
    }
    # 缺 reason/observation/source_ids，模拟"字段后来才加上"的旧记录
    broken = _agent_turn([{"step": 1, "tool": "search_novels"}])
    monkeypatch.setattr(main, "load_turns", lambda _sid: [good, broken])

    resp = client.get("/api/sessions/s1")

    assert resp.status_code == 200, "坏掉的结构化字段不该让整个会话读不出来"
    turns = resp.json()["turns"]
    assert len(turns) == 2
    assert turns[0]["content"] == "庄主是谁"
    # 正文保住了，只有解析不了的结构化字段被丢弃
    assert turns[1]["content"] == "顾长风[1]。"
    assert turns[1]["agent_steps"] is None


def test_unparseable_core_fields_still_surface_as_error():
    """正文本身坏掉（不是结构化字段的形状演进）时不能静默吞掉，仍要抛出。

    直接测 `_restore_turn` 而不走 HTTP：这条路径的正确行为是让异常冒泡给全局
    异常处理器转成 500，而 TestClient 默认会把服务端异常重新抛出、不返回响应，
    从端点层断言反而测不出意图。
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        main._restore_turn({"turn_index": "not-an-int", "role": "user"})


class _FakeRag:
    """/api/ask 依赖的最小 NovelRAG 替身：只实现路由会调用到的几个方法。"""

    def retrieve_hybrid_stream(self, question, top_k):
        """真实实现是个生成器：每完成一步 yield 一条 step，最后 yield result。"""
        yield "step", {"step": "理解问题", "detail": "识别到你在问《雾隐山庄》", "ms": 12}
        yield "step", {"step": "精排", "detail": "取最相关的 3 段", "ms": 2100}
        yield (
            "result",
            [
                type(
                    "S",
                    (),
                    {
                        "novel": "雾隐山庄",
                        "chunk_id": 0,
                        "chapter_title": "第一章 风雪来客",
                        "text": "顾长风是庄主",
                    },
                )()
            ],
        )

    def build_answer_context(self, sources):
        """M3.4 起的上下文组装统一入口；off 档等价旧 expand_neighbors 且无 trace 步骤。"""
        return sources, None

    def build_prompt(self, question, sources, history=None, summary=None):
        return f"问题：{question}"


class _LibraryRag:
    """目录问题不应触发普通向量/BM25 检索或模型生成。"""

    def library_answer(self, question):
        assert question == "现在一共有几部小说"
        return (
            "当前书架一共有 4 部小说：《凡人修仙传》、《诡秘之主》、《降龙》、《雾隐山庄》。"
        )

    def retrieve_hybrid_stream(self, *_args, **_kwargs):
        raise AssertionError("目录问题不应进入片段检索")


def test_ask_streams_trace_sources_and_tokens(client, monkeypatch):
    main.state["rag"] = _FakeRag()
    main.state["model"] = "fake-model"  # 不带 claude:/glm: 前缀，走本地 Ollama 适配器
    monkeypatch.setattr(
        main,
        "generate_ollama_prompt_stream",
        lambda prompt, model: iter(["顾长", "风。"]),
    )

    resp = client.post("/api/ask", json={"question": "雾隐山庄的庄主是谁", "top_k": 5})

    assert resp.status_code == 200
    body = resp.text
    # 每一步单独一个 step 事件——不是等检索全跑完再整包推
    assert body.count("event: step") == 3, "回答路径和两个检索阶段应逐条推送"
    assert body.index("回答路径") < body.index("理解问题")
    assert "识别到你在问《雾隐山庄》" in body
    assert '"ms": 2100' in body, "每步耗时要带上，界面才能显示慢在哪"
    # step 必须全部排在 sources 前面：检索没跑完就没有出处可发
    assert body.index("event: sources") > body.rindex("event: step")
    assert "event: sources" in body
    assert "顾长风是庄主" in body
    assert "第一章 风雪来客" in body
    assert 'data: "顾长"' in body
    assert 'data: "风。"' in body
    assert "event: done" in body


def test_ask_passes_session_history_into_answer_prompt(client, monkeypatch):
    """M3.6：追问时历史必须真的进最终回答的 prompt，并在 trace 里说明带了几轮。

    在此之前历史只喂给查询改写，回答 prompt 里一个字都没有——"再展开讲讲"
    这类追问因此必然失效。这里断言的是端到端的接线，不是 build_history_block
    本身的裁剪行为（那部分在 test_history_context.py）。
    """
    seen: dict = {}

    class _HistoryRag(_FakeRag):
        def build_prompt(self, question, sources, history=None, summary=None):
            seen["history"] = history
            return f"问题：{question}"

    main.state["rag"] = _HistoryRag()
    main.state["model"] = "fake-model"
    # 改写关掉：这条用例只验证"历史到没到 prompt"，不该被改写模型的可用性干扰
    monkeypatch.setattr(main, "QUERY_REWRITE_ENABLED", False)
    monkeypatch.setattr(
        main,
        "load_turns",
        lambda _sid: [
            {"role": "user", "content": "顾长风得了什么病？"},
            {"role": "assistant", "content": "中了蚀骨散[1]。"},
        ],
    )
    monkeypatch.setattr(
        main, "generate_ollama_prompt_stream", lambda prompt, model: iter(["好"])
    )

    resp = client.post("/api/ask", json={"question": "再展开讲讲", "session_id": "s-hist"})

    assert resp.status_code == 200
    assert seen["history"], "历史必须传到 build_prompt，否则追问依旧无效"
    assert seen["history"][0]["content"] == "顾长风得了什么病？"
    assert "对话背景" in resp.text
    assert "带入最近 2/2 轮对话" in resp.text


def test_ask_without_session_sends_no_history(client, monkeypatch):
    """没有 session_id 时行为必须和改造前完全一致：不读库、不加背景段。"""
    seen: dict = {}

    class _HistoryRag(_FakeRag):
        def build_prompt(self, question, sources, history=None, summary=None):
            seen["history"] = history
            return f"问题：{question}"

    main.state["rag"] = _HistoryRag()
    main.state["model"] = "fake-model"
    monkeypatch.setattr(
        main,
        "load_turns",
        lambda _sid: (_ for _ in ()).throw(AssertionError("没有会话就不该读历史")),
    )
    monkeypatch.setattr(
        main, "generate_ollama_prompt_stream", lambda prompt, model: iter(["好"])
    )

    resp = client.post("/api/ask", json={"question": "顾长风是谁"})

    assert resp.status_code == 200
    assert not seen["history"]
    assert "对话背景" not in resp.text


def test_library_question_answers_from_complete_catalog(client, monkeypatch):
    main.state["rag"] = _LibraryRag()
    main.state["model"] = "fake-model"
    monkeypatch.setattr(
        main,
        "generate_ollama_prompt_stream",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("目录事实不应调用模型猜测")
        ),
    )

    resp = client.post("/api/ask", json={"question": "现在一共有几部小说"})

    assert resp.status_code == 200
    assert "结构化查询" in resp.text
    assert "当前书架一共有 4 部小说" in resp.text
    assert "event: sources\ndata: []" in resp.text


def test_ask_without_index_returns_409(client):
    # state 里没有 "rag" key（索引未建立时 lifespan 也会是这个状态）
    resp = client.post("/api/ask", json={"question": "随便问问"})
    assert resp.status_code == 409


def test_free_mode_works_without_index(client, monkeypatch):
    """自由问答不依赖书架；这是模式拆分最重要的行为边界。"""
    main.state["model"] = "fake-model"
    received = {}

    def fake_generate(prompt, model):
        received["prompt"] = prompt
        received["model"] = model
        yield "RAG 是检索增强生成。"

    monkeypatch.setattr(main, "generate_ollama_prompt_stream", fake_generate)

    resp = client.post("/api/ask", json={"question": "什么是 RAG？", "mode": "free"})

    assert resp.status_code == 200
    assert "不搜索小说" in resp.text
    assert "RAG 是检索增强生成" in resp.text
    assert "event: sources\ndata: []" in resp.text
    assert "不检索用户的小说书架" in received["prompt"]


def test_auto_open_question_works_without_index(client, monkeypatch):
    main.state["model"] = "fake-model"
    monkeypatch.setattr(
        main, "generate_ollama_prompt_stream", lambda prompt, model: iter(["你好！"])
    )

    resp = client.post("/api/ask", json={"question": "你好", "mode": "auto"})

    assert resp.status_code == 200
    assert "识别为闲聊" in resp.text


def test_grounded_mode_still_requires_index(client):
    resp = client.post("/api/ask", json={"question": "什么是 RAG？", "mode": "grounded"})
    assert resp.status_code == 409


def test_auto_novel_question_still_requires_index(client):
    resp = client.post(
        "/api/ask", json={"question": "《凡人修仙传》的结局是什么？", "mode": "auto"}
    )
    assert resp.status_code == 409


def test_agent_lab_streams_steps_sources_and_tokens(client, monkeypatch):
    """Agent Lab 使用独立事件类型，不应污染普通 /api/ask 的协议。"""
    main.state["rag"] = object()
    main.state["model"] = "fake-model"
    source = type(
        "S",
        (),
        {
            "novel": "雾隐山庄",
            "chunk_id": 4,
            "chapter_title": "第二章",
            "text": "顾长风守住了山庄",
        },
    )()

    def fake_run_agent(*_args, **_kwargs):
        yield (
            "agent_step",
            {
                "step": 1,
                "reason": "先搜索",
                "tool": "search_novels",
                "args": {"query": "顾长风"},
                "observation": "找到 S1",
                "source_ids": ["S1"],
            },
        )
        yield "sources", [source]
        yield "token", "顾长风[1]"
        yield "done", {}

    monkeypatch.setattr(main, "run_agent", fake_run_agent)

    resp = client.post("/api/agent/ask", json={"question": "顾长风做了什么？", "max_steps": 3})

    assert resp.status_code == 200
    assert "event: agent_step" in resp.text
    assert '"tool": "search_novels"' in resp.text
    assert "event: sources" in resp.text
    assert "顾长风守住了山庄" in resp.text
    assert 'data: "顾长风[1]"' in resp.text
    assert "event: done" in resp.text


def test_agent_lab_validates_three_to_five_steps(client):
    main.state["rag"] = object()

    response = client.post("/api/agent/ask", json={"question": "测试", "max_steps": 2})

    assert response.status_code == 422


def test_agent_lab_saves_history_when_session_id_provided(client, monkeypatch):
    """带 session_id 时，Agent Lab 也要落库——这个端点上线时漏了这一步，
    刷新页面必然清空 Agent Lab 的对话，跟普通问答模式的历史恢复不一致。
    """
    main.state["rag"] = object()
    main.state["model"] = "fake-model"
    source = type(
        "S",
        (),
        {"novel": "雾隐山庄", "chunk_id": 4, "chapter_title": "第二章", "text": "守住了山庄"},
    )()

    def fake_run_agent(*_args, **_kwargs):
        yield (
            "agent_step",
            {
                "step": 1,
                "reason": "先搜索",
                "tool": "search_novels",
                "args": {"query": "顾长风"},
                "observation": "找到 S1",
                "source_ids": ["S1"],
            },
        )
        yield "sources", [source]
        yield "token", "顾长风[1]"
        yield "done", {}

    monkeypatch.setattr(main, "run_agent", fake_run_agent)

    saved: list[tuple] = []
    monkeypatch.setattr(main, "next_turn_index", lambda _session_id: 0)
    monkeypatch.setattr(
        main, "save_turn", lambda *args, **kwargs: saved.append((args, kwargs))
    )

    resp = client.post(
        "/api/agent/ask",
        json={"question": "顾长风做了什么？", "max_steps": 3, "session_id": "s1"},
    )

    assert resp.status_code == 200
    assert len(saved) == 2, "用户提问和最终回答都应该各存一次"

    user_args, _ = saved[0]
    assert user_args[:3] == ("s1", 0, "user")
    assert user_args[3] == "顾长风做了什么？"

    assistant_args, assistant_kwargs = saved[1]
    assert assistant_args[:3] == ("s1", 1, "assistant")
    assert assistant_args[3] == "顾长风[1]"
    assert assistant_kwargs["agent_steps"][0]["tool"] == "search_novels"
    assert assistant_kwargs["status"] == "complete"


def test_agent_lab_skips_history_without_session_id(client, monkeypatch):
    """不带 session_id 时行为不变：纯内存，不调用任何落库函数。"""
    main.state["rag"] = object()
    main.state["model"] = "fake-model"

    def fake_run_agent(*_args, **_kwargs):
        yield "token", "答案"
        yield "done", {}

    monkeypatch.setattr(main, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        main,
        "save_turn",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("不该调用 save_turn")),
    )

    resp = client.post("/api/agent/ask", json={"question": "随便问问", "max_steps": 3})

    assert resp.status_code == 200
    assert 'data: "答案"' in resp.text


# ---------------------------------------------------------------- 按需核实引用


def test_verify_citation_only_judges_sentences_that_cite_that_source(client, monkeypatch):
    """只把"真正引用了这条出处的句子"送给 Judge。

    把整段回答一起塞过去，Judge 多半会因为无关句子判 unsupported——那不是引用
    错了，是我们问错了问题。
    """
    main.state["model"] = "fake-model"
    seen = {}

    def fake_judge(statement, evidence, _generate_fn, *args, **kwargs):
        seen["statement"] = statement
        seen["evidence"] = list(evidence)
        return {"label": "supported", "reason": "证据直接支持"}

    monkeypatch.setattr(main, "judge_support", fake_judge)

    resp = client.post(
        "/api/citations/verify",
        json={
            "answer": "顾长风中了蚀骨散[1]。沈砚之是江南来的郎中[2]。",
            "citation": 1,
            "evidence": ["顾长风所患并非寻常旧疾，而是蚀骨散之毒"],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "supported"
    assert body["model"] == "fake-model", "必须回传判定用的模型——准确率是模型相关的"
    # 只核实引用了 [1] 的那句，不含 [2] 那句
    assert "蚀骨散[1]" in seen["statement"]
    assert "沈砚之" not in seen["statement"]


def test_verify_citation_rejects_a_number_the_answer_never_cites(client, monkeypatch):
    """回答里没有 [n] 时直接报错，而不是把空字符串送去判定。"""
    main.state["model"] = "fake-model"
    monkeypatch.setattr(
        main,
        "judge_support",
        lambda *a, **k: pytest.fail("不该在没有对应引用时调用 Judge"),
    )

    resp = client.post(
        "/api/citations/verify",
        json={"answer": "顾长风中了蚀骨散[1]。", "citation": 3, "evidence": ["原文"]},
    )

    assert resp.status_code == 400


def test_verify_citation_passes_through_uncertain_verdict(client, monkeypatch):
    """Judge 说不准时如实返回 uncertain，不擅自转成"有据"或"无据"。"""
    main.state["model"] = "glm:glm-4-flash"
    monkeypatch.setattr(
        main,
        "judge_support",
        lambda *a, **k: {"label": "uncertain", "reason": "Judge 调用失败：超时"},
    )

    body = client.post(
        "/api/citations/verify",
        json={"answer": "顾长风中了蚀骨散[1]。", "citation": 1, "evidence": ["原文"]},
    ).json()

    assert body["label"] == "uncertain"
    assert "超时" in body["reason"]
