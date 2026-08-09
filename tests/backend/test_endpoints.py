"""关键端点的响应形状/基本行为测试。

全部走 TestClient，不依赖真实数据库、Ollama 或云端 key——按需要 monkeypatch
掉会碰真实资源的函数（load_embedder、connect、_list_ollama_models 等）。
主要验证这一批 response_model 改造之后，真实返回值仍然符合声明的形状。
"""
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


class _FakeRag:
    """/api/ask 依赖的最小 NovelRAG 替身：只实现路由会调用到的几个方法。"""

    def retrieve_hybrid_stream(self, question, top_k):
        """真实实现是个生成器：每完成一步 yield 一条 step，最后 yield result。"""
        yield "step", {"step": "理解问题", "detail": "识别到你在问《雾隐山庄》", "ms": 12}
        yield "step", {"step": "精排", "detail": "取最相关的 3 段", "ms": 2100}
        yield "result", [
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
        ]

    def expand_neighbors(self, sources):
        return sources

    def build_prompt(self, question, sources):
        return f"问题：{question}"

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

    resp = client.post(
        "/api/ask", json={"question": "什么是 RAG？", "mode": "free"}
    )

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
    resp = client.post(
        "/api/ask", json={"question": "什么是 RAG？", "mode": "grounded"}
    )
    assert resp.status_code == 409


def test_auto_novel_question_still_requires_index(client):
    resp = client.post(
        "/api/ask", json={"question": "《凡人修仙传》的结局是什么？", "mode": "auto"}
    )
    assert resp.status_code == 409
