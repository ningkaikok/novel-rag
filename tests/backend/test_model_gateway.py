from backend import model_gateway


def test_codex_prefix_routes_to_codex_provider():
    assert model_gateway.provider_name("codex:gpt-5-codex") == "codex"


def test_codex_factory_is_used_for_codex_prefixed_model():
    result = list(
        model_gateway.generate_stream(
            "问题",
            "codex:gpt-5-codex",
            codex_factory=lambda model, prompt: iter([f"{model}:{prompt}"]),
        )
    )
    assert result == ["codex:gpt-5-codex:问题"]


def test_answer_task_respects_requested_model(monkeypatch):
    monkeypatch.setattr(model_gateway, "MODEL_ROUTING_ENABLED", True)
    assert model_gateway.resolve_model("answer", "qwen2.5:7b") == "qwen2.5:7b"


def test_internal_task_can_use_dedicated_model(monkeypatch):
    monkeypatch.setattr(model_gateway, "MODEL_ROUTING_ENABLED", True)
    monkeypatch.setitem(model_gateway.TASK_MODELS, "query_rewrite", "glm:rewrite")
    assert model_gateway.resolve_model("query_rewrite", "claude:sonnet") == "glm:rewrite"


def test_generation_records_metadata_without_prompt_or_answer(monkeypatch):
    stats = []

    def fake_ollama(model, prompt):
        assert model == "qwen2.5:7b"
        assert prompt == "秘密正文"
        return iter(["你好", "世界"])

    result = list(
        model_gateway.generate_stream(
            "秘密正文", "qwen2.5:7b", stats=stats, ollama_factory=fake_ollama
        )
    )

    assert result == ["你好", "世界"]
    snapshot = stats[0].snapshot()
    assert snapshot["completed"] is True
    assert snapshot["chunks"] == 2
    assert snapshot["characters"] == 4
    assert snapshot["estimated_input_tokens"] > 0
    assert snapshot["estimated_output_tokens"] > 0
    assert "秘密正文" not in snapshot
    assert "你好世界" not in snapshot


def test_fallback_only_happens_before_first_chunk(monkeypatch):
    monkeypatch.setattr(model_gateway, "MODEL_FALLBACK_MODEL", "qwen2.5:3b")
    stats = []
    calls = []

    def fake_ollama(model, prompt):
        calls.append(model)
        if model == "qwen2.5:7b":
            raise RuntimeError("primary unavailable")
        return iter(["降级回答"])

    result = list(
        model_gateway.generate_stream(
            "问题", "qwen2.5:7b", stats=stats, ollama_factory=fake_ollama
        )
    )

    assert result == ["降级回答"]
    assert calls == ["qwen2.5:7b", "qwen2.5:3b"]
    assert stats[0].fallback_used is True
    assert stats[0].selected_model == "qwen2.5:3b"


def test_partial_generation_error_is_not_silently_retried(monkeypatch):
    monkeypatch.setattr(model_gateway, "MODEL_FALLBACK_MODEL", "qwen2.5:3b")

    def broken_ollama(model, prompt):
        yield "半截"
        raise RuntimeError("stream broken")

    try:
        list(model_gateway.generate_stream("问题", "qwen2.5:7b", ollama_factory=broken_ollama))
    except RuntimeError as exc:
        assert str(exc) == "stream broken"
    else:  # pragma: no cover - 防止测试被错误吞掉
        raise AssertionError("部分输出后的错误不能静默切换模型")


def test_cloud_permission_and_output_budget_are_enforced(monkeypatch):
    monkeypatch.setattr(model_gateway, "MODEL_CLOUD_ALLOWED", False)
    monkeypatch.setattr(model_gateway, "MODEL_FALLBACK_MODEL", "")
    try:
        list(
            model_gateway.generate_stream(
                "问题", "glm:flash", zhipu_factory=lambda _model, _prompt: iter(["答案"])
            )
        )
    except RuntimeError as exc:
        assert "MODEL_CLOUD_ALLOWED" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("云端权限关闭时不应调用模型")

    monkeypatch.setattr(model_gateway, "MODEL_CLOUD_ALLOWED", True)
    monkeypatch.setattr(model_gateway, "MODEL_MAX_OUTPUT_CHARACTERS", 2)
    try:
        list(
            model_gateway.generate_stream(
                "问题", "qwen2.5:7b", ollama_factory=lambda _model, _prompt: iter(["超过预算"])
            )
        )
    except RuntimeError as exc:
        assert "输出字符预算" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("输出预算未生效")
