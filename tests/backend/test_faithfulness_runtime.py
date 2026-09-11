import backend.main as main


def test_shadow_judgment_persists_only_aggregates(monkeypatch):
    saved = []
    monkeypatch.setattr(main, "FAITHFULNESS_JUDGE_MODE", "two_step")
    monkeypatch.setattr(
        main,
        "judge_support_two_step",
        lambda *args, **kwargs: {
            "label": "partial",
            "claims": ["断言A", "断言B"],
            "verdicts": [
                {"label": "supported"},
                {"label": "not_found"},
            ],
        },
    )
    monkeypatch.setattr(main, "save_citation_judgment", lambda **kwargs: saved.append(kwargs))

    main._persist_faithfulness_shadow(
        answer="顾长风中了蚀骨散[1]。",
        sources=[
            {
                "novel": "雾隐山庄",
                "chunk_id": 3,
                "text": "顾长风所患的是蚀骨散之毒",
            }
        ],
        model="glm:glm-4-flash",
        session_id="s-shadow",
        turn_index=1,
    )

    assert len(saved) == 1
    assert saved[0]["label"] == "partial"
    assert saved[0]["method"] == "shadow_two_step"
    assert saved[0]["claim_count"] == 2
    assert saved[0]["supported_count"] == 1
    assert saved[0]["not_found_count"] == 1
    assert "claims" not in saved[0]
    assert "text" not in saved[0]


def test_shadow_runner_ignores_uncited_sources(monkeypatch):
    monkeypatch.setattr(
        main,
        "judge_support_two_step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("不应调用 Judge")),
    )
    monkeypatch.setattr(
        main,
        "save_citation_judgment",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("不应落库")),
    )

    main._persist_faithfulness_shadow(
        answer="没有引用。",
        sources=[{"novel": "雾隐山庄", "chunk_id": 0, "text": "证据"}],
        model="glm:glm-4-flash",
        session_id=None,
        turn_index=None,
    )
