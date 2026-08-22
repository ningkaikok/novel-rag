"""M4 关系审核端点测试：/api/graph/edges 与 /api/graph/review。

走 TestClient，monkeypatch 掉 postgres 的读写函数，不连真实数据库。
"""

import backend.main as main


def _edge(**overrides):
    edge = {
        "novel": "青梧镇异闻",
        "person_a": "小顺",
        "person_b": "沈砚秋",
        "relation": "师徒",
        "weight": 3,
        "direction": "沈砚秋→小顺",
        "confidence": 0.85,
        "evidence_type": "explicit",
        "source_chunk_ids": [4],
        "review_status": "pending",
        "evidence_excerpt": "想学手艺可以，明天来铺子里扫地…",
    }
    edge.update(overrides)
    return edge


def test_edges_endpoint_returns_pending_queue(client, monkeypatch):
    captured = {}

    def fake_list(status, limit, offset):
        captured.update(status=status, limit=limit, offset=offset)
        return [_edge()], 27

    monkeypatch.setattr(main, "list_relation_edges", fake_list)

    resp = client.get("/api/graph/edges?limit=20&offset=5")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 27 and len(body["edges"]) == 1
    assert body["edges"][0]["evidence_type"] == "explicit"
    assert captured == {"status": "pending", "limit": 20, "offset": 5}, "默认只看待审核队列"


def test_edges_endpoint_supports_all_and_rejected_filters(client, monkeypatch):
    captured = {}

    def fake_list(status, limit, offset):
        captured.update(status=status)
        return [], 0

    monkeypatch.setattr(main, "list_relation_edges", fake_list)

    assert client.get("/api/graph/edges?status=all").status_code == 200
    assert captured["status"] is None
    assert client.get("/api/graph/edges?status=rejected").status_code == 200
    assert captured["status"] == "rejected"


def test_edges_endpoint_rejects_unknown_status(client):
    resp = client.get("/api/graph/edges?status=bogus")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "validation_error"


def test_review_endpoint_writes_decision(client, monkeypatch):
    captured = {}

    def fake_review(novel, a, b, relation, status):
        captured.update(
            novel=novel,
            person_a=a,
            person_b=b,
            relation=relation,
            status=status,
        )
        return 1

    monkeypatch.setattr(main, "set_relation_review", fake_review)

    payload = {
        "novel": "青梧镇异闻",
        "person_a": "小顺",
        "person_b": "沈砚秋",
        "relation": "师徒",
        "status": "rejected",
    }
    resp = client.post("/api/graph/review", json=payload)

    assert resp.status_code == 200
    assert resp.json() == {"review_status": "rejected"}
    assert captured == payload, "四元组主键原样传给存储层"


def test_review_endpoint_validates_action(client, monkeypatch):
    """只允许 approved/rejected；pending 不是人工动作。"""
    monkeypatch.setattr(main, "set_relation_review", lambda *a: 1)

    base = {"novel": "书", "person_a": "甲", "person_b": "乙", "relation": "同伴"}
    for bad in ("maybe", "", "pending"):
        resp = client.post("/api/graph/review", json={**base, "status": bad})
        assert resp.status_code == 400


def test_review_endpoint_404_when_edge_missing(client, monkeypatch):
    monkeypatch.setattr(main, "set_relation_review", lambda *a: 0)

    resp = client.post(
        "/api/graph/review",
        json={
            "novel": "书",
            "person_a": "甲",
            "person_b": "乙",
            "relation": "同伴",
            "status": "approved",
        },
    )
    assert resp.status_code == 404
