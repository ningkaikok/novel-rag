"""V1 → 通用知识库目录 API 的最小契约测试。"""

import backend.main as main


def test_knowledge_documents_are_metadata_only_and_use_v1_manifest(
    client, tmp_path, monkeypatch
):
    (tmp_path / "雾隐山庄.txt").write_text("不应出现在目录响应里的正文")
    (tmp_path / "说明.md").write_text("非 TXT 输入不进入旧小说目录")
    monkeypatch.setattr(main, "NOVELS_DIR", tmp_path)
    monkeypatch.setattr(
        main,
        "load_index_manifest",
        lambda: {
            "雾隐山庄.txt": {
                "source_hash": "source-1",
                "pipeline_hash": "pipeline-1",
                "chunk_count": 7,
            }
        },
    )

    response = client.get("/api/knowledge/documents")

    assert response.status_code == 200
    body = response.json()
    assert len(body["documents"]) == 1
    document = body["documents"][0]
    assert document["title"] == "雾隐山庄"
    assert document["source_type"] == "novel"
    assert document["status"] == "indexed"
    assert document["metadata"] == {
        "legacy_novel": "雾隐山庄.txt",
        "storage_schema": "v1",
    }
    assert "text" not in document
    assert document["versions"][0]["source_hash"] == "source-1"
    assert document["versions"][0]["chunk_count"] == 7


def test_knowledge_catalog_degrades_to_source_only_without_database(
    client, tmp_path, monkeypatch
):
    (tmp_path / "孤本.txt").write_text("本地文件")
    monkeypatch.setattr(main, "NOVELS_DIR", tmp_path)

    def unavailable_database():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(main, "load_index_manifest", unavailable_database)

    response = client.get("/api/knowledge/documents")

    assert response.status_code == 200
    document = response.json()["documents"][0]
    assert document["status"] == "source_only"
    assert document["versions"][0]["source_hash"] is None
    assert document["versions"][0]["chunk_count"] is None


def test_knowledge_collections_are_stable_one_document_summaries(
    client, tmp_path, monkeypatch
):
    (tmp_path / "甲.txt").write_text("甲")
    (tmp_path / "乙.txt").write_text("乙")
    monkeypatch.setattr(main, "NOVELS_DIR", tmp_path)
    monkeypatch.setattr(main, "load_index_manifest", lambda: {})

    response = client.get("/api/knowledge/collections")

    assert response.status_code == 200
    assert sorted(
        (item["name"], item["document_count"])
        for item in response.json()["collections"]
    ) == [("乙", 1), ("甲", 1)]
