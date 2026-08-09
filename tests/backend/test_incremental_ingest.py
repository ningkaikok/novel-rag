from pathlib import Path

import ingest


def _manifest(source_hash: str, pipeline_hash: str) -> dict:
    return {
        "source_hash": source_hash,
        "pipeline_hash": pipeline_hash,
        "chunk_count": 1,
    }


def test_plan_index_detects_added_modified_deleted_and_unchanged(tmp_path, monkeypatch):
    (tmp_path / "不变.txt").write_text("相同内容", encoding="utf-8")
    (tmp_path / "修改.txt").write_text("新内容", encoding="utf-8")
    (tmp_path / "新增.txt").write_text("第一次出现", encoding="utf-8")
    monkeypatch.setattr(ingest, "index_pipeline_hash", lambda: "pipeline-v1")

    unchanged_hash = ingest._file_hash(tmp_path / "不变.txt")
    plan = ingest.plan_index(
        tmp_path,
        manifest={
            "不变": _manifest(unchanged_hash, "pipeline-v1"),
            "修改": _manifest("old-hash", "pipeline-v1"),
            "已删除": _manifest("gone", "pipeline-v1"),
        },
        database_novels={"不变", "修改", "已删除"},
    )

    assert plan.added == ["新增"]
    assert plan.modified == ["修改"]
    assert plan.deleted == ["已删除"]
    assert plan.unchanged == ["不变"]


def test_plan_index_force_rebuilds_existing_books(tmp_path, monkeypatch):
    path = tmp_path / "小说.txt"
    path.write_text("没有变化", encoding="utf-8")
    monkeypatch.setattr(ingest, "index_pipeline_hash", lambda: "pipeline-v1")
    manifest = {"小说": _manifest(ingest._file_hash(path), "pipeline-v1")}

    plan = ingest.plan_index(
        tmp_path,
        force=True,
        manifest=manifest,
        database_novels={"小说"},
    )

    assert plan.modified == ["小说"]
    assert plan.unchanged == []


def test_old_index_without_manifest_gets_one_time_migration(tmp_path, monkeypatch):
    (tmp_path / "旧书.txt").write_text("旧索引仍有片段，但没有清单", encoding="utf-8")
    monkeypatch.setattr(ingest, "index_pipeline_hash", lambda: "pipeline-v1")

    plan = ingest.plan_index(
        tmp_path,
        manifest={},
        database_novels={"旧书"},
    )

    assert plan.modified == ["旧书"]


class _FakeEmbedder:
    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, **kwargs):
        return [[0.1, 0.2] for _ in texts]


def test_build_index_only_prepares_changed_book(tmp_path, monkeypatch):
    changed_path = tmp_path / "变化.txt"
    changed_path.write_text("第一章 开始\n这是发生变化的内容。", encoding="utf-8")
    plan = ingest.IndexPlan(
        paths={"变化": changed_path},
        source_hashes={"变化": "new-hash"},
        added=[],
        modified=["变化"],
        deleted=[],
        unchanged=["未变化"],
        pipeline_hash="pipeline-v2",
    )
    replaced = []
    stages = []
    monkeypatch.setattr(ingest, "plan_index", lambda *args, **kwargs: plan)
    monkeypatch.setattr(ingest, "ensure_index_schema", lambda dimension: None)
    monkeypatch.setattr(ingest, "index_chunk_count", lambda: 9)
    monkeypatch.setattr(
        ingest,
        "replace_novel_index",
        lambda novel, rows, terms, source_hash, pipeline_hash, relations, check: replaced.append(
            (novel, rows, terms, source_hash, pipeline_hash)
        ),
    )

    result = ingest.build_index(
        model=_FakeEmbedder(),
        novels_dir=tmp_path,
        progress=lambda stage, percent, message: stages.append((stage, percent)),
    )

    assert [item[0] for item in replaced] == ["变化"]
    assert replaced[0][3:] == ("new-hash", "pipeline-v2")
    assert result["modified"] == ["变化"]
    assert result["unchanged"] == ["未变化"]
    assert result["chunk_count"] == 9
    assert {stage for stage, _ in stages} >= {
        "scan",
        "split",
        "embedding",
        "bm25",
        "database",
        "complete",
    }
    assert stages[-1] == ("complete", 100)


def test_build_index_cancelled_before_write_keeps_database_untouched(
    tmp_path, monkeypatch
):
    path = tmp_path / "取消.txt"
    path.write_text("正文内容", encoding="utf-8")
    plan = ingest.IndexPlan(
        paths={"取消": path},
        source_hashes={"取消": "hash"},
        added=["取消"],
        modified=[],
        deleted=[],
        unchanged=[],
        pipeline_hash="pipeline",
    )
    writes = []
    checks = 0

    def cancel_during_preparation():
        nonlocal checks
        checks += 1
        if checks >= 3:
            raise ingest.IndexCancelled()

    monkeypatch.setattr(ingest, "plan_index", lambda *args, **kwargs: plan)
    monkeypatch.setattr(ingest, "ensure_index_schema", lambda dimension: None)
    monkeypatch.setattr(ingest, "replace_novel_index", lambda *args: writes.append(args))

    try:
        ingest.build_index(
            model=_FakeEmbedder(),
            novels_dir=tmp_path,
            cancel_check=cancel_during_preparation,
        )
    except ingest.IndexCancelled:
        pass
    else:
        raise AssertionError("expected cancellation")

    assert writes == []
