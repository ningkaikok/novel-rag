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


def test_unchanged_book_without_hierarchy_manifest_gets_summary_backfill(
    tmp_path, monkeypatch
):
    path = tmp_path / "旧书.txt"
    path.write_text("第一章 开始\n旧索引内容没有变化", encoding="utf-8")
    monkeypatch.setattr(ingest, "index_pipeline_hash", lambda: "pipeline-v1")
    monkeypatch.setattr(ingest, "hierarchy_pipeline_hash", lambda: "hierarchy-v1")
    source_hash = ingest._file_hash(path)

    plan = ingest.plan_index(
        tmp_path,
        manifest={"旧书": _manifest(source_hash, "pipeline-v1")},
        database_novels={"旧书"},
        hierarchy_manifest={},
    )

    assert plan.modified == []
    assert plan.unchanged == ["旧书"]
    assert plan.hierarchy_pending == ["旧书"]
    assert plan.hierarchy_hash == "hierarchy-v1"


class _FakeEmbedder:
    class _Tokenizer:
        name_or_path = "test-tokenizer"
        model_max_length = 512

        def __call__(self, text, **_kwargs):
            return {"input_ids": list(range(len(text)))}

    tokenizer = _Tokenizer()
    max_seq_length = 512

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
    monkeypatch.setattr(ingest, "hierarchy_node_count", lambda: 2)
    monkeypatch.setattr(
        ingest,
        "replace_novel_index",
        lambda novel, rows, terms, source_hash, pipeline_hash, relations, check, hierarchy_rows, hierarchy_hash, **kwargs: (
            replaced.append(
                (
                    novel,
                    rows,
                    terms,
                    source_hash,
                    pipeline_hash,
                    hierarchy_rows,
                    hierarchy_hash,
                )
            )
        ),
    )

    result = ingest.build_index(
        model=_FakeEmbedder(),
        novels_dir=tmp_path,
        progress=lambda stage, percent, message: stages.append((stage, percent)),
    )

    assert [item[0] for item in replaced] == ["变化"]
    assert replaced[0][3:5] == ("new-hash", "pipeline-v2")
    assert len(replaced[0][5]) == 2  # 一个章节节点 + 一个全书节点
    assert replaced[0][6] == ""
    assert result["modified"] == ["变化"]
    assert result["unchanged"] == ["未变化"]
    assert result["chunk_count"] == 9
    assert result["hierarchy_nodes"] == 2
    assert {stage for stage, _ in stages} >= {
        "scan",
        "split",
        "embedding",
        "bm25",
        "database",
        "complete",
    }
    assert stages[-1] == ("complete", 100)


def test_build_index_cancelled_before_write_keeps_database_untouched(tmp_path, monkeypatch):
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
    monkeypatch.setattr(
        ingest, "replace_novel_index", lambda *args, **kwargs: writes.append(args)
    )

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


def test_build_index_backfills_only_hierarchy_for_unchanged_book(tmp_path, monkeypatch):
    path = tmp_path / "旧书.txt"
    path.write_text("第一章 开始\n正文内容", encoding="utf-8")
    plan = ingest.IndexPlan(
        paths={"旧书": path},
        source_hashes={"旧书": "same-hash"},
        added=[],
        modified=[],
        deleted=[],
        unchanged=["旧书"],
        pipeline_hash="pipeline-v1",
        hierarchy_pending=["旧书"],
        hierarchy_hash="hierarchy-v1",
    )
    base_writes = []
    hierarchy_writes = []
    monkeypatch.setattr(ingest, "plan_index", lambda *args, **kwargs: plan)
    monkeypatch.setattr(ingest, "ensure_index_schema", lambda dimension: None)
    monkeypatch.setattr(ingest, "index_chunk_count", lambda: 1)
    monkeypatch.setattr(ingest, "hierarchy_node_count", lambda: 2)
    monkeypatch.setattr(ingest, "replace_novel_index", lambda *args: base_writes.append(args))
    monkeypatch.setattr(
        ingest, "replace_novel_hierarchy", lambda *args: hierarchy_writes.append(args)
    )

    result = ingest.build_index(model=_FakeEmbedder(), novels_dir=tmp_path)

    assert base_writes == []
    assert len(hierarchy_writes) == 1
    novel, rows, source_hash, hierarchy_hash, _check = hierarchy_writes[0]
    assert (novel, source_hash, hierarchy_hash) == (
        "旧书",
        "same-hash",
        "hierarchy-v1",
    )
    assert len(rows) == 2
    assert result["hierarchy_nodes"] == 2
