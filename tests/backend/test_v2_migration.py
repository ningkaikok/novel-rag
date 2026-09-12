import pytest

from v2_migration import (
    LegacyNovelSnapshot,
    MigrationPlanError,
    build_v2_migration_plan,
    validate_v2_plan,
)


def _snapshot(novel="乙.txt", source_hash="source-2"):
    return LegacyNovelSnapshot(
        novel=novel,
        source_hash=source_hash,
        pipeline_hash="pipeline-1",
        chunks=(
            {
                "novel": novel,
                "chunk_id": 1,
                "chapter_title": "第二章",
                "text": "第二段",
                "context": "",
            },
            {
                "novel": novel,
                "chunk_id": 0,
                "chapter_title": "第一章",
                "text": "第一段",
                "context": "",
            },
        ),
        terms={0: {"第一": 2}, 1: {"第二": 1}},
    )


def test_legacy_snapshot_plan_is_deterministic_and_sorted():
    first = build_v2_migration_plan([_snapshot("乙.txt"), _snapshot("甲.txt", "source-1")])
    second = build_v2_migration_plan([_snapshot("甲.txt", "source-1"), _snapshot("乙.txt")])

    assert first == second
    assert [item.document.title for item in first.documents] == ["乙.txt", "甲.txt"]
    assert [chunk.ordinal for chunk in first.documents[0].chunks] == [0, 1]
    assert first.chunk_count == 4
    assert first.term_count == 4
    assert first.idempotency_fingerprint() == second.idempotency_fingerprint()
    assert validate_v2_plan(first).valid


def test_migration_plan_preserves_hash_and_locator_chain():
    plan = build_v2_migration_plan([_snapshot()])
    item = plan.documents[0]

    assert item.version.source_hash == "source-2"
    assert item.manifest.source_hash == item.version.source_hash
    assert item.manifest.chunk_count == len(item.chunks)
    assert item.chunks[0].document_version_id == item.version.id
    assert item.chunks[0].section_path == ("第一章",)
    assert item.terms[0].chunk_id == item.chunks[0].id


def test_migration_rejects_duplicates_and_orphan_terms():
    with pytest.raises(MigrationPlanError, match="重复小说"):
        build_v2_migration_plan([_snapshot("同一本"), _snapshot("同一本")])

    orphaned = _snapshot()
    orphaned = LegacyNovelSnapshot(
        novel=orphaned.novel,
        source_hash=orphaned.source_hash,
        pipeline_hash=orphaned.pipeline_hash,
        chunks=orphaned.chunks,
        terms={99: {"不存在": 1}},
    )
    with pytest.raises(MigrationPlanError, match="不存在的 chunk"):
        build_v2_migration_plan([orphaned])
