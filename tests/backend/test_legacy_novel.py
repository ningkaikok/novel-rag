from chunk_model import SourceChunk
from legacy_novel import LegacyNovelAdapter


def test_legacy_novel_mapping_is_deterministic_and_keeps_novel_metadata_localized():
    novel = "示例小说.txt"
    first = LegacyNovelAdapter.document(novel)
    second = LegacyNovelAdapter.document(novel)
    chunk = SourceChunk(
        novel=novel,
        chunk_id=7,
        text="原文片段，作为运行时测试数据。" * 8,
        distance=0.2,
        chapter_title="第一章 初遇",
        context="上一层检索上下文",
    )

    domain_chunk = LegacyNovelAdapter.chunk(chunk)
    source_ref = LegacyNovelAdapter.source_ref(chunk)

    assert first == second
    assert domain_chunk.ordinal == 7
    assert domain_chunk.section_path == ("第一章 初遇",)
    assert domain_chunk.context == chunk.context
    assert source_ref.document_id == first.id
    assert source_ref.version_id == domain_chunk.document_version_id
    assert source_ref.locator.value == "7"
    assert source_ref.locator.label == "第一章 初遇"
    assert len(source_ref.excerpt) <= 80


def test_legacy_scope_round_trip_is_safe_for_unknown_generic_scope():
    novels = ["甲.txt", "乙.txt"]
    scope = LegacyNovelAdapter.scope("乙.txt")

    assert LegacyNovelAdapter.novel_for_scope(scope, novels) == ["乙.txt"]
    assert (
        LegacyNovelAdapter.novel_for_scope(
            scope.model_copy(update={"document_id": "v2:unknown"}), novels
        )
        == []
    )


def test_source_chunk_exposes_domain_views_without_changing_legacy_fields():
    chunk = SourceChunk("雾隐山庄", 3, "证据", 0.0, "第三章")

    assert chunk.novel == "雾隐山庄"
    assert chunk.to_document_chunk().ordinal == 3
    assert chunk.to_source_ref().locator.value == "3"
    restored = SourceChunk.from_legacy_row(
        {
            "novel": "雾隐山庄",
            "chunk_id": 3,
            "chapter_title": "第三章",
            "text": "证据",
            "context": "",
        },
        distance=0.4,
    )
    assert restored == chunk.__class__("雾隐山庄", 3, "证据", 0.4, "第三章")
