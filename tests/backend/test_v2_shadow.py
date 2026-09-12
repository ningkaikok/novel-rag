from domain_models import DocumentChunk
from legacy_novel import LegacyNovelAdapter
from v2_shadow import (
    compare_shadow,
    snapshot_from_v1_rows,
    snapshot_from_v2_document,
)


def _v2(novel: str, *, source_hash: str = "source-1", page_number: int | None = None):
    document = LegacyNovelAdapter.document(novel)
    version = LegacyNovelAdapter.version(novel, source_hash=source_hash)
    chunks = (
        DocumentChunk(
            id="chunk-0",
            document_version_id=version.id,
            ordinal=0,
            text="短文本",
            section_path=(),
            page_number=page_number,
        ),
    )
    return document, version, chunks


def test_shadow_match_uses_stable_document_chunk_and_candidate_keys():
    novel = "演示.txt"
    rows = ({"chunk_id": 0, "chapter_title": "第一章", "text": "短文本"},)
    v1 = snapshot_from_v1_rows(novel, rows, source_hash="source-1", candidate_ordinals=(0,))
    v2 = snapshot_from_v2_document(*_v2(novel), candidate_ordinals=(0,))

    comparison = compare_shadow(v1, v2)

    assert comparison.matches
    assert comparison.categories == ()


def test_shadow_classifies_locator_and_candidate_mismatches():
    novel = "演示.txt"
    rows = ({"chunk_id": 0, "chapter_title": "第一章", "text": "短文本"},)
    v1 = snapshot_from_v1_rows(novel, rows, source_hash="source-1", candidate_ordinals=(0,))
    v2 = snapshot_from_v2_document(*_v2(novel, page_number=2), candidate_ordinals=())

    comparison = compare_shadow(v1, v2)

    assert not comparison.matches
    assert "source_locator" in comparison.categories
    assert "candidate_missing" in comparison.categories


def test_shadow_classifies_document_hash_and_chunk_count_mismatches():
    novel = "演示.txt"
    rows = ({"chunk_id": 0, "chapter_title": "第一章", "text": "短文本"},)
    v1 = snapshot_from_v1_rows(novel, rows, source_hash="source-1")
    v2 = snapshot_from_v2_document(*_v2("另一本.txt", source_hash="source-2"))

    comparison = compare_shadow(v1, v2)
    categories = set(comparison.categories)

    assert {"document_identity", "version_identity", "source_hash"} <= categories
