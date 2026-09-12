from chunk_model import SourceChunk
from domain_models import Document, DocumentChunk, DocumentVersion, RetrievalScope
from legacy_novel import LegacyNovelAdapter
from response_adapters import (
    source_ref_from_document_chunk,
    source_ref_from_legacy_chunk,
    source_ref_payload,
)
from retrieval_scope import select_legacy_novels


def test_scope_selector_matches_collection_document_and_known_version_only():
    novels = ["甲.txt", "乙.txt"]
    source_hashes = {"甲.txt": "hash-a", "乙.txt": "hash-b"}

    assert select_legacy_novels(None, novels) is None
    assert select_legacy_novels(RetrievalScope(document_id="bad"), novels) == []
    assert (
        select_legacy_novels(
            RetrievalScope(
                collection_id="legacy:collection:not-this",
                document_id="legacy:document:not-this",
            ),
            novels,
        )
        == []
    )

    document_scope = LegacyNovelAdapter.scope("乙.txt")
    assert select_legacy_novels(document_scope, novels) == ["乙.txt"]
    version_scope = LegacyNovelAdapter.scope("乙.txt", source_hash="hash-b")
    assert select_legacy_novels(version_scope, novels, source_hashes=source_hashes) == [
        "乙.txt"
    ]
    assert select_legacy_novels(version_scope, novels) == []


def test_document_source_ref_has_identity_locator_and_excerpt_limit():
    document = Document(
        id="doc-1",
        collection_id="collection-1",
        title="研究资料.md",
        source_type="markdown",
    )
    version = DocumentVersion(
        id="version-1",
        document_id=document.id,
        version_no=1,
        parser_name="markdown",
        parser_version="1.0",
        source_hash="hash-1",
    )
    chunk = DocumentChunk(
        id="chunk-1",
        document_version_id=version.id,
        ordinal=0,
        text="正文" * 100,
        section_path=("总览", "设计"),
    )

    ref = source_ref_from_document_chunk(chunk, document)
    payload = source_ref_payload(ref)

    assert ref.locator.kind == "heading"
    assert ref.locator.value == "总览/设计"
    assert payload["document_id"] == document.id
    assert payload["version_id"] == version.id
    assert payload["chunk_id"] == chunk.id
    assert len(payload["excerpt"]) == 80
    assert "text" not in payload


def test_legacy_source_ref_adapter_preserves_old_fields_and_limits_excerpt():
    source = SourceChunk(
        novel="演示.txt",
        chunk_id=3,
        text="原文" * 100,
        distance=0.1,
        chapter_title="第三章",
    )

    ref = source_ref_from_legacy_chunk(source, source_hash="hash-1")
    payload = source_ref_payload(ref)

    assert ref.document_title == source.novel
    assert ref.version_id == LegacyNovelAdapter.version_id(source.novel, source_hash="hash-1")
    assert ref.locator.kind == "chunk"
    assert ref.locator.value == "3"
    assert ref.locator.label == "第三章"
    assert len(payload["excerpt"]) == 80
