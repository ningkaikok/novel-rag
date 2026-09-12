import pytest
from pydantic import ValidationError

from domain_models import (
    Collection,
    Document,
    DocumentChunk,
    DocumentVersion,
    RetrievalScope,
    SourceLocator,
    SourceRef,
)


def test_domain_models_have_versioned_schema_and_stable_locator():
    collection = Collection(id="c-1", name="研究资料")
    document = Document(
        id="d-1",
        collection_id=collection.id,
        title="设计文档",
        source_type="markdown",
    )
    version = DocumentVersion(
        id="v-1",
        document_id=document.id,
        version_no=1,
        parser_name="markdown",
        parser_version="1",
    )
    chunk = DocumentChunk(
        id="ch-1",
        document_version_id=version.id,
        ordinal=0,
        text="一段可检索内容",
        section_path=("概览",),
    )
    ref = SourceRef(
        document_id=document.id,
        document_title=document.title,
        source_type=document.source_type,
        version_id=version.id,
        chunk_id=chunk.id,
        section_path=chunk.section_path,
        locator=SourceLocator(kind="heading", value="概览", label="概览"),
        excerpt=chunk.text,
    )

    assert {item.schema_version for item in (collection, document, version, chunk, ref)} == {
        "1"
    }
    assert ref.locator.kind == "heading"
    assert ref.locator.value == "概览"
    assert ref.model_dump(mode="json")["schema_version"] == "1"


def test_retrieval_scope_requires_document_for_version_and_has_stable_fingerprint():
    with pytest.raises(ValidationError):
        RetrievalScope(version_id="v-1")

    first = RetrievalScope(collection_id="c", document_id="d", version_id="v")
    second = RetrievalScope(collection_id="c", document_id="d", version_id="v")
    assert first.cache_fingerprint() == second.cache_fingerprint()
    assert first.cache_fingerprint() != RetrievalScope(document_id="d").cache_fingerprint()
