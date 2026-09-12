from contextlib import contextmanager

import pytest

from domain_models import Collection, Document, DocumentChunk, DocumentVersion
from v2_repository import (
    V2IndexInput,
    V2PublicationError,
    V2PublishDisabled,
    build_v2_publication_plan,
    dry_run_v2_index,
    publish_v2_index,
)


class _FakeExecutor:
    def __init__(self, *, fail_on: str | None = None):
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.events: list[str] = []
        self.fail_on = fail_on

    def execute(self, query: str, params=()):
        if self.fail_on and self.fail_on in query:
            raise RuntimeError("模拟数据库失败")
        self.calls.append((query, tuple(params)))

    @contextmanager
    def transaction(self):
        self.events.append("begin")
        try:
            yield self
        except Exception:
            self.events.append("rollback")
            raise
        else:
            self.events.append("commit")


def _input(document_id: str = "doc-1", *, pipeline_hash: str = "pipeline-1"):
    collection = Collection(id="collection-1", name="测试集合")
    document = Document(
        id=document_id,
        collection_id=collection.id,
        title="测试文档",
        source_type="text",
    )
    version = DocumentVersion(
        id=f"{document_id}-v1",
        document_id=document.id,
        version_no=1,
        source_hash="source-1",
        parser_name="txt",
        parser_version="1.0",
    )
    chunks = tuple(
        DocumentChunk(
            id=f"{document_id}-chunk-{ordinal}",
            document_version_id=version.id,
            ordinal=ordinal,
            text=f"短文本 {ordinal}",
            section_path=("第一节",) if ordinal == 0 else (),
        )
        for ordinal in range(2)
    )
    return V2IndexInput(
        collection=collection,
        document=document,
        version=version,
        chunks=chunks,
        embeddings={chunk.id: (0.1, 0.2, 0.3) for chunk in chunks},
        terms={chunks[0].id: {"短文": 2}, chunks[1].id: {"文本": 1}},
        pipeline_hash=pipeline_hash,
    )


def _plan():
    return build_v2_publication_plan([_input()], embedding_dimension=3)


def test_publication_sql_is_ordered_idempotent_and_manifest_is_last():
    plan = _plan()
    first = _FakeExecutor()
    second = _FakeExecutor()

    result = publish_v2_index(first, plan, storage_schema="v2")
    publish_v2_index(second, plan, storage_schema="v2")

    assert result.documents == 1
    assert result.chunks == 2
    assert result.terms == 2
    assert first.calls == second.calls
    assert first.events == ["begin", "commit"]
    assert "collections" in first.calls[0][0]
    assert "documents" in first.calls[1][0]
    assert "document_versions" in first.calls[2][0]
    assert "DELETE FROM knowledge_v2.chunk_terms" in first.calls[3][0]
    assert "DELETE FROM knowledge_v2.document_chunks" in first.calls[4][0]
    chunk_call_indices = [
        index
        for index, (query, _params) in enumerate(first.calls)
        if "document_chunks" in query and not query.lstrip().startswith("DELETE")
    ]
    term_call_indices = [
        index
        for index, (query, _params) in enumerate(first.calls)
        if "chunk_terms" in query and not query.lstrip().startswith("DELETE")
    ]
    assert chunk_call_indices
    assert term_call_indices
    assert max(chunk_call_indices) < min(term_call_indices)
    assert "index_manifests" in first.calls[-1][0]
    assert all(
        "ON CONFLICT" in query
        for query, _params in first.calls
        if not query.lstrip().startswith("DELETE")
    )


def test_republishing_reduced_version_replaces_old_v2_rows():
    full = _input()
    reduced = V2IndexInput(
        collection=full.collection,
        document=full.document,
        version=full.version,
        chunks=(full.chunks[0],),
        embeddings={full.chunks[0].id: full.embeddings[full.chunks[0].id]},
        terms={full.chunks[0].id: full.terms[full.chunks[0].id]},
        pipeline_hash=full.pipeline_hash,
    )
    executor = _FakeExecutor()

    publish_v2_index(executor, _plan(), storage_schema="v2")
    before = len(executor.calls)
    publish_v2_index(
        executor,
        build_v2_publication_plan([reduced], embedding_dimension=3),
        storage_schema="v2",
    )
    replacement_calls = executor.calls[before:]
    queries = [query for query, _params in replacement_calls]

    assert any("DELETE FROM knowledge_v2.chunk_terms" in query for query in queries)
    assert any("DELETE FROM knowledge_v2.document_chunks" in query for query in queries)
    manifest_params = next(
        params for query, params in replacement_calls if "index_manifests" in query
    )
    assert manifest_params[3] == 1


def test_transaction_failure_rolls_back_and_does_not_publish_manifest():
    executor = _FakeExecutor(fail_on="chunk_terms")

    with pytest.raises(RuntimeError, match="模拟数据库失败"):
        publish_v2_index(executor, _plan(), storage_schema="shadow")

    assert executor.events == ["begin", "rollback"]
    assert not any("index_manifests" in query for query, _params in executor.calls)


def test_publication_validates_dimension_ordinal_and_pipeline_hash():
    with pytest.raises(V2PublicationError, match="维度"):
        build_v2_publication_plan([_input()], embedding_dimension=2)

    invalid = _input()
    invalid = V2IndexInput(
        **{**invalid.__dict__, "chunks": (invalid.chunks[1], invalid.chunks[0])}
    )
    with pytest.raises(V2PublicationError, match="ordinal"):
        build_v2_publication_plan([invalid], embedding_dimension=3)

    with pytest.raises(V2PublicationError, match="pipeline_hash"):
        build_v2_publication_plan([_input(pipeline_hash="")], embedding_dimension=3)

    invalid_parent = _input()
    invalid_parent = V2IndexInput(
        **{
            **invalid_parent.__dict__,
            "version": DocumentVersion(
                **{**invalid_parent.version.model_dump(), "document_id": "other-doc"}
            ),
        }
    )
    with pytest.raises(V2PublicationError, match="version 未归属"):
        build_v2_publication_plan([invalid_parent], embedding_dimension=3)


def test_default_v1_rejects_apply_and_dry_run_has_no_executor():
    plan = _plan()
    executor = _FakeExecutor()

    with pytest.raises(V2PublishDisabled, match="STORAGE_SCHEMA"):
        publish_v2_index(executor, plan)

    assert executor.calls == []
    assert dry_run_v2_index(plan)["chunks"] == 2
