"""把现有 V1 文件和 manifest 映射成无正文的通用知识库目录。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from backend.schemas import (
    KnowledgeCollectionList,
    KnowledgeCollectionSummary,
    KnowledgeDocumentList,
    KnowledgeDocumentSummary,
    KnowledgeVersionSummary,
)
from legacy_novel import LegacyNovelAdapter
from postgres import connect


def build_v1_documents(
    files: Sequence[Path], manifests: Mapping[str, Mapping[str, Any]] | None = None
) -> list[KnowledgeDocumentSummary]:
    """从 TXT 文件构造稳定目录；manifest 缺失时仍返回 source-only 文档。"""

    manifest_map = manifests or {}
    documents: list[KnowledgeDocumentSummary] = []
    for path in sorted(files, key=lambda item: item.name):
        legacy_name = path.name
        document = LegacyNovelAdapter.document(legacy_name)
        record = manifest_map.get(legacy_name) or manifest_map.get(path.stem) or {}
        source_hash = record.get("source_hash")
        source_hash = source_hash if isinstance(source_hash, str) else None
        version = LegacyNovelAdapter.version(legacy_name, source_hash=source_hash)
        chunk_count = record.get("chunk_count")
        chunk_count = chunk_count if isinstance(chunk_count, int) else None
        indexed = bool(record) and source_hash is not None
        documents.append(
            KnowledgeDocumentSummary(
                id=document.id,
                collection_id=document.collection_id,
                title=path.stem,
                source_type=document.source_type,
                metadata={"legacy_novel": legacy_name, "storage_schema": "v1"},
                status="indexed" if indexed else "source_only",
                versions=[
                    KnowledgeVersionSummary(
                        id=version.id,
                        version_no=version.version_no,
                        source_hash=version.source_hash,
                        parser_name=version.parser_name,
                        parser_version=version.parser_version,
                        chunk_count=chunk_count,
                    )
                ],
            )
        )
    return documents


def build_v1_catalog(
    novels_dir: Path, manifest_loader: Any
) -> tuple[KnowledgeCollectionList, KnowledgeDocumentList]:
    """读取 V1 元数据；数据库不可用时退化为文件目录，不影响旧书架。"""

    try:
        manifests = manifest_loader()
    except Exception:
        manifests = {}
    documents = build_v1_documents(list(novels_dir.glob("*.txt")), manifests)
    collections = KnowledgeCollectionList(
        collections=[
            KnowledgeCollectionSummary(
                id=document.collection_id,
                name=document.title,
                document_count=1,
            )
            for document in documents
        ]
    )
    return collections, KnowledgeDocumentList(documents=documents)


def build_v2_catalog() -> tuple[KnowledgeCollectionList, KnowledgeDocumentList]:
    """从 V2 真实目录读取无正文的文档/版本摘要。"""

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT c.id AS collection_id, c.name AS collection_name,
                   d.id AS document_id, d.title, d.source_type, d.metadata,
                   dv.id AS version_id, dv.version_no, dv.source_hash,
                   dv.parser_name, dv.parser_version,
                   im.chunk_count
            FROM knowledge_v2.documents d
            JOIN knowledge_v2.collections c ON c.id = d.collection_id
            LEFT JOIN knowledge_v2.document_versions dv ON dv.document_id = d.id
            LEFT JOIN knowledge_v2.index_manifests im ON im.document_version_id = dv.id
            ORDER BY c.name, d.title, dv.version_no
            """
        ).fetchall()

    grouped: dict[str, dict[str, Any]] = {}
    collection_names: dict[str, str] = {}
    for row in rows:
        document_id = str(row["document_id"])
        entry = grouped.setdefault(
            document_id,
            {
                "id": document_id,
                "collection_id": str(row["collection_id"]),
                "title": str(row["title"]),
                "source_type": str(row["source_type"]),
                "metadata": dict(row.get("metadata") or {}),
                "versions": [],
            },
        )
        collection_names[str(row["collection_id"])] = str(row["collection_name"])
        if row.get("version_id") is not None:
            entry["versions"].append(
                KnowledgeVersionSummary(
                    id=str(row["version_id"]),
                    version_no=int(row["version_no"]),
                    source_hash=(str(row["source_hash"]) if row.get("source_hash") else None),
                    parser_name=str(row["parser_name"]),
                    parser_version=str(row["parser_version"]),
                    chunk_count=(
                        int(row["chunk_count"]) if row.get("chunk_count") is not None else None
                    ),
                )
            )

    documents = [
        KnowledgeDocumentSummary(
            **entry,
            status="indexed"
            if any(version.chunk_count is not None for version in entry["versions"])
            else "source_only",
        )
        for entry in grouped.values()
    ]
    document_counts: dict[str, int] = {}
    for document in documents:
        document_counts[document.collection_id] = (
            document_counts.get(document.collection_id, 0) + 1
        )
    collections = KnowledgeCollectionList(
        collections=[
            KnowledgeCollectionSummary(
                id=collection_id,
                name=collection_names[collection_id],
                document_count=document_counts.get(collection_id, 0),
            )
            for collection_id in sorted(
                collection_names, key=lambda item: collection_names[item]
            )
        ]
    )
    return collections, KnowledgeDocumentList(documents=documents)
