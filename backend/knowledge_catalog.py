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
