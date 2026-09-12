#!/usr/bin/env python3
"""比较当前 V1/V2 迁移结果的稳定身份、数量和定位，不输出正文。"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from domain_models import Document, DocumentChunk, DocumentVersion  # noqa: E402
from postgres import connect  # noqa: E402
from v2_shadow import (  # noqa: E402
    compare_shadow,
    snapshot_from_v1_rows,
    snapshot_from_v2_document,
)


def main() -> int:
    with connect() as conn:
        manifests = {
            str(row["novel"]): row
            for row in conn.execute(
                "SELECT novel, source_hash FROM public.index_manifest ORDER BY novel"
            ).fetchall()
        }
        v1_rows = conn.execute(
            """
            SELECT novel, chunk_id, chapter_title
            FROM public.novel_chunks
            ORDER BY novel, chunk_id
            """
        ).fetchall()
        v2_rows = conn.execute(
            """
            SELECT d.metadata ->> 'legacy_novel' AS novel,
                   c.id AS collection_id, c.name AS collection_name,
                   d.id AS document_id, d.title AS document_title,
                   d.source_type, d.metadata AS document_metadata,
                   dv.id AS version_id, dv.version_no, dv.source_hash,
                   dv.parser_name, dv.parser_version,
                   dc.id AS chunk_id, dc.ordinal, dc.section_path,
                   dc.page_number, dc.text, dc.context, dc.metadata AS chunk_metadata
            FROM knowledge_v2.document_chunks dc
            JOIN knowledge_v2.document_versions dv ON dv.id = dc.document_version_id
            JOIN knowledge_v2.documents d ON d.id = dv.document_id
            JOIN knowledge_v2.collections c ON c.id = d.collection_id
            ORDER BY novel, dc.ordinal
            """
        ).fetchall()

    v1_by_novel: dict[str, list[dict]] = defaultdict(list)
    for row in v1_rows:
        v1_by_novel[str(row["novel"])].append(dict(row))
    v2_by_novel: dict[str, list[dict]] = defaultdict(list)
    for row in v2_rows:
        novel = row.get("novel")
        if novel:
            v2_by_novel[str(novel)].append(dict(row))

    comparisons: dict[str, object] = {}
    mismatch_count = 0
    for novel in sorted(set(v1_by_novel) | set(v2_by_novel)):
        source_hash = manifests.get(novel, {}).get("source_hash")
        v1_snapshot = snapshot_from_v1_rows(
            novel,
            v1_by_novel.get(novel, []),
            source_hash=str(source_hash) if source_hash else None,
        )
        rows = v2_by_novel.get(novel, [])
        if rows:
            first = rows[0]
            document = Document(
                id=str(first["document_id"]),
                collection_id=str(first["collection_id"]),
                title=str(first["document_title"]),
                source_type=str(first["source_type"]),  # type: ignore[arg-type]
                metadata=dict(first.get("document_metadata") or {}),
            )
            version = DocumentVersion(
                id=str(first["version_id"]),
                document_id=document.id,
                version_no=int(first["version_no"]),
                source_hash=str(first["source_hash"]) if first.get("source_hash") else None,
                parser_name=str(first["parser_name"]),
                parser_version=str(first["parser_version"]),
            )
            chunks = tuple(
                DocumentChunk(
                    id=str(row["chunk_id"]),
                    document_version_id=version.id,
                    ordinal=int(row["ordinal"]),
                    text=str(row["text"]),
                    section_path=tuple(str(item) for item in (row.get("section_path") or [])),
                    page_number=(
                        int(row["page_number"]) if row.get("page_number") is not None else None
                    ),
                    context=str(row.get("context") or ""),
                    metadata=dict(row.get("chunk_metadata") or {}),
                )
                for row in rows
            )
            v2_snapshot = snapshot_from_v2_document(document, version, chunks)
        else:
            v2_snapshot = snapshot_from_v1_rows(novel, [], source_hash=None)
        comparison = compare_shadow(v1_snapshot, v2_snapshot)
        mismatch_count += len(comparison.mismatches)
        comparisons[novel] = {
            "v1_chunks": len(v1_snapshot.chunks),
            "v2_chunks": len(v2_snapshot.chunks),
            "matches": comparison.matches,
            "categories": list(comparison.categories),
        }

    result = {
        "documents": len(comparisons),
        "mismatches": mismatch_count,
        "matches": mismatch_count == 0,
        "by_document": comparisons,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if mismatch_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
