#!/usr/bin/env python3
"""把当前 V1 novel_chunks/index 结果安全发布到独立 V2 schema。

默认只做 dry-run。只有显式传入 ``--apply`` 才会按文档逐个在事务内发布；V1 表不
会被删除或更新，V2 仍然作为 shadow 数据源，不会改变当前 API/RAG 读路径。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from domain_models import DocumentChunk  # noqa: E402
from legacy_novel import LegacyNovelAdapter  # noqa: E402
from postgres import connect  # noqa: E402
from v2_repository import (  # noqa: E402
    V2IndexInput,
    build_v2_publication_plan,
    publish_v2_index,
)


def _vector(value: object) -> tuple[float, ...]:
    raw = str(value).strip()
    if not (raw.startswith("[") and raw.endswith("]")):
        raise ValueError("V1 embedding 不是 pgvector 文本格式")
    return tuple(float(item) for item in raw[1:-1].split(",") if item.strip())


def _load_input(conn, novel: str, manifest: Mapping[str, object]) -> V2IndexInput:
    source_hash = str(manifest.get("source_hash") or "")
    pipeline_hash = str(manifest.get("pipeline_hash") or "")
    if not source_hash or not pipeline_hash:
        raise ValueError(f"V1 manifest 缺少 hash: {novel}")

    rows = conn.execute(
        """
        SELECT novel, chunk_id, chapter_title, text, context, embedding::text AS embedding
        FROM public.novel_chunks
        WHERE novel = %s
        ORDER BY chunk_id
        """,
        (novel,),
    ).fetchall()
    expected_count = manifest.get("chunk_count")
    if isinstance(expected_count, int) and expected_count != len(rows):
        raise ValueError(f"V1 manifest chunk_count 不一致: {novel}")

    chunks: list[DocumentChunk] = [
        LegacyNovelAdapter.chunk_from_row(row, source_hash=source_hash) for row in rows
    ]
    embeddings = {
        chunk.id: _vector(row["embedding"]) for chunk, row in zip(chunks, rows, strict=True)
    }
    terms: dict[str, dict[str, int]] = defaultdict(dict)
    term_rows = conn.execute(
        """
        SELECT chunk_id, term, tf
        FROM public.chunk_terms
        WHERE novel = %s
        ORDER BY chunk_id, term
        """,
        (novel,),
    ).fetchall()
    by_ordinal = {chunk.ordinal: chunk.id for chunk in chunks}
    for row in term_rows:
        ordinal = int(row["chunk_id"])
        if ordinal not in by_ordinal:
            raise ValueError(f"V1 term 指向不存在的 chunk: {novel}/{ordinal}")
        terms[by_ordinal[ordinal]][str(row["term"])] = int(row["tf"])

    document = LegacyNovelAdapter.document(novel)
    return V2IndexInput(
        collection=LegacyNovelAdapter.collection(novel),
        document=document,
        version=LegacyNovelAdapter.version(novel, source_hash=source_hash),
        chunks=tuple(chunks),
        embeddings=embeddings,
        terms=dict(terms),
        pipeline_hash=pipeline_hash,
    )


def _load_manifest_rows(conn) -> list[Mapping[str, object]]:
    manifests = conn.execute(
        """
        SELECT novel, source_hash, pipeline_hash, chunk_count
        FROM public.index_manifest
        ORDER BY novel
        """
    ).fetchall()
    return [dict(row) for row in manifests]


def _build_document_plan(conn, manifest: Mapping[str, object], dimension: int):
    item = _load_input(conn, str(manifest["novel"]), manifest)
    return build_v2_publication_plan([item], embedding_dimension=dimension)


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 → V2 shadow migration")
    parser.add_argument("--apply", action="store_true", help="显式执行 V2 shadow 发布")
    parser.add_argument("--embedding-dimension", type=int, default=512)
    args = parser.parse_args()

    try:
        documents = chunks = terms = 0
        version_ids: list[str] = []
        with connect() as read_conn:
            manifest_rows = _load_manifest_rows(read_conn)
            if args.apply:
                with connect() as write_conn:
                    for manifest in manifest_rows:
                        plan = _build_document_plan(
                            read_conn, manifest, args.embedding_dimension
                        )
                        documents += len(plan.documents)
                        chunks += plan.chunk_count
                        terms += plan.term_count
                        version_ids.extend(document.version.id for document in plan.documents)
                        publish_v2_index(write_conn, plan, storage_schema="shadow")
            else:
                for manifest in manifest_rows:
                    plan = _build_document_plan(read_conn, manifest, args.embedding_dimension)
                    documents += len(plan.documents)
                    chunks += plan.chunk_count
                    terms += plan.term_count
                    version_ids.extend(document.version.id for document in plan.documents)
        result: dict[str, object] = {
            "mode": "apply-shadow" if args.apply else "dry-run",
            "schema": "knowledge_v2",
            "documents": documents,
            "chunks": chunks,
            "terms": terms,
            "embedding_dimension": args.embedding_dimension,
            "version_ids": version_ids,
        }
        if args.apply:
            result["published"] = {"documents": documents, "chunks": chunks, "terms": terms}
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"V2 迁移失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
