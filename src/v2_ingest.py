"""通用文档的解析、embedding、BM25 和 V2 发布流水线。

这是 V2 文档索引的纯业务入口，Web/API 或后台 worker 只需提供文件 bytes、标题和
embedding 模型即可复用。默认发布到 shadow，生产调用方必须显式选择 ``v2``；不会
修改 V1 表，也不会自动创建数据库 schema。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from config import CHUNK_OVERLAP, CHUNK_SIZE
from domain_models import Collection, Document
from parsers import ParserLimits, _BaseParser, parser_for_path
from postgres import connect
from tokenizer import term_frequencies
from v2_repository import (
    V2IndexInput,
    V2PublishResult,
    build_v2_publication_plan,
    publish_v2_index,
)

EmbeddingModel = object
ProgressFn = Callable[[str, int, str], None]
CancelCheck = Callable[[], None]


@dataclass(frozen=True)
class V2IngestResult:
    document_id: str
    version_id: str
    parser: str
    chunks: int
    terms: int
    published: V2PublishResult | None = None


def _stable_id(kind: str, *parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return f"knowledge:{kind}:{hashlib.sha256(payload).hexdigest()[:24]}"


def _pipeline_hash(parser: _BaseParser, embedder: object) -> str:
    model_name = getattr(embedder, "model_name", None) or getattr(embedder, "name", None)
    settings = {
        "version": 1,
        "parser": parser.name,
        "parser_version": parser.version,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": str(model_name or type(embedder).__name__),
    }
    raw = json.dumps(settings, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _encode(embedder: object, texts: Sequence[str]):
    encode = getattr(embedder, "encode", None)
    if encode is None:
        raise TypeError("embedding model 必须提供 encode 方法")
    return encode(texts, normalize_embeddings=True, show_progress_bar=False)


def build_v2_index_input(
    payload: bytes,
    *,
    title: str,
    collection_name: str,
    embedder: object,
    parser: _BaseParser | None = None,
    limits: ParserLimits | None = None,
    progress: ProgressFn | None = None,
    cancel_check: CancelCheck | None = None,
    batch_size: int = 32,
) -> V2IndexInput:
    """把单个通用文档转换为可校验、可发布的 ``V2IndexInput``。"""

    if not title.strip() or not collection_name.strip():
        raise ValueError("title 和 collection_name 不能为空")
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    parser = parser or parser_for_path(title, limits=limits)
    chunks = parser.parse(payload, title=title)
    if not chunks:
        raise ValueError("文档没有可索引的内容")
    version = parser.document_version(payload, title=title)
    collection = Collection(
        id=_stable_id("collection", collection_name),
        name=collection_name,
    )
    document = Document(
        id=version.document_id,
        collection_id=collection.id,
        title=title,
        source_type=parser.source_type,
        metadata={
            "parser_name": parser.name,
            "parser_version": parser.version,
            "source_hash": version.source_hash,
        },
    )

    if progress:
        progress("parse", 5, f"已解析 {len(chunks)} 个片段")
    check = cancel_check or (lambda: None)
    embeddings: dict[str, Sequence[float]] = {}
    texts = [chunk.text for chunk in chunks]
    for start in range(0, len(texts), batch_size):
        check()
        batch = texts[start : start + batch_size]
        encoded = _encode(embedder, batch)
        for chunk, embedding in zip(chunks[start : start + batch_size], encoded, strict=True):
            embeddings[chunk.id] = tuple(float(value) for value in embedding)
        if progress:
            done = min(start + len(batch), len(texts))
            progress(
                "embedding", int(5 + done / len(texts) * 60), f"Embedding {done}/{len(texts)}"
            )

    terms: dict[str, Mapping[str, int]] = {}
    for index, chunk in enumerate(chunks, start=1):
        if index == 1 or index % 25 == 0:
            check()
        terms[chunk.id] = term_frequencies(chunk.text)
        if progress and (index % 100 == 0 or index == len(chunks)):
            progress("bm25", int(65 + index / len(chunks) * 30), f"BM25 {index}/{len(chunks)}")

    return V2IndexInput(
        collection=collection,
        document=document,
        version=version,
        chunks=tuple(chunks),
        embeddings=embeddings,
        terms=terms,
        pipeline_hash=_pipeline_hash(parser, embedder),
    )


def index_v2_document(
    payload: bytes,
    *,
    title: str,
    collection_name: str,
    embedder: object,
    embedding_dimension: int,
    storage_schema: str = "shadow",
    parser: _BaseParser | None = None,
    limits: ParserLimits | None = None,
    progress: ProgressFn | None = None,
    cancel_check: CancelCheck | None = None,
) -> V2IngestResult:
    """构建并显式发布单个文档；数据库写入只发生在最后的 V2 事务。"""

    item = build_v2_index_input(
        payload,
        title=title,
        collection_name=collection_name,
        embedder=embedder,
        parser=parser,
        limits=limits,
        progress=progress,
        cancel_check=cancel_check,
    )
    # DocumentVersion 的唯一约束是 (document_id, version_no)。parser 只负责根据
    # 内容产生稳定 version id，这里在发布前读取当前最大版本号：同内容重传复用已有
    # version_no，不同内容顺延到下一个版本，避免同名文档永远撞在 version 1。
    with connect() as conn:
        existing = conn.execute(
            """
            SELECT version_no
            FROM knowledge_v2.document_versions
            WHERE document_id = %s AND source_hash = %s
            """,
            (item.document.id, item.version.source_hash),
        ).fetchone()
        if existing is None:
            latest = conn.execute(
                """
                SELECT COALESCE(MAX(version_no), 0) AS version_no
                FROM knowledge_v2.document_versions
                WHERE document_id = %s
                """,
                (item.document.id,),
            ).fetchone()
            if latest is None:
                raise RuntimeError("无法读取文档最新版本号")
            version_no = int(cast(Any, latest["version_no"])) + 1
        else:
            version_no = int(existing["version_no"])
    item = V2IndexInput(
        **{
            **item.__dict__,
            "version": item.version.model_copy(update={"version_no": version_no}),
        }
    )
    plan = build_v2_publication_plan([item], embedding_dimension=embedding_dimension)
    if progress:
        progress("database", 98, "正在发布 V2 文档索引")
    with connect() as conn:
        published = publish_v2_index(cast(Any, conn), plan, storage_schema=storage_schema)
    if progress:
        progress("complete", 100, "V2 文档索引已发布")
    return V2IngestResult(
        document_id=item.document.id,
        version_id=item.version.id,
        parser=item.version.parser_name,
        chunks=plan.chunk_count,
        terms=plan.term_count,
        published=published,
    )
