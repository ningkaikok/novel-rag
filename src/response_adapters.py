"""通用领域对象到 API/trace 边界的纯适配函数。

现有 API 仍返回兼容的 ``novel/chunk_id/text`` 形状；本模块为后续 API/trace 切换
提供不依赖 FastAPI 的通用引用 payload。完整正文不进入引用对象，excerpt 始终限制
为 80 字以内。
"""

from __future__ import annotations

from typing import cast

from chunk_model import SourceChunk
from domain_models import Document, DocumentChunk, LocatorKind, SourceLocator, SourceRef
from legacy_novel import LegacyNovelAdapter

EXCERPT_MAX_CHARS = 80


def source_ref_from_document_chunk(
    chunk: DocumentChunk,
    document: Document,
    *,
    excerpt: str | None = None,
) -> SourceRef:
    """将通用 chunk 转成完整身份链和机器 locator 的安全引用。"""

    if chunk.page_number is not None:
        kind: LocatorKind = "page"
        value = str(chunk.page_number)
        label = f"第 {value} 页"
    elif chunk.section_path:
        kind = "heading"
        value = "/".join(chunk.section_path)
        label = value
    else:
        kind = "chunk"
        value = str(chunk.ordinal)
        label = f"片段 {value}"
    return SourceRef(
        document_id=document.id,
        document_title=document.title,
        source_type=document.source_type,
        version_id=chunk.document_version_id,
        chunk_id=chunk.id,
        section_path=chunk.section_path,
        locator=SourceLocator(kind=kind, value=value, label=label),
        excerpt=(excerpt if excerpt is not None else chunk.text)[:EXCERPT_MAX_CHARS],
    )


def source_ref_from_legacy_chunk(
    chunk: SourceChunk,
    *,
    source_hash: str | None = None,
    excerpt: str | None = None,
) -> SourceRef:
    """通过 LegacyNovelAdapter 暴露旧检索结果的通用引用。"""

    safe_excerpt = (excerpt if excerpt is not None else chunk.text)[:EXCERPT_MAX_CHARS]
    return LegacyNovelAdapter.source_ref(
        chunk,
        source_hash=source_hash,
        excerpt=safe_excerpt,
    )


def source_ref_payload(ref: SourceRef) -> dict[str, object]:
    """返回 JSON-safe 的 API/trace payload，并再次守住 excerpt 红线。"""

    payload = cast(dict[str, object], ref.model_dump(mode="json"))
    payload["excerpt"] = str(payload.get("excerpt") or "")[:EXCERPT_MAX_CHARS]
    return payload
