"""显式连接通用领域 SourceRef 与旧 ToolResult SourceRef。

两个 ``SourceRef`` 类型刻意不互相继承或隐式转换：ToolResult 仍保留
``novel/chapter/chunk_id`` 旧字段，只有这个 adapter 负责把 LegacyNovelAdapter
生成的通用身份链投影进去。这样 MCP/Agent 继续兼容旧客户端，同时不会把完整正文
或通用定位误当成旧小说字段。
"""

from __future__ import annotations

from chunk_model import SourceChunk
from domain_models import SourceRef as DomainSourceRef
from legacy_novel import LegacyNovelAdapter
from tool_spec import EXCERPT_MAX_CHARS
from tool_spec import SourceRef as ToolSourceRef


def tool_source_ref_from_domain(
    ref: DomainSourceRef,
    *,
    legacy_novel: str,
    legacy_chapter: str = "",
    legacy_chunk_id: int | None = None,
) -> ToolSourceRef:
    """将一个通用引用显式投影为旧 ToolResult 形状。

    旧字段无法表达任意 heading/page locator，因此非 legacy 调用必须显式提供
    可逆的 chunk id；否则拒绝映射，而不是猜测或扩大正文输出。
    """

    if legacy_chunk_id is None:
        try:
            legacy_chunk_id = int(ref.locator.value)
        except ValueError as exc:
            raise ValueError("通用 locator 无法安全映射为旧 chunk_id") from exc
    return ToolSourceRef(
        novel=legacy_novel,
        chapter=legacy_chapter or (ref.section_path[-1] if ref.section_path else ""),
        chunk_id=legacy_chunk_id,
        excerpt=ref.excerpt[:EXCERPT_MAX_CHARS],
        document_id=ref.document_id,
        version_id=ref.version_id,
        locator_kind=ref.locator.kind,
        locator_value=ref.locator.value,
        source_type=ref.source_type,
    )


def tool_source_ref_from_legacy_chunk(source: SourceChunk) -> ToolSourceRef:
    """通过 LegacyNovelAdapter 生成通用引用，再显式投影回旧 ToolResult。"""

    return tool_source_ref_from_domain(
        LegacyNovelAdapter.source_ref(
            source,
            excerpt=(source.text or "")[:EXCERPT_MAX_CHARS],
        ),
        legacy_novel=source.novel,
        legacy_chapter=source.chapter_title or "",
        legacy_chunk_id=source.chunk_id,
    )
