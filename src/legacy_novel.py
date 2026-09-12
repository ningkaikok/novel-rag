"""旧 ``novel`` / ``novel_chunks`` 概念到通用领域模型的兼容适配层。

这是 Phase 1 唯一允许处理小说特有字段（``novel``、``chapter_title``、``chunk_id``）
的模块。适配层不读写数据库，也不改变旧表结构；它只提供确定性的身份、版本、片段
和引用映射，供后续 V2 migration 与当前检索结果逐步接入。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Protocol

from domain_models import (
    Collection,
    Document,
    DocumentChunk,
    DocumentVersion,
    RetrievalScope,
    SourceLocator,
    SourceRef,
)


class LegacyNovelChunk(Protocol):
    """loader.Chunk 与 retrieval SourceChunk 共同满足的最小字段集合。"""

    novel: str
    chunk_id: int
    text: str
    chapter_title: str | None


def _stable_id(kind: str, *parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return f"legacy:{kind}:{sha256(payload).hexdigest()[:24]}"


class LegacyNovelAdapter:
    """把一个旧小说名映射到通用 collection/document/version 身份。"""

    SOURCE_TYPE = "novel"
    PARSER_NAME = "legacy-novel-txt"
    PARSER_VERSION = "1"

    @classmethod
    def collection_id(cls, novel: str) -> str:
        return _stable_id("collection", novel)

    @classmethod
    def document_id(cls, novel: str) -> str:
        return _stable_id("document", novel)

    @classmethod
    def version_id(cls, novel: str, *, source_hash: str | None = None) -> str:
        return _stable_id("version", cls.document_id(novel), source_hash or "v1")

    @classmethod
    def collection(cls, novel: str) -> Collection:
        return Collection(id=cls.collection_id(novel), name=novel)

    @classmethod
    def document(cls, novel: str) -> Document:
        return Document(
            id=cls.document_id(novel),
            collection_id=cls.collection_id(novel),
            title=novel,
            source_type=cls.SOURCE_TYPE,
            metadata={"legacy_novel": novel},
        )

    @classmethod
    def version(cls, novel: str, *, source_hash: str | None = None) -> DocumentVersion:
        return DocumentVersion(
            id=cls.version_id(novel, source_hash=source_hash),
            document_id=cls.document_id(novel),
            version_no=1,
            source_hash=source_hash,
            parser_name=cls.PARSER_NAME,
            parser_version=cls.PARSER_VERSION,
        )

    @classmethod
    def scope(
        cls,
        novel: str,
        *,
        source_hash: str | None = None,
    ) -> RetrievalScope:
        """构造只限定当前旧小说的通用检索范围。"""

        return RetrievalScope(
            collection_id=cls.collection_id(novel),
            document_id=cls.document_id(novel),
            version_id=cls.version_id(novel, source_hash=source_hash),
        )

    @classmethod
    def novel_for_scope(
        cls,
        scope: RetrievalScope,
        novels: Sequence[str],
        *,
        source_hashes: Mapping[str, str | None] | None = None,
    ) -> list[str]:
        """把可识别的旧范围还原为小说名；无法识别时返回空列表。

        ``novels`` 来自当前数据库，适配器不会猜测通用 ID 与小说名的对应关系。
        这让未来 V2 范围可以安全传入旧检索器，而不会意外扩大到全库。

        version scope 需要额外的 V1 manifest source_hash 才能映射到当前版本；没有
        这份证明时，只有默认的 ``v1`` 兼容版本 ID 可识别，其他未知版本一律不匹配。
        """

        matched: list[str] = []
        for novel in novels:
            candidate = cls.scope(novel)
            if scope.collection_id and scope.collection_id != candidate.collection_id:
                continue
            if scope.document_id and scope.document_id != candidate.document_id:
                continue
            if scope.version_id:
                known_versions = {candidate.version_id}
                if source_hashes and novel in source_hashes and source_hashes[novel]:
                    known_versions.add(cls.version_id(novel, source_hash=source_hashes[novel]))
                if scope.version_id not in known_versions:
                    continue
            matched.append(novel)
        return matched

    @classmethod
    def chunk(
        cls,
        chunk: LegacyNovelChunk,
        *,
        source_hash: str | None = None,
    ) -> DocumentChunk:
        """将旧 loader/retrieval 片段映射为通用 ``DocumentChunk``。"""

        chapter_title = getattr(chunk, "chapter_title", None)
        context = getattr(chunk, "context", "") or ""
        metadata: dict[str, object] = {
            "legacy_novel": chunk.novel,
            "legacy_chunk_id": int(chunk.chunk_id),
        }
        if chapter_title:
            metadata["legacy_chapter_title"] = chapter_title
        return DocumentChunk(
            id=_stable_id(
                "chunk",
                cls.version_id(chunk.novel, source_hash=source_hash),
                str(chunk.chunk_id),
            ),
            document_version_id=cls.version_id(chunk.novel, source_hash=source_hash),
            ordinal=int(chunk.chunk_id),
            text=chunk.text,
            section_path=(chapter_title,) if chapter_title else (),
            context=context,
            metadata=metadata,
        )

    @classmethod
    def chunk_from_row(
        cls,
        row: Mapping[str, object],
        *,
        source_hash: str | None = None,
    ) -> DocumentChunk:
        """将 ``novel_chunks`` 查询行映射为通用片段，不复制数据库 schema。"""

        novel = str(row["novel"])
        chapter_title = row.get("chapter_title")
        context = row.get("context") or ""
        return cls.chunk(
            _RowChunk(
                novel=novel,
                chunk_id=int(str(row["chunk_id"])),
                text=str(row["text"]),
                chapter_title=str(chapter_title) if chapter_title is not None else None,
                context=str(context),
            ),
            source_hash=source_hash,
        )

    @classmethod
    def source_ref(
        cls,
        chunk: LegacyNovelChunk,
        *,
        source_hash: str | None = None,
        excerpt: str | None = None,
    ) -> SourceRef:
        """生成不含完整正文的通用引用，保留旧章节/片段定位能力。"""

        domain_chunk = cls.chunk(chunk, source_hash=source_hash)
        chapter_title = getattr(chunk, "chapter_title", None)
        value = str(int(chunk.chunk_id))
        return SourceRef(
            document_id=cls.document_id(chunk.novel),
            document_title=chunk.novel,
            source_type=cls.SOURCE_TYPE,
            version_id=domain_chunk.document_version_id,
            chunk_id=domain_chunk.id,
            section_path=domain_chunk.section_path,
            locator=SourceLocator(
                kind="chunk",
                value=value,
                label=chapter_title or f"片段 {value}",
            ),
            excerpt=(excerpt if excerpt is not None else chunk.text[:80]),
        )


class _RowChunk:
    def __init__(
        self,
        *,
        novel: str,
        chunk_id: int,
        text: str,
        chapter_title: str | None,
        context: str,
    ) -> None:
        self.novel = novel
        self.chunk_id = chunk_id
        self.text = text
        self.chapter_title = chapter_title
        self.context = context
