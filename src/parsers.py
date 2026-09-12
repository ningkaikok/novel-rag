"""通用文档 parser：TXT、Markdown 和文本型 PDF。

Phase 3 只负责把输入安全地解析为 ``DocumentChunk``。它不连接数据库、不创建索引，
也不改变现有 ``loader.load_novel_file`` 或 V1 ``novel_chunks`` 路径。Parser 的名称、
版本、来源哈希和切分配置会进入 ``DocumentVersion``/chunk metadata，供后续 V2 索引入口
建立可复现的 pipeline fingerprint。

PDF 明确只支持可以提取文本的 PDF：优先使用 pdfplumber，缺少时回退到 pypdf；扫描型
PDF、图片 OCR、加密 PDF 和空文本 PDF 都会得到明确错误，不会静默生成空索引。
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from config import CHUNK_OVERLAP, CHUNK_SIZE
from domain_models import (
    DocumentChunk,
    DocumentVersion,
    LocatorKind,
    SourceLocator,
    SourceRef,
    SourceType,
)
from loader import (
    _chunk_paragraphs,
    _clean_text,
    _split_chapter_sections,
    _split_paragraphs,
    read_text_bytes_with_metadata,
)


class ParserError(ValueError):
    """输入内容不能安全解析。"""


class ParserLimitError(ParserError):
    """输入超出 parser 的资源限制。"""


class ParserDependencyError(ParserError):
    """当前 parser 所需的可选依赖没有安装。"""


@dataclass(frozen=True)
class ParserLimits:
    """解析阶段的资源上限；限制在进入第三方 parser 前生效。"""

    max_bytes: int = 20 * 1024 * 1024
    max_pages: int = 1000
    max_chunks: int = 20_000

    def validate(self) -> None:
        if self.max_bytes <= 0 or self.max_pages <= 0 or self.max_chunks <= 0:
            raise ValueError("parser limits must be positive")


def _stable_id(kind: str, *parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return f"parser:{kind}:{hashlib.sha256(payload).hexdigest()[:24]}"


def _source_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _markdown_heading(line: str) -> tuple[int, str] | None:
    match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line.strip())
    if not match:
        return None
    title = match.group(2).strip()
    return (len(match.group(1)), title) if title else None


def _markdown_sections(text: str) -> list[tuple[tuple[str, ...], list[str]]]:
    """按 Markdown heading 层级保留 section path，并复用现有字符切分。"""

    sections: list[tuple[tuple[str, ...], list[str]]] = []
    heading_stack: list[tuple[int, str]] = []
    paragraphs: list[str] = []
    lines: list[str] = []

    def flush_paragraph() -> None:
        if lines:
            paragraphs.append("\n".join(lines).strip())
            lines.clear()

    def flush_section() -> None:
        flush_paragraph()
        if paragraphs:
            path = tuple(title for _level, title in heading_stack)
            sections.append((path, paragraphs.copy()))
            paragraphs.clear()

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        heading = _markdown_heading(raw_line)
        if heading:
            flush_section()
            level, title = heading
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            lines.append(raw_line.strip())
        elif raw_line.strip():
            lines.append(raw_line.rstrip())
        else:
            flush_paragraph()
    flush_section()
    return sections


class _BaseParser:
    name: str
    version: str
    source_type: SourceType

    def __init__(self, limits: ParserLimits | None = None) -> None:
        self.limits = limits or ParserLimits()
        self.limits.validate()

    def _check_payload(self, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise ParserError("parser payload must be bytes")
        if len(payload) > self.limits.max_bytes:
            raise ParserLimitError(f"输入文件超过 {self.limits.max_bytes} bytes 限制")

    def _document_id(self, title: str) -> str:
        return _stable_id("document", self.name, title)

    def parse(self, payload: bytes, *, title: str) -> list[DocumentChunk]:
        """把输入解析为可索引片段；具体格式 parser 必须实现。"""

        raise NotImplementedError

    def document_version(self, payload: bytes, *, title: str) -> DocumentVersion:
        """返回与 parser 配置绑定的可复现文档版本。"""

        source_hash = _source_hash(payload)
        return DocumentVersion(
            id=_stable_id("version", self.name, self.version, title, source_hash),
            document_id=self._document_id(title),
            version_no=1,
            source_hash=source_hash,
            parser_name=self.name,
            parser_version=self.version,
        )

    def _chunks(
        self,
        payload: bytes,
        *,
        title: str,
        sections: list[tuple[tuple[str, ...], list[str]]],
        page_numbers: list[int | None] | None = None,
        parser_metadata: dict[str, object] | None = None,
    ) -> list[DocumentChunk]:
        version = self.document_version(payload, title=title)
        base_metadata: dict[str, object] = {
            "document_id": version.document_id,
            "document_title": title,
            "source_type": self.source_type,
            "parser_name": self.name,
            "parser_version": self.version,
            "source_hash": version.source_hash,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            **(parser_metadata or {}),
        }
        chunks: list[DocumentChunk] = []
        ordinal = 0
        for section_path, paragraphs in sections:
            for text in _chunk_paragraphs(paragraphs):
                page_number = (
                    page_numbers[ordinal]
                    if page_numbers and ordinal < len(page_numbers)
                    else None
                )
                chunks.append(
                    DocumentChunk(
                        id=_stable_id("chunk", version.id, str(ordinal)),
                        document_version_id=version.id,
                        ordinal=ordinal,
                        text=text,
                        section_path=section_path,
                        page_number=page_number,
                        metadata=base_metadata.copy(),
                    )
                )
                ordinal += 1
                if ordinal > self.limits.max_chunks:
                    raise ParserLimitError(f"解析结果超过 {self.limits.max_chunks} 个片段限制")
        return chunks


class TxtParser(_BaseParser):
    """复用小说现有清洗/章节识别/固定尺寸切分规则的通用 TXT parser。"""

    name = "txt"
    version = "1.0"
    source_type: SourceType = "text"

    def parse(self, payload: bytes, *, title: str) -> list[DocumentChunk]:
        self._check_payload(payload)
        text, decode_metadata = read_text_bytes_with_metadata(payload)
        cleaned = _clean_text(text)
        sections = [
            ((chapter_title,) if chapter_title else (), paragraphs)
            for chapter_title, paragraphs in _split_chapter_sections(
                _split_paragraphs(cleaned)
            )
        ]
        return self._chunks(
            payload, title=title, sections=sections, parser_metadata=decode_metadata
        )


class MarkdownParser(_BaseParser):
    """保留 ATX heading 层级的 Markdown parser；不执行 HTML/脚本。"""

    name = "markdown"
    version = "1.0"
    source_type: SourceType = "markdown"

    def parse(self, payload: bytes, *, title: str) -> list[DocumentChunk]:
        self._check_payload(payload)
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ParserError("Markdown 必须是 UTF-8 文本") from exc
        sections = _markdown_sections(text)
        return self._chunks(payload, title=title, sections=sections)


class PdfTextParser(_BaseParser):
    """优先 pdfplumber、回退 pypdf 的文本型 PDF parser，不做 OCR。"""

    name = "pdf-text"
    version = "1.0"
    source_type: SourceType = "pdf"

    def _extract_pages(self, payload: bytes) -> tuple[list[str], str]:
        try:
            import pdfplumber
        except ImportError:
            pdfplumber = None

        if pdfplumber is not None:
            try:
                with pdfplumber.open(io.BytesIO(payload)) as pdf:
                    page_count = len(pdf.pages)
                    if page_count > self.limits.max_pages:
                        raise ParserLimitError(
                            f"PDF 页数 {page_count} 超过 {self.limits.max_pages} 页限制"
                        )
                    pages = [page.extract_text() or "" for page in pdf.pages]
                return [str(text) for text in pages], "pdfplumber"
            except ParserError:
                raise
            except Exception as exc:
                raise ParserError(f"PDF 文本抽取失败（pdfplumber）：{exc}") from exc

        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ParserDependencyError("PDF parser 需要安装 pdfplumber 或 pypdf") from exc

        try:
            reader = PdfReader(io.BytesIO(payload))
            if reader.is_encrypted:
                raise ParserError("加密 PDF 暂不支持")
            page_count = len(reader.pages)
            if page_count > self.limits.max_pages:
                raise ParserLimitError(
                    f"PDF 页数 {page_count} 超过 {self.limits.max_pages} 页限制"
                )
            return [page.extract_text() or "" for page in reader.pages], "pypdf"
        except ParserError:
            raise
        except Exception as exc:
            raise ParserError(f"PDF 文本抽取失败（pypdf）：{exc}") from exc

    def parse(self, payload: bytes, *, title: str) -> list[DocumentChunk]:
        self._check_payload(payload)
        pages, backend = self._extract_pages(payload)
        sections: list[tuple[tuple[str, ...], list[str]]] = []
        page_numbers: list[int | None] = []
        for page_number, page_text in enumerate(pages, start=1):
            cleaned = _clean_text(page_text)
            if not cleaned:
                continue
            sections.append(((), _split_paragraphs(cleaned)))
            page_numbers.extend(
                [page_number] * len(_chunk_paragraphs(_split_paragraphs(cleaned)))
            )
        if not sections:
            raise ParserError("PDF 未提取到文本；扫描型 PDF/OCR 不支持")
        return self._chunks(
            payload,
            title=title,
            sections=sections,
            page_numbers=page_numbers,
            parser_metadata={"pdf_backend": backend},
        )


def source_ref_for_chunk(chunk: DocumentChunk) -> SourceRef:
    """从 parser 产出的 chunk 创建安全、可定位且不含完整正文的引用。"""

    metadata = chunk.metadata
    try:
        document_id = str(metadata["document_id"])
        title = str(metadata["document_title"])
        source_type = cast(SourceType, str(metadata["source_type"]))
    except KeyError as exc:
        raise ParserError("DocumentChunk 缺少引用所需的 document metadata") from exc
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
        document_id=document_id,
        document_title=title,
        source_type=source_type,
        version_id=chunk.document_version_id,
        chunk_id=chunk.id,
        section_path=chunk.section_path,
        locator=SourceLocator(kind=kind, value=value, label=label),
        excerpt=chunk.text[:80],
    )


PARSER_BY_SUFFIX = {
    ".txt": TxtParser,
    ".md": MarkdownParser,
    ".markdown": MarkdownParser,
    ".pdf": PdfTextParser,
}


def parser_for_path(path: str | Path, *, limits: ParserLimits | None = None) -> _BaseParser:
    """按扩展名选择 parser；不识别的格式明确报错。"""

    suffix = Path(path).suffix.lower()
    parser_type = PARSER_BY_SUFFIX.get(suffix)
    if parser_type is None:
        raise ParserError(f"不支持的文档类型：{suffix or '<无扩展名>'}")
    return parser_type(limits=limits)
