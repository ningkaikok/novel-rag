import sys
from types import SimpleNamespace

import pytest

from domain_models import DocumentParser
from loader import load_novel_file
from parsers import (
    MarkdownParser,
    ParserError,
    ParserLimitError,
    ParserLimits,
    PdfTextParser,
    TxtParser,
    parser_for_path,
    source_ref_for_chunk,
)


def test_txt_parser_reuses_novel_cleaning_and_chunking_semantics(tmp_path):
    path = tmp_path / "演示.txt"
    payload = (
        "作者信息\n\n第一章 开始\n\n顾长风走进山庄。\n\n第二章 发现\n\n他找到了线索。".encode()
    )
    path.write_bytes(payload)

    legacy_chunks = load_novel_file(path)
    parsed_chunks = TxtParser().parse(payload, title="演示.txt")

    assert isinstance(TxtParser(), DocumentParser)
    assert [(item.text, item.section_path) for item in parsed_chunks] == [
        (item.text, (item.chapter_title,) if item.chapter_title else ())
        for item in legacy_chunks
    ]
    assert parsed_chunks[0].metadata["parser_name"] == "txt"
    assert parsed_chunks[0].metadata["parser_version"] == "1.0"
    ref = source_ref_for_chunk(parsed_chunks[1])
    assert ref.locator.kind == "heading"
    assert ref.locator.value == "第一章 开始"
    assert len(ref.excerpt) <= 80


def test_markdown_parser_preserves_nested_heading_path_and_locator():
    payload = b"# Project\n\nOverview\n\n## Design\n\nDetails\n\n### API\n\nContract"

    chunks = MarkdownParser().parse(payload, title="notes.md")

    assert [chunk.section_path for chunk in chunks] == [
        ("Project",),
        ("Project", "Design"),
        ("Project", "Design", "API"),
    ]
    assert chunks[1].text.startswith("## Design")
    assert source_ref_for_chunk(chunks[2]).locator.value == "Project/Design/API"


def test_pdf_parser_prefers_pdfplumber_and_keeps_page_numbers(monkeypatch):
    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakePdf:
        pages = [FakePage("第一页内容"), FakePage("第二页内容")]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake_pdfplumber = SimpleNamespace(open=lambda _stream: FakePdf())
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_pdfplumber)

    chunks = PdfTextParser().parse(b"%PDF-fake", title="report.pdf")

    assert [chunk.page_number for chunk in chunks] == [1, 2]
    assert all(chunk.metadata["pdf_backend"] == "pdfplumber" for chunk in chunks)
    assert source_ref_for_chunk(chunks[1]).locator.kind == "page"
    assert source_ref_for_chunk(chunks[1]).locator.value == "2"


def test_pdf_parser_rejects_page_limit_and_empty_text(monkeypatch):
    class FakePdf:
        def __init__(self, pages):
            self.pages = pages

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    page = SimpleNamespace(extract_text=lambda: "文本")
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda _stream: FakePdf([page, page])),
    )
    with pytest.raises(ParserLimitError, match="页数"):
        PdfTextParser(ParserLimits(max_pages=1)).parse(b"pdf", title="too-many.pdf")

    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(
            open=lambda _stream: FakePdf([SimpleNamespace(extract_text=lambda: "")])
        ),
    )
    with pytest.raises(ParserError, match="扫描型 PDF"):
        PdfTextParser().parse(b"pdf", title="scan.pdf")


def test_pdf_parser_falls_back_to_pypdf(monkeypatch):
    page = SimpleNamespace(extract_text=lambda: "pypdf 页面")
    fake_reader = SimpleNamespace(is_encrypted=False, pages=[page])
    monkeypatch.setitem(sys.modules, "pdfplumber", None)
    monkeypatch.setitem(
        sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda _stream: fake_reader)
    )

    chunks = PdfTextParser().parse(b"pdf", title="fallback.pdf")

    assert chunks[0].page_number == 1
    assert chunks[0].metadata["pdf_backend"] == "pypdf"


def test_parsers_enforce_input_size_and_extension():
    with pytest.raises(ParserLimitError, match="bytes"):
        TxtParser(ParserLimits(max_bytes=2)).parse(b"too long", title="x.txt")
    with pytest.raises(ParserError, match="不支持"):
        parser_for_path("notes.docx")
