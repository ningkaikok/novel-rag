"""文本切分与章节元数据测试。"""

from loader import is_chapter_heading, load_novel_chunks


def test_recognizes_common_chapter_headings():
    assert is_chapter_heading("第一章 山边小村")
    assert is_chapter_heading("第102章：新的旅程")
    assert is_chapter_heading("楔子")
    assert is_chapter_heading("番外篇 南宫婉")


def test_does_not_treat_normal_sentence_as_heading():
    assert not is_chapter_heading("第一章的内容主要介绍了山庄的来历。")
    assert not is_chapter_heading("他翻到第一章，看见了一行小字。")


def test_chunks_keep_chapter_and_do_not_cross_boundaries(tmp_path):
    novel = tmp_path / "示例.txt"
    novel.write_text(
        "作者简介\n\n第一章 初遇\n\n顾长风来到山庄。\n\n"
        "第二章 真相\n\n沈砚之找到了蚀骨散的解药。",
        encoding="utf-8",
    )

    chunks = load_novel_chunks(tmp_path)

    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert chunks[0].chapter_title is None
    first = next(c for c in chunks if c.chapter_title == "第一章 初遇")
    second = next(c for c in chunks if c.chapter_title == "第二章 真相")
    assert "第一章 初遇" in first.text and "顾长风" in first.text
    assert "第二章 真相" not in first.text
    assert "第二章 真相" in second.text and "蚀骨散" in second.text
    assert "第一章 初遇" not in second.text
