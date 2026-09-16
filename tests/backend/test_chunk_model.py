"""SourceChunk 及其 V2 投影的单元测试。"""

from chunk_model import SourceChunk
from domain_models import Collection, Document, DocumentChunk, DocumentVersion
from v2_retrieval import V2SearchHit


def _hit(*, section_path=(), page_number=None, context="") -> V2SearchHit:
    return V2SearchHit(
        collection=Collection(id="c-1", name="资料库"),
        document=Document(
            id="d-1", collection_id="c-1", title="产品需求文档.md", source_type="markdown"
        ),
        version=DocumentVersion(
            id="v-1", document_id="d-1", version_no=1, parser_name="markdown", parser_version="1"
        ),
        chunk=DocumentChunk(
            id="ch-1",
            document_version_id="v-1",
            ordinal=2,
            text="V2 原文内容",
            section_path=section_path,
            page_number=page_number,
            context=context,
        ),
        distance=0.12,
    )


def test_from_v2_hit_maps_document_and_chunk_identity():
    source = SourceChunk.from_v2_hit(_hit(), distance=0.12)

    assert source.novel == "产品需求文档.md"
    assert source.chunk_id == 2
    assert source.text == "V2 原文内容"
    assert source.distance == 0.12
    assert source.origin == "v2_document"


def test_from_v2_hit_chapter_title_prefers_section_path():
    source = SourceChunk.from_v2_hit(_hit(section_path=("需求背景", "目标用户"), page_number=3))

    assert source.chapter_title == "需求背景 › 目标用户"


def test_from_v2_hit_chapter_title_falls_back_to_page_number():
    source = SourceChunk.from_v2_hit(_hit(page_number=5))

    assert source.chapter_title == "第5页"


def test_from_v2_hit_chapter_title_none_without_section_or_page():
    source = SourceChunk.from_v2_hit(_hit())

    assert source.chapter_title is None


def test_from_v2_hit_keeps_context_for_rerank_indexed_text():
    source = SourceChunk.from_v2_hit(_hit(context="这是概览说明"))

    assert source.context == "这是概览说明"
    assert source.indexed_text == "这是概览说明\nV2 原文内容"


def test_legacy_row_origin_defaults_to_legacy_novel():
    """default 值不应该影响任何既有构造点——旧 V1 结果必须继续是 legacy_novel。"""
    row = {
        "novel": "雾隐山庄",
        "chunk_id": 0,
        "chapter_title": "第一章",
        "text": "原文",
        "context": "",
    }

    source = SourceChunk.from_legacy_row(row, distance=0.1)

    assert source.origin == "legacy_novel"
