from parsers import MarkdownParser
from v2_ingest import build_v2_index_input


class _Embedder:
    model_name = "test-embedding"

    def encode(self, texts, *, normalize_embeddings, show_progress_bar):
        assert normalize_embeddings is True
        assert show_progress_bar is False
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_build_v2_index_input_connects_parser_embedding_and_terms():
    item = build_v2_index_input(
        "# 概览\n\n这是一个可检索的 AI 文档。".encode(),
        title="产品说明.md",
        collection_name="项目资料",
        embedder=_Embedder(),
        parser=MarkdownParser(),
    )

    assert item.document.title == "产品说明.md"
    assert item.document.source_type == "markdown"
    assert len(item.chunks) == 1
    assert item.chunks[0].section_path == ("概览",)
    assert item.embeddings[item.chunks[0].id] == (0.1, 0.2, 0.3)
    assert item.terms[item.chunks[0].id]["检索"] == 1
    assert item.pipeline_hash
