from rag import NovelRAG, SourceChunk


def test_prompt_numbers_sources_and_includes_chapter_metadata():
    rag = NovelRAG.__new__(NovelRAG)
    rag._graph_hint = lambda question: ""
    sources = [
        SourceChunk(
            novel="雾隐山庄",
            chunk_id=7,
            chapter_title="第二章 蚀骨奇毒",
            text="顾长风中了蚀骨散。",
            distance=0.1,
        ),
        SourceChunk(
            novel="雾隐山庄",
            chunk_id=8,
            text="沈砚之取出银针。",
            distance=0.2,
        ),
    ]

    prompt = rag.build_prompt("顾长风得了什么病？", sources)

    assert "[1] 《雾隐山庄》 · 第二章 蚀骨奇毒 · 片段 #7" in prompt
    assert "[2] 《雾隐山庄》 · 片段 #8" in prompt
    assert "只能使用下面真实存在的编号" in prompt
    assert "顾长风得了什么病？" in prompt
