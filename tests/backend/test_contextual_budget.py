"""Contextual Retrieval 说明拼接后的长度预算：只压说明，原文绝不截断。

用假 tokenizer（一字一 token）验证 ingest._build_indexed_texts：以前一旦
「说明 + 原文」拼接超过模型有效长度，整本书的索引会在质量门禁阶段直接失败
（不会写入半成品数据，这点本身没问题，但代价是一条说明超长就拖累整本书）。
现在超长时只压缩说明、压到空也没关系，原文片段本身在任何情况下都逐字不变。
"""

from contextualizer import text_hash
from ingest import _build_indexed_texts
from loader import Chunk


class _Tokenizer:
    name_or_path = "test-tokenizer"
    model_max_length = 5

    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(len(text)))}


class _Model:
    tokenizer = _Tokenizer()
    max_seq_length = 5


def _chunk(chunk_id: int, text: str) -> Chunk:
    return Chunk("示例", chunk_id, text, None)


def test_chunks_without_a_generated_context_are_untouched():
    chunks = [_chunk(0, "abc")]
    indexed_texts, truncated, dropped = _build_indexed_texts(chunks, {}, _Model())

    assert indexed_texts == ["abc"]
    assert (truncated, dropped) == (0, 0)


def test_context_within_budget_is_kept_in_full():
    chunks = [_chunk(0, "cd")]
    contexts = {text_hash("cd"): "ab"}

    indexed_texts, truncated, dropped = _build_indexed_texts(chunks, contexts, _Model())

    assert indexed_texts == ["ab\ncd"]
    assert (truncated, dropped) == (0, 0)


def test_oversized_context_is_truncated_and_original_text_survives_intact():
    chunks = [_chunk(0, "ef")]
    contexts = {text_hash("ef"): "abcd"}  # "abcd\nef" = 7 字符，超过预算 5

    indexed_texts, truncated, dropped = _build_indexed_texts(chunks, contexts, _Model())

    assert indexed_texts == ["ab\nef"]
    assert indexed_texts[0].endswith("ef"), "原文必须逐字保留在结尾，不能被截断"
    assert (truncated, dropped) == (1, 0)


def test_context_is_dropped_when_even_one_char_does_not_fit():
    chunks = [_chunk(0, "abcde")]  # 原文自身恰好等于预算 5
    contexts = {text_hash("abcde"): "x"}

    indexed_texts, truncated, dropped = _build_indexed_texts(chunks, contexts, _Model())

    assert indexed_texts == ["abcde"], "说明被彻底放弃后，索引文本应等于原始未增强的原文"
    assert (truncated, dropped) == (0, 1)


def test_oversized_raw_chunk_is_handed_back_for_the_hard_gate_to_catch():
    """原文本身就超长——压缩说明救不了，必须原样交还，交给下游硬性门禁处理。

    这不是本次修复要解决的问题：切分参数配得比模型窗口还大是另一类错误，
    仍然应该让整本书的索引失败，而不是被这里悄悄放过一个超长片段。
    """
    chunks = [_chunk(0, "abcdef")]  # 6 字符，已经超过预算 5
    contexts = {text_hash("abcdef"): "x"}

    indexed_texts, truncated, dropped = _build_indexed_texts(chunks, contexts, _Model())

    assert indexed_texts == ["x\nabcdef"], (
        "无法挽救时应保持原状，交由 assert_embedding_inputs 拦截"
    )
    assert (truncated, dropped) == (0, 0)


def test_mixed_batch_only_touches_the_chunks_that_actually_need_it():
    """一批片段里只有一条超长：其余片段（有说明的、没说明的）都不该被牵连。"""
    chunks = [_chunk(0, "cd"), _chunk(1, "ef"), _chunk(2, "gh")]
    contexts = {
        text_hash("cd"): "ab",  # 装得下，不用压缩
        text_hash("ef"): "abcd",  # 装不下，压缩说明
        # chunk 2 没有生成说明
    }

    indexed_texts, truncated, dropped = _build_indexed_texts(chunks, contexts, _Model())

    assert indexed_texts[0] == "ab\ncd"
    assert indexed_texts[1] == "ab\nef"
    assert indexed_texts[2] == "gh"
    assert (truncated, dropped) == (1, 0)
