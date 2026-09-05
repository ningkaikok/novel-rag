"""M3.3 索引质量门禁：用轻量 tokenizer/model 替身验证真实规则。"""

import pytest

from index_quality import (
    IndexQualityError,
    analyze_embedding_inputs,
    assert_embedding_inputs,
    fit_context_to_embedding_budget,
    make_quality_report,
    validate_embedding_vectors,
)
from loader import Chunk


class _Tokenizer:
    name_or_path = "test-tokenizer"
    model_max_length = 5

    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(len(text)))}


class _Model:
    tokenizer = _Tokenizer()
    max_seq_length = 5


def test_counts_real_tokens_and_rejects_overflow():
    info = analyze_embedding_inputs(_Model(), ["abcd", "abcdef"], kind="chunk")

    assert info["tokens"]["p50"] == 4
    assert info["overflow_count"] == 1
    with pytest.raises(IndexQualityError, match="超过 embedding 有效长度"):
        assert_embedding_inputs(_Model(), ["abcdef"], kind="chunk")


def test_report_keeps_empty_and_duplicate_as_expected_signal():
    chunks = [Chunk("示例", 0, "重复", None), Chunk("示例", 1, "重复", None)]
    report = make_quality_report(
        novel="示例",
        source_hash="source",
        source={"encoding": "utf-8", "fallback": False},
        chunks=chunks,
        model=_Model(),
        embedding_inputs={"chunk": ["重复", "重复"]},
        lineage={"quality_gate_version": 1},
    )

    assert report.passed
    assert report.chunks["duplicate_count"] == 1
    assert report.warnings


def test_vector_validation_catches_bad_dimension_and_non_finite_values():
    result = validate_embedding_vectors(
        [[0.1, 0.2], [float("nan")]], expected_dimension=2, kind="chunk"
    )

    assert result["wrong_dimension_count"] == 1
    assert result["non_finite_count"] == 1


def test_report_warns_about_replacement_and_control_characters():
    chunks = [Chunk("示例", 0, "正文�\x01", "第一章")]
    report = make_quality_report(
        novel="示例",
        source_hash="source",
        source={"encoding": "utf-8", "fallback": True},
        chunks=chunks,
        model=_Model(),
        embedding_inputs={"chunk": ["正文�\x01"]},
        lineage={"quality_gate_version": 1},
    )

    assert report.passed
    assert report.chunks["replacement_char_count"] == 1
    assert report.chunks["control_char_count"] == 1
    assert len(report.warnings) >= 2


# ------------------------------------------------------------------------
# fit_context_to_embedding_budget：Contextual Retrieval 的说明超长时，只压
# 说明，绝不动原文（假 tokenizer 一字一 token，max_seq_length=5，见上面的
# _Model/_Tokenizer）。


def test_no_compression_when_already_within_budget():
    """没超预算就什么都不该动——闸门不能变成常态截断。"""
    fitted, changed = fit_context_to_embedding_budget(_Model(), "ab", "cd")

    assert fitted == "ab"
    assert changed is False


def test_context_is_truncated_but_chunk_text_is_never_touched():
    """超预算时压缩的必须是说明，原文哪怕一个字都不能少。"""
    fitted, changed = fit_context_to_embedding_budget(_Model(), "abcd", "ef")

    assert changed is True
    assert fitted == "ab", "应保留能装下的最长说明前缀"
    assert len(f"{fitted}\nef") <= 5


def test_context_is_dropped_entirely_when_even_one_char_does_not_fit():
    """连一个字的说明都装不下时，压到空串——原文本身没有超预算，不该被牵连。"""
    fitted, changed = fit_context_to_embedding_budget(_Model(), "x", "abcde")

    assert fitted == ""
    assert changed is True


def test_oversized_chunk_text_is_handed_back_unchanged():
    """原文自己就超长时，压缩说明毫无意义——原样交还，让硬性门禁去拦截整本书，
    不能在这里悄悄放过一个真正超长的原文片段。"""
    fitted, changed = fit_context_to_embedding_budget(_Model(), "x", "abcdef")

    assert fitted == "x"
    assert changed is False
    with pytest.raises(IndexQualityError, match="超过 embedding 有效长度"):
        assert_embedding_inputs(_Model(), [f"{fitted}\nabcdef"], kind="chunk")


def test_empty_context_is_a_no_op():
    fitted, changed = fit_context_to_embedding_budget(_Model(), "", "abc")

    assert fitted == ""
    assert changed is False


def test_no_limit_available_is_a_no_op():
    """model_max_length 是代表"无限"的哨兵值时（≥100万），视为没有长度上限。"""

    class _NoLimitTokenizer(_Tokenizer):
        model_max_length = 10**9

    class _NoLimitModel:
        tokenizer = _NoLimitTokenizer()
        max_seq_length = None

    fitted, changed = fit_context_to_embedding_budget(_NoLimitModel(), "abcd", "ef")

    assert fitted == "abcd"
    assert changed is False
