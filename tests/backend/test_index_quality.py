"""M3.3 索引质量门禁：用轻量 tokenizer/model 替身验证真实规则。"""

import pytest

from index_quality import (
    IndexQualityError,
    analyze_embedding_inputs,
    assert_embedding_inputs,
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
