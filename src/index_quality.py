"""索引质量检查与数据血缘。

M3.3 的关键原则是：质量检查必须针对真正交给 embedding 模型的文本。
``CHUNK_SIZE`` 是字符上限，``token_count`` 是 BM25 的分词数量，两者都不能
代表模型会收到多少 token。本模块使用模型自己的 tokenizer，在 ``encode`` 之前
发现会被截断的输入，并提供不包含原文的结构化报告。

这里不连接数据库，也不依赖 FastAPI，方便初学者单独运行、测试和扩展评测规则。
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


QUALITY_GATE_VERSION = 1


class IndexQualityError(RuntimeError):
    """索引输入或 embedding 结果不满足硬性质量门禁。"""

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report or {}


@dataclass
class QualityReport:
    """一本书的质量报告。

    报告只保存哈希、计数、分布和配置，不保存小说原文，适合写入日志或 JSON 文件。
    """

    novel: str
    source_hash: str
    source: dict[str, Any]
    chunks: dict[str, Any]
    embedding: dict[str, Any]
    lineage: dict[str, Any]
    # 分级原则：errors 是「数据已经不可信，入库必然污染检索」的硬性错误
    # （空片段、输入会被静默截断、tokenizer 缺失导致长度检查失效），必须阻断；
    # warnings 只是质量信号（章节识别率低、疑似乱码、精确重复），可能来自
    # 合理的 overlap 或源文件本身，报告给人判断，不替人做决定。
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "novel": self.novel,
            "source_hash": self.source_hash,
            "source": self.source,
            "chunks": self.chunks,
            "embedding": self.embedding,
            "lineage": self.lineage,
            "warnings": self.warnings,
            "errors": self.errors,
            "passed": self.passed,
        }


def _percentile(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return int(ordered[index])


def _summary(values: list[int]) -> dict[str, int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def _tokenizer(model: Any) -> Any | None:
    return getattr(model, "tokenizer", None)


def embedding_model_metadata(model: Any) -> dict[str, Any]:
    """读取模型实际使用的 tokenizer 名称和有效长度。

    SentenceTransformer 的 ``max_seq_length`` 优先于 tokenizer 自己的默认值；
    后者有时是一个代表“无限”的巨大哨兵值。
    """
    tokenizer = _tokenizer(model)
    max_length = getattr(model, "max_seq_length", None)
    if max_length is None and tokenizer is not None:
        max_length = getattr(tokenizer, "model_max_length", None)
    try:
        max_length = int(max_length) if max_length is not None else None
    except (TypeError, ValueError):
        max_length = None
    if max_length is not None and max_length >= 1_000_000:
        max_length = None
    name = (
        getattr(tokenizer, "name_or_path", None)
        or getattr(tokenizer, "__class__", type(tokenizer)).__name__
        if tokenizer is not None
        else None
    )
    return {
        "model_class": model.__class__.__name__,
        "tokenizer": name,
        "max_seq_length": max_length,
        "available": tokenizer is not None and max_length is not None,
    }


def _token_count(model: Any, text: str) -> int | None:
    """用模型 tokenizer 计算未截断输入的 token 数。

    ``truncation=False`` 很重要：SentenceTransformer 的 encode 默认会截断，
    我们必须在它截断之前知道输入是否越界。
    """
    tokenizer = _tokenizer(model)
    if tokenizer is None:
        return None
    try:
        encoded = tokenizer(
            text,
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
        ids = encoded["input_ids"]
        # 单文本 tokenizer 通常返回 list[int]；某些替身返回 list[list[int]]。
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return len(ids)
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def analyze_embedding_inputs(
    model: Any, texts: list[str], *, kind: str
) -> dict[str, Any]:
    """分析一批最终 embedding 输入，返回 token 分布和超长信息。"""
    metadata = embedding_model_metadata(model)
    counts = [_token_count(model, text) for text in texts]
    available_counts = [count for count in counts if count is not None]
    limit = metadata["max_seq_length"]
    overflow = (
        [index for index, count in enumerate(counts) if count is not None and limit and count > limit]
        if limit
        else []
    )
    return {
        "kind": kind,
        "text_count": len(texts),
        "tokenizer_available": metadata["available"],
        "token_count_unavailable": len(counts) - len(available_counts),
        "tokens": _summary(available_counts),
        "max_seq_length": limit,
        "overflow_count": len(overflow),
        "overflow_indices": overflow[:20],
    }


def assert_embedding_inputs(model: Any, texts: list[str], *, kind: str) -> dict[str, Any]:
    """在调用 ``model.encode`` 前执行长度门禁并返回分析结果。"""
    info = analyze_embedding_inputs(model, texts, kind=kind)
    errors: list[str] = []
    if not info["tokenizer_available"]:
        errors.append(f"{kind} 输入无法读取 embedding tokenizer")
    if info["overflow_count"]:
        errors.append(
            f"{kind} 有 {info['overflow_count']} 个输入超过 embedding 有效长度 "
            f"{info['max_seq_length']}"
        )
    if errors:
        raise IndexQualityError(
            f"索引输入质量门禁失败：{'；'.join(errors)}",
            {"kind": kind, "embedding": info, "errors": errors, "passed": False},
        )
    return info


def fit_text_to_embedding_limit(model: Any, text: str) -> tuple[str, bool]:
    """把导航摘要显式压到模型上限内，返回 ``(文本, 是否发生压缩)``。

    这不是依赖模型 ``encode`` 的静默截断：压缩后的文本会写回摘要表，并在质量报告
    中计数。原文片段不会走这条路径，片段超长仍然直接阻断索引。
    """
    info = embedding_model_metadata(model)
    limit = info["max_seq_length"]
    if not text or not limit:
        return text, False
    count = _token_count(model, text)
    if count is None:
        raise IndexQualityError("无法用真实 tokenizer 检查摘要长度")
    if count <= limit:
        return text, False
    low, high = 1, len(text)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = text[:middle]
        candidate_count = _token_count(model, candidate)
        if candidate_count is not None and candidate_count <= limit:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    if not best:
        raise IndexQualityError("层级摘要无法压缩到 embedding 有效长度")
    return best, True


def validate_embedding_vectors(
    embeddings: Any, *, expected_dimension: int, kind: str
) -> dict[str, Any]:
    """检查向量维度和有限数值，避免坏向量进入 pgvector。"""
    count = len(embeddings)
    dimensions: Counter[int] = Counter()
    non_finite = 0
    for vector in embeddings:
        try:
            dimension = len(vector)
            dimensions[dimension] += 1
            if any(not math.isfinite(float(value)) for value in vector):
                non_finite += 1
        except (TypeError, ValueError):
            non_finite += 1
    wrong_dimension = sum(
        amount for dimension, amount in dimensions.items() if dimension != expected_dimension
    )
    return {
        "kind": kind,
        "vector_count": count,
        "expected_dimension": expected_dimension,
        "dimensions": dict(dimensions),
        "wrong_dimension_count": wrong_dimension,
        "non_finite_count": non_finite,
    }


def chunk_statistics(chunks: list[Any]) -> tuple[dict[str, Any], list[str]]:
    """统计片段质量；重复只报告不阻断，避免误伤合理 overlap。"""
    texts = [str(getattr(chunk, "text", "")) for chunk in chunks]
    hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    duplicate_count = sum(amount - 1 for amount in Counter(hashes).values() if amount > 1)
    chapter_count = sum(1 for chunk in chunks if getattr(chunk, "chapter_title", None))
    char_lengths = [len(text) for text in texts]
    replacement_char_count = sum(text.count("�") for text in texts)
    control_char_count = sum(
        sum(1 for char in text if ord(char) < 32 and char not in "\n\r\t")
        for text in texts
    )
    warnings: list[str] = []
    if duplicate_count:
        warnings.append(f"发现 {duplicate_count} 个精确重复片段（请结合 overlap 判断是否合理）")
    if texts and chapter_count == 0:
        warnings.append("没有识别到章节标题；已使用无标题片段窗口，不视为硬错误")
    if replacement_char_count:
        warnings.append(f"片段中发现 {replacement_char_count} 个 Unicode 替换字符，疑似乱码")
    if control_char_count:
        warnings.append(f"片段中发现 {control_char_count} 个控制字符，请检查源文件")
    return (
        {
            "count": len(texts),
            "empty_count": sum(not text.strip() for text in texts),
            "duplicate_count": duplicate_count,
            "chapter_chunk_count": chapter_count,
            "chapter_coverage": chapter_count / len(texts) if texts else 0.0,
            "replacement_char_count": replacement_char_count,
            "control_char_count": control_char_count,
            "characters": _summary(char_lengths),
        },
        warnings,
    )


def make_quality_report(
    *,
    novel: str,
    source_hash: str,
    source: dict[str, Any],
    chunks: list[Any],
    model: Any,
    embedding_inputs: dict[str, list[str]],
    lineage: dict[str, Any],
) -> QualityReport:
    """创建报告并执行所有不依赖向量值的硬性检查。"""
    chunk_info, warnings = chunk_statistics(chunks)
    embedding_info = {
        kind: analyze_embedding_inputs(model, texts, kind=kind)
        for kind, texts in embedding_inputs.items()
    }
    errors: list[str] = []
    if not chunks:
        errors.append("没有可索引的片段")
    if chunk_info["empty_count"]:
        errors.append(f"存在 {chunk_info['empty_count']} 个空片段")
    for kind, info in embedding_info.items():
        if info["text_count"] and not info["tokenizer_available"]:
            errors.append(f"{kind} 输入无法读取 embedding tokenizer，拒绝静默跳过长度检查")
        if info["overflow_count"]:
            errors.append(
                f"{kind} 有 {info['overflow_count']} 个输入超过 embedding 有效长度 "
                f"{info['max_seq_length']}"
            )
    if source.get("fallback"):
        warnings.append(
            "源文件未能用 UTF-8/GB18030 严格解码，已使用忽略错误字节的兜底路径"
        )
    return QualityReport(
        novel=novel,
        source_hash=source_hash,
        source=source,
        chunks=chunk_info,
        embedding=embedding_info,
        lineage=lineage,
        warnings=warnings,
        errors=errors,
    )


def assert_quality_report(report: QualityReport | dict[str, Any]) -> None:
    """在原子写入前阻断硬性质量错误。"""
    errors = report.errors if isinstance(report, QualityReport) else report.get("errors", [])
    if errors:
        novel = report.novel if isinstance(report, QualityReport) else report.get("novel", "")
        raise IndexQualityError(f"《{novel}》索引质量门禁失败：" + "；".join(errors), report.as_dict() if isinstance(report, QualityReport) else report)
