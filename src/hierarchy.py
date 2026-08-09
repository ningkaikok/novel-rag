"""层级摘要的纯算法：把片段组织成章节节点和全书节点。

这里刻意不调用 LLM，也不连接数据库，原因有三点：

1. 初学者可以直接给 ``build_hierarchy_nodes`` 传假片段写单元测试；
2. 给几千章小说建索引不会产生云端费用，也不会因模型限流半途失败；
3. 摘要只做“导航索引”，回答证据仍回到原文，所以抽取式摘要已经能完成第一版闭环。

后续想实验 LLM 摘要，只需替换 ``extract_summary``，数据库表和检索链路无需重写。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from config import HIERARCHY_SUMMARY_MAX_CHARS, HIERARCHY_UNTITLED_CHUNKS


@dataclass(frozen=True)
class HierarchyNode:
    """一个可向量检索的层级节点。

    ``start_chunk_id``/``end_chunk_id`` 是回到原文证据的桥梁。摘要命中后，RAG
    不直接把 summary 当事实，而是从这个范围抽取原文候选再进入 RRF/重排。
    """

    novel: str
    level: str  # chapter / novel
    node_id: str
    title: str
    node_order: int
    start_chunk_id: int
    end_chunk_id: int
    summary: str


def hierarchy_pipeline_hash() -> str:
    """层级算法指纹；摘要规则变化时只重建摘要层，不重算全部片段向量。"""
    payload = {
        "version": 1,
        "strategy": "extractive-begin-middle-end",
        "summary_max_chars": HIERARCHY_SUMMARY_MAX_CHARS,
        "untitled_chunks": HIERARCHY_UNTITLED_CHUNKS,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _even_indices(size: int, count: int) -> list[int]:
    """在 ``0..size-1`` 上均匀取样，始终覆盖开头和结尾。"""
    if size <= count:
        return list(range(size))
    if count <= 1:
        return [0]
    return sorted({round(i * (size - 1) / (count - 1)) for i in range(count)})


def extract_summary(title: str, texts: list[str], max_chars: int | None = None) -> str:
    """从一组有序原文中抽取“开头 + 中间 + 结尾”，生成稳定的导航摘要。

    只取开头会让人物后期变化永远不可见；全部拼接又会超过 embedding 长度。
    均匀抽样让章节首尾转折都进入表示，同时保持结果可重复、容易评测。
    """
    limit = max_chars or HIERARCHY_SUMMARY_MAX_CHARS
    cleaned = [_compact(text) for text in texts if _compact(text)]
    if not cleaned:
        return title[:limit]
    # 最多取 5 个位置；每个位置共享字符预算。先保留标题，它对章节检索很有价值。
    chosen = [cleaned[i] for i in _even_indices(len(cleaned), 5)]
    prefix = f"{title}：" if title else ""
    per_piece = max(40, (limit - len(prefix) - 8) // max(len(chosen), 1))
    pieces = [text[:per_piece] for text in chosen]
    return (prefix + " … ".join(pieces))[:limit]


def _chapter_groups(chunks: list) -> list[tuple[str, list]]:
    """按连续章节标题分组；无标题内容再按窗口切成虚拟章节。"""
    ordered = sorted(chunks, key=lambda chunk: chunk.chunk_id)
    groups: list[tuple[str | None, list]] = []
    current_title: str | None = None
    current: list = []
    for chunk in ordered:
        title = chunk.chapter_title
        if current and title != current_title:
            groups.append((current_title, current))
            current = []
        current_title = title
        current.append(chunk)
    if current:
        groups.append((current_title, current))

    normalized: list[tuple[str, list]] = []
    for title, group in groups:
        if title:
            normalized.append((title, group))
            continue
        window = max(1, HIERARCHY_UNTITLED_CHUNKS)
        for start in range(0, len(group), window):
            part = group[start : start + window]
            normalized.append(
                (f"未命名章节（片段 {part[0].chunk_id}–{part[-1].chunk_id}）", part)
            )
    return normalized


def build_hierarchy_nodes(chunks: list) -> list[HierarchyNode]:
    """生成若干章节节点和一个全书节点；空输入返回空列表。"""
    if not chunks:
        return []
    novel = chunks[0].novel
    chapters: list[HierarchyNode] = []
    for order, (title, group) in enumerate(_chapter_groups(chunks)):
        start = group[0].chunk_id
        end = group[-1].chunk_id
        chapters.append(
            HierarchyNode(
                novel=novel,
                level="chapter",
                node_id=f"chapter:{order}:{start}-{end}",
                title=title,
                node_order=order,
                start_chunk_id=start,
                end_chunk_id=end,
                summary=extract_summary(title, [chunk.text for chunk in group]),
            )
        )

    # 全书节点不是把所有章节摘要硬拼起来，而是均匀采样。这样长篇小说的后半段
    # 仍有机会进入全书向量，同时严格控制 embedding 输入长度。
    sampled = [chapters[i] for i in _even_indices(len(chapters), 12)]
    book_summary = extract_summary(
        f"《{novel}》全书概览",
        [f"{node.title} {node.summary}" for node in sampled],
    )
    book = HierarchyNode(
        novel=novel,
        level="novel",
        node_id="novel",
        title=f"《{novel}》全书概览",
        node_order=0,
        start_chunk_id=min(chunk.chunk_id for chunk in chunks),
        end_chunk_id=max(chunk.chunk_id for chunk in chunks),
        summary=book_summary,
    )
    return [*chapters, book]


_GLOBAL_SIGNALS = (
    "全书",
    "整本",
    "整体",
    "主题",
    "主旨",
    "贯穿",
    "成长",
    "变化",
    "发展",
    "人物弧光",
    "比较",
    "对比",
    "异同",
    "共同点",
    "不同点",
)


def is_global_question(question: str) -> bool:
    """是否需要先看章节/全书层级，而不是只靠局部 top-k 碰运气。"""
    text = question.strip()
    return any(signal in text for signal in _GLOBAL_SIGNALS)
