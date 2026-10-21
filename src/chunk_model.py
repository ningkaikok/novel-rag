"""检索结果的数据类与候选 trace 记录。

初学者可以把这里看成 RAG 的“统一数据语言”：无论哪一路召回（向量、BM25、
结构性、层级摘要），最终都折算成 ``SourceChunk`` 才能进入融合和重排；

    novel        书名（文件名式全名，用于定位数据库行）
    chunk_id     片段编号，和 novel 一起构成主键
    text         原文片段——唯一允许作为“证据”交给模型的内容
    distance     各路分数的统一出口：约定“越小越好”（BM25 分数取负号存进来）
    chapter_title / context   可选的增强元数据，见字段注释

``_trace_candidates`` 则负责把内部候选压成轻量的排名记录，供前端“思考过程”
展示和评测复盘使用，避免把大段原文塞进 SSE/JSONB。
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain_models import DocumentChunk, SourceRef

if TYPE_CHECKING:
    from v2_retrieval import V2SearchHit


@dataclass
class SourceChunk:
    novel: str
    chunk_id: int
    text: str
    distance: float
    # 章节识别是增强元数据：旧索引和无规范标题的 txt 都可能为空。
    chapter_title: str | None = None
    # Contextual Retrieval 生成的上下文说明（没做增强时是空串）。
    # 重排要用它（见 reranker.rerank 里 indexed_text 的说明），
    # 但 build_prompt 只用 text——不把 AI 生成的说明当原文依据给模型。
    context: str = ""
    # 候选来自 V1 小说索引还是 V2 通用文档；前端/序列化据此调整展示，
    # 默认值保证所有既有构造点（from_legacy_row 等）行为不变。
    origin: str = "legacy_novel"
    # V2 命中的稳定内部标识（document_id / chunk_id 都是 V2 的 UUID）。
    # novel/chunk_id 只是“展示用投影”——文档标题可能重复、ordinal 只在一份
    # 文档内唯一，拿它们当去重键会让两份同名文档互相吞掉候选。这两个字段
    # 只在 origin="v2_document" 时非空，且不进 HTTP 契约（不参与序列化）。
    document_id: str | None = None
    chunk_uid: str | None = None

    @property
    def is_legacy_novel(self) -> bool:
        """是否是 V1 小说片段。

        只有 V1 小说能按 ``chunk_id`` 回 ``novel_chunks`` 补邻居/整章正文；V2 文档
        片段（``origin="v2_document"``）的 novel/chunk_id 只是展示投影，回那张表
        查到的是另一套 ID 空间的同名小说，绝不能拿它的正文替换 V2 命中。
        """

        return self.origin == "legacy_novel"

    @property
    def identity_key(self) -> tuple:
        """跨召回路径唯一的来源身份键，用于去重/融合/证据注册。

        V1 小说和 V2 文档是两套完全独立的 ID 空间（``novel_chunks`` 主键 vs
        ``knowledge_v2`` 的 UUID），只用 ``(novel, chunk_id)`` 会撞车：一份叫
        ``雾隐山庄.txt`` 的上传文档，ordinal 0 和 V1 小说第 0 段会被当成同一段
        证据，融合时互相覆盖正文。所以身份键里必须带上来源域。

        V2 侧优先用 ``document_id + chunk_uid``；缺这两个字段的旧构造点（测试
        里手工拼的 SourceChunk、shadow 观测等）退回展示字段，保证不影响既有
        行为。
        """

        if self.origin == "v2_document":
            if self.document_id is not None and self.chunk_uid is not None:
                return ("v2_document", self.document_id, self.chunk_uid)
            return ("v2_document", self.novel, self.chunk_id)
        return ("legacy_novel", self.novel, self.chunk_id)

    @classmethod
    def from_v2_hit(cls, hit: "V2SearchHit", *, distance: float = 0.0) -> "SourceChunk":
        """从 V2 原生文档候选（非小说）投影成 SourceChunk，复用融合/重排/引用链路。

        ``novel``/``chunk_id`` 只是展示投影（文档标题 + 片段序号），真正用于
        去重和证据注册的是 ``identity_key``——它落在 V2 的 ``document_id`` 和
        全局唯一的 chunk id 上，因此两份同名文档、或一份文档与一本同名小说
        都不会再被合并成同一条证据。
        """

        chunk = hit.chunk
        if chunk.section_path:
            chapter_title: str | None = " › ".join(chunk.section_path)
        elif chunk.page_number is not None:
            chapter_title = f"第{chunk.page_number}页"
        else:
            chapter_title = None
        return cls(
            novel=hit.document.title,
            chunk_id=chunk.ordinal,
            text=chunk.text,
            distance=distance,
            chapter_title=chapter_title,
            context=chunk.context or "",
            origin="v2_document",
            document_id=hit.document.id,
            chunk_uid=chunk.id,
        )

    @classmethod
    def from_legacy_row(cls, row: dict, *, distance: float = 0.0) -> "SourceChunk":
        """从旧 ``novel_chunks`` 行经适配器恢复旧结果对象。

        查询层仍可继续使用 ``novel/chunk_id``，但小说专有字段的映射集中在
        ``LegacyNovelAdapter``，为后续 V2 repository 留出唯一接缝。
        """

        from legacy_novel import LegacyNovelAdapter

        domain_chunk = LegacyNovelAdapter.chunk_from_row(row)
        metadata = domain_chunk.metadata
        return cls(
            novel=str(metadata["legacy_novel"]),
            chunk_id=int(str(metadata["legacy_chunk_id"])),
            text=domain_chunk.text,
            distance=distance,
            chapter_title=(
                str(metadata["legacy_chapter_title"])
                if "legacy_chapter_title" in metadata
                else None
            ),
            context=domain_chunk.context,
        )

    @property
    def indexed_text(self) -> str:
        """建索引时用的文本，也是重排该看到的文本。

        必须和索引保持一致：索引的是「说明 + 原文」，重排如果只看原文，
        就会把上下文增强的效果整个抵消掉。
        """
        return f"{self.context}\n{self.text}" if self.context else self.text

    def to_document_chunk(self) -> DocumentChunk:
        """通过小说适配器暴露通用片段视图，不改变旧检索结果字段。"""

        from legacy_novel import LegacyNovelAdapter

        return LegacyNovelAdapter.chunk(self)

    def to_source_ref(self) -> SourceRef:
        """返回不含完整正文的通用引用，供后续 API/评测逐步迁移。"""

        from legacy_novel import LegacyNovelAdapter

        return LegacyNovelAdapter.source_ref(self)


_TRACE_CANDIDATE_LIMIT = 10


def _trace_candidates(
    sources: list[SourceChunk],
    *,
    score_label: str,
    score_of,
    previous_ranks: dict[tuple, int] | None = None,
    selected_count: int = 0,
) -> list[dict]:
    """把内部候选压成适合 SSE/JSONB 的轻量排名记录，避免保存大段原文。"""
    payload: list[dict] = []
    for rank, source in enumerate(sources[:_TRACE_CANDIDATE_LIMIT], start=1):
        key = source.identity_key
        score = score_of(source, key)
        payload.append(
            {
                "novel": source.novel,
                "chunk_id": source.chunk_id,
                "chapter_title": source.chapter_title,
                "rank": rank,
                "score": round(float(score), 6) if score is not None else None,
                "score_label": score_label,
                "previous_rank": (previous_ranks or {}).get(key),
                "selected": bool(selected_count and rank <= selected_count),
                "origin": source.origin,
            }
        )
    return payload
