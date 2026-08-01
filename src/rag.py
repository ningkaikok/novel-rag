"""检索 + 生成：从 PostgreSQL + pgvector 检索相关片段，调用本地 Ollama 生成回答。"""
import json
from collections.abc import Iterator
from dataclasses import dataclass

import requests
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    CONTEXT_NEIGHBORS,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    RECALL_K,
    TOP_K,
)
from postgres import connect, has_index, vector_literal

PROMPT_TEMPLATE = """你是一个小说问答助手。请仅根据下面提供的原文片段回答问题。
如果片段中没有足够信息回答，请明确说“根据提供的片段无法确定”，不要编造内容。

原文片段：
{context}

问题：{question}

回答："""


@dataclass
class SourceChunk:
    novel: str
    chunk_id: int
    text: str
    distance: float


class NovelRAG:
    def __init__(self, embedder: SentenceTransformer | None = None):
        self.embedder = embedder or SentenceTransformer(EMBEDDING_MODEL)
        if not has_index():
            raise RuntimeError("PostgreSQL novel_chunks 表不存在，请先重建索引")

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[SourceChunk]:
        query_embedding = self.embedder.encode([question], normalize_embeddings=True)
        query_vector = vector_literal(query_embedding[0])
        with connect() as conn:
            rows = conn.execute(
                """
                SELECT novel, chunk_id, text,
                       embedding <=> %s::vector AS distance
                FROM novel_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vector, query_vector, top_k),
            ).fetchall()
        return [
            SourceChunk(
                novel=row["novel"],
                chunk_id=int(row["chunk_id"]),
                text=row["text"],
                distance=float(row["distance"]),
            )
            for row in rows
        ]

    def keyword_retrieve(self, question: str, top_k: int = TOP_K) -> list[SourceChunk]:
        """先用原文关键词查找，确保人物名、专有名词和原句不会被向量检索漏掉。"""
        needle = question.strip().casefold()
        if not needle:
            return []
        with connect() as conn:
            rows = conn.execute(
                """
                SELECT novel, chunk_id, text
                FROM novel_chunks
                WHERE position(lower(%s) in lower(text)) > 0
                ORDER BY novel, chunk_id
                LIMIT %s
                """,
                (needle, top_k),
            ).fetchall()
        return [
            SourceChunk(
                novel=row["novel"],
                chunk_id=int(row["chunk_id"]),
                text=row["text"],
                distance=0.0,
            )
            for row in rows
        ]

    def retrieve_hybrid(self, question: str, top_k: int = TOP_K) -> list[SourceChunk]:
        """统一的两阶段召回：候选池合并后用轻量 RRF 排序，最终取 top-k。"""
        candidate_k = max(top_k, RECALL_K)
        semantic_sources = self.retrieve(question, top_k=candidate_k)
        keyword_sources = self.keyword_retrieve(question, top_k=candidate_k)

        # Reciprocal Rank Fusion：两个召回来源都贡献分数，不需要额外的重排模型。
        # 对开放性问题，关键词通常为空，结果自然退化为纯语义检索。
        rrf_k = 60
        scores: dict[tuple[str, int], float] = {}
        items: dict[tuple[str, int], SourceChunk] = {}
        for ranked_sources in (semantic_sources, keyword_sources):
            for rank, source in enumerate(ranked_sources, start=1):
                key = (source.novel, source.chunk_id)
                scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + rank)
                items.setdefault(key, source)

        ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
        return [items[key] for key in ranked_keys[:top_k]]

    def expand_neighbors(
        self,
        sources: list[SourceChunk],
        neighbors: int = CONTEXT_NEIGHBORS,
    ) -> list[SourceChunk]:
        """为命中的片段补齐同一本书前后的相邻片段。

        检索结果仍保留为 top-k，扩展结果只用于生成上下文，避免前端出处卡片
        一次展示大量重复内容。相邻片段通过 PostgreSQL 的书名和片段编号读取。
        """
        if not sources or neighbors <= 0:
            return sources

        ranges: list[tuple[str, int, int]] = []
        for source in sources:
            ranges.append(
                (
                    source.novel,
                    max(0, source.chunk_id - neighbors),
                    source.chunk_id + neighbors,
                )
            )

        conditions = " OR ".join(
            "(novel = %s AND chunk_id BETWEEN %s AND %s)" for _ in ranges
        )
        params = [value for item in ranges for value in item]
        with connect() as conn:
            rows = conn.execute(
                f"SELECT novel, chunk_id, text FROM novel_chunks WHERE {conditions}",
                params,
            ).fetchall()
        by_key: dict[tuple[str, int], SourceChunk] = {}
        for row in rows:
            key = (row["novel"], int(row["chunk_id"]))
            by_key[key] = SourceChunk(
                novel=row["novel"],
                chunk_id=int(row["chunk_id"]),
                text=row["text"],
                distance=0.0,
            )

        expanded: list[SourceChunk] = []
        seen: set[tuple[str, int]] = set()
        # 按检索相关性保留不同命中簇的顺序；每个命中簇内部按原文顺序排列。
        for source in sources:
            group = [
                by_key[(source.novel, chunk_id)]
                for chunk_id in range(
                    max(0, source.chunk_id - neighbors), source.chunk_id + neighbors + 1
                )
                if (source.novel, chunk_id) in by_key
            ]
            for item in group:
                key = (item.novel, item.chunk_id)
                if key not in seen:
                    expanded.append(item)
                    seen.add(key)
        return expanded or sources

    def build_prompt(self, question: str, sources: list[SourceChunk]) -> str:
        """拼装检索片段 + 问题成完整 prompt。Ollama 和其他生成后端（如 Claude CLI）共用。"""
        context = "\n\n---\n\n".join(
            f"[{s.novel} #{s.chunk_id}]\n{s.text}" for s in sources
        )
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def generate(
        self, question: str, sources: list[SourceChunk], model: str = OLLAMA_MODEL
    ) -> str:
        prompt = self.build_prompt(question, sources)
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    def generate_stream(
        self, question: str, sources: list[SourceChunk], model: str = OLLAMA_MODEL
    ) -> Iterator[str]:
        """逐字（token）流式返回回答，供界面实时展示。model 可按次调用覆盖，便于前端切换模型。"""
        prompt = self.build_prompt(question, sources)
        with requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line).get("response", "")
                if chunk:
                    yield chunk

    def query(
        self, question: str, top_k: int = TOP_K, model: str = OLLAMA_MODEL
    ) -> tuple[str, list[SourceChunk]]:
        sources = self.retrieve(question, top_k)
        answer = self.generate(question, sources, model=model)
        return answer, sources
