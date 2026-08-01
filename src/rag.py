"""检索 + 生成：从 PostgreSQL + pgvector 检索相关片段，调用本地 Ollama 生成回答。"""
import json
import re
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


def _mentions_novel(question: str, novel: str) -> bool:
    """判断问题里是否提到了这本书。

    库里的书名是文件名（如"《凡人修仙传》（校对版全本+番外）作者：忘语"），
    用户只会说"凡人修仙传"，所以取《》内的标题、去掉修饰后做包含匹配。
    """
    inner = re.findall(r"《([^》]+)》", novel)
    candidates = [name.strip() for name in inner if name.strip()]
    if not candidates:
        # 没有书名号就退化为用文件名主体（截断，避免整串带作者名匹配不上）
        candidates = [novel.split("（")[0].split("作者")[0].strip()]
    return any(name and name in question for name in candidates)


# 正文收尾的结构标记。网上流传的 txt 常是"全本+番外"，文件最末往往是番外或
# 作者后记，而不是正文结局，所以要靠这些标记定位真正的结局位置。
_ENDING_MARKERS = ("全书完", "（大结局）", "(大结局)", "大结局", "全文完", "尾声")


def _find_ending_anchor(conn, novel: str) -> int | None:
    """找出正文结局所在的片段编号；找不到标记时返回 None（调用方退回文件末尾）。

    取最后一个出现结束标记的片段：既能跳过目录里提前出现的"大结局"字样，
    也能避免把后面的番外/后记误当结局。
    """
    row = conn.execute(
        """
        SELECT MAX(chunk_id) AS anchor
        FROM novel_chunks
        WHERE novel = %s
          AND (text LIKE '%%全书完%%' OR text LIKE '%%大结局%%'
               OR text LIKE '%%全文完%%' OR text LIKE '%%尾声%%')
        """,
        (novel,),
    ).fetchone()
    anchor = row and row["anchor"]
    return int(anchor) if anchor is not None else None


def _dominant_novels(sources: list["SourceChunk"]) -> list[str]:
    """从已召回的片段里推断问题主要在问哪本书。

    取命中数最多的书；若有其他书命中数达到它的一半以上，则一并保留
    （问题可能确实跨书，例如"两本书的结局有什么不同"）。
    """
    if not sources:
        return []
    counts: dict[str, int] = {}
    for source in sources:
        counts[source.novel] = counts.get(source.novel, 0) + 1
    top = max(counts.values())
    return [novel for novel, n in counts.items() if n * 2 >= top]


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

    def positional_retrieve(
        self,
        question: str,
        top_k: int = TOP_K,
        hint_novels: list[str] | None = None,
    ) -> list[SourceChunk]:
        """按"书里的位置"召回，解决语义检索根本答不了的结构性问题。

        "结局是什么"这类问题，答案所在的原文里并不会出现"结局"二字，
        向量检索因此几乎必然失败。这里改为直接按 chunk_id 取书的首/尾片段。

        只在问题命中结构性词时生效。确定"哪本书"的优先级：
        1. 问题里直接写了书名 → 只取那本；
        2. 否则用 hint_novels（由语义/关键词召回推断出的书）→ 只取那些书。
           这样"韩立的结局"（只提人物不提书名）也能定位到《凡人修仙传》；
        3. 都没有 → 每本书各取一段。
        """
        text = question.strip()
        tail_words = ("结局", "结尾", "最后", "最终", "收尾", "大结局", "结束")
        head_words = ("开头", "开篇", "最初", "一开始", "起初", "开始时")
        at_tail = any(w in text for w in tail_words)
        at_head = any(w in text for w in head_words)
        if not (at_tail or at_head):
            return []

        with connect() as conn:
            novels = [
                row["novel"]
                for row in conn.execute("SELECT DISTINCT novel FROM novel_chunks").fetchall()
            ]
            # 问题里提到了某本书就只查那本，避免把别的书的结尾混进来
            matched = [n for n in novels if _mentions_novel(text, n)]
            if matched:
                targets = matched
            elif hint_novels:
                # 只保留确实存在于库里的提示书名
                targets = [n for n in novels if n in set(hint_novels)] or novels
            else:
                targets = novels

            # 每本书分配的配额：只查一本时全给它，多本时平摊但至少 1 段
            per_novel = max(1, top_k // len(targets)) if targets else top_k
            order = "DESC" if at_tail else "ASC"
            results: list[SourceChunk] = []
            for novel in targets:
                anchor = (
                    _find_ending_anchor(conn, novel) if at_tail else None
                )
                if anchor is not None:
                    # 有"大结局/全书完"标记：以它为终点向前取，避免把后面的
                    # 番外、后记当成结局（很多"全本+番外"的 txt 末尾都不是正文结局）。
                    rows = conn.execute(
                        """
                        SELECT novel, chunk_id, text
                        FROM novel_chunks
                        WHERE novel = %s AND chunk_id <= %s
                        ORDER BY chunk_id DESC
                        LIMIT %s
                        """,
                        (novel, anchor, per_novel),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        f"""
                        SELECT novel, chunk_id, text
                        FROM novel_chunks
                        WHERE novel = %s
                        ORDER BY chunk_id {order}
                        LIMIT %s
                        """,
                        (novel, per_novel),
                    ).fetchall()
                # 保持 SQL 的 ORDER BY 顺序（问结局时最末片段排最前），
                # 这个顺序会成为 RRF 的排名依据——排序错了最关键的片段就挤不进 top-k。
                for row in rows:
                    results.append(
                        SourceChunk(
                            novel=row["novel"],
                            chunk_id=int(row["chunk_id"]),
                            text=row["text"],
                            distance=0.0,
                        )
                    )
        return results

    def retrieve_hybrid(self, question: str, top_k: int = TOP_K) -> list[SourceChunk]:
        """统一的两阶段召回：候选池合并后用轻量 RRF 排序，最终取 top-k。"""
        candidate_k = max(top_k, RECALL_K)
        semantic_sources = self.retrieve(question, top_k=candidate_k)
        keyword_sources = self.keyword_retrieve(question, top_k=candidate_k)

        # 先用前两路召回推断问题属于哪本书，再据此做结构性召回。
        # 这样"韩立的结局"这类只提人物、不提书名的问题也能定位到正确的书。
        hint_novels = _dominant_novels(semantic_sources + keyword_sources)
        positional_sources = self.positional_retrieve(
            question, top_k=candidate_k, hint_novels=hint_novels
        )

        # Reciprocal Rank Fusion：两个召回来源都贡献分数，不需要额外的重排模型。
        # 对开放性问题，关键词通常为空，结果自然退化为纯语义检索。
        rrf_k = 60
        scores: dict[tuple[str, int], float] = {}
        items: dict[tuple[str, int], SourceChunk] = {}
        for ranked_sources in (semantic_sources, keyword_sources, positional_sources):
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
