"""NovelRAG 的单路检索 Mixin：向量、BM25、结构性与邻居扩展。

初学者可以把这里看成 RAG 的“召回层”：每个方法只负责一种召回路径，返回各自
排好序的 ``SourceChunk`` 列表，互相之间不感知——

    retrieve            向量语义检索（pgvector 余弦距离）
    keyword_retrieve    BM25 关键词检索（逐词精确匹配 + 相关性打分）
    positional_retrieve 结构性召回（按 chunk_id 直接取书的首/尾片段）
    expand_neighbors    为命中片段补齐前后相邻片段，供上下文组装使用

多路召回怎么融合、重排、串成流水线，见 ``rag.retrieve_hybrid_stream``；
书名/意图识别的纯函数在 ``novel_match``，数据类在 ``chunk_model``。
"""
from config import (
    BM25_B,
    BM25_K1,
    CONTEXT_NEIGHBORS,
    TOP_K,
)
from postgres import (
    connect,
    vector_literal,
)
from tokenizer import query_terms
from chunk_model import SourceChunk
from novel_match import _find_ending_anchor, _mentions_novel


class RetrievalMixin:
    def retrieve(
        self,
        question: str,
        top_k: int = TOP_K,
        only_novels: list[str] | None = None,
    ) -> list[SourceChunk]:
        """向量检索。only_novels 非空时把搜索范围限定在这些书内。"""
        query_embedding = self.embedder.encode([question], normalize_embeddings=True)
        query_vector = vector_literal(query_embedding[0])
        scope = "WHERE novel = ANY(%s)" if only_novels else ""
        params: list = [query_vector]
        if only_novels:
            params.append(only_novels)
        params.extend([query_vector, top_k])
        with connect() as conn:
            rows = conn.execute(
                f"""
                SELECT novel, chunk_id, chapter_title, text, context,
                       embedding <=> %s::vector AS distance
                FROM novel_chunks
                {scope}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                params,
            ).fetchall()
        return [
            SourceChunk(
                novel=row["novel"],
                chunk_id=int(row["chunk_id"]),
                text=row["text"],
                distance=float(row["distance"]),
                chapter_title=row.get("chapter_title"),
                context=row.get("context") or "",
            )
            for row in rows
        ]

    def keyword_retrieve(
        self,
        question: str,
        top_k: int = TOP_K,
        only_novels: list[str] | None = None,
    ) -> list[SourceChunk]:
        """BM25 关键词检索：按词精确匹配，并按相关性打分排序。

        为什么需要它（向量检索补不上的洞）
        ----------------------------------
        向量检索靠语义相似度，对"必须逐字匹配"的东西不可靠——人名、功法名、
        专有名词这些，语义上"韩铸"和"韩立"极其接近，但它们是两个人。
        BM25 走的是完全不同的路子：按词精确匹配，谁都不会把「韩铸」匹配成「韩立」。

        BM25 公式（这段是本函数的核心，SQL 里逐项对应）
        ------------------------------------------------
            score(D, Q) = Σ  IDF(t) · ────────tf · (k1 + 1)────────
                         t∈Q            tf + k1 · (1 - b + b · |D|/avgdl)

        三个部分各自解决一个问题：

        1. **IDF(t) = ln((N - df + 0.5) / (df + 0.5) + 1)** —— 词的区分度
           df 是"这个词出现在多少个片段里"。「韩立」出现在 15536 个片段里，
           df 极大 → IDF 极小 → 权重被自动压到接近 0；「窝头」只出现 1 次，
           df=1 → IDF 很大 → 命中它的片段分数被大幅拉高。

           这一项直接取代了改造前那个手写的 KEYWORD_GENERIC_LIMIT 启发式
           （"命中超过 300 个片段的词就整个丢掉"）。IDF 做的是同一件事，但是
           **平滑降权**而不是**硬性丢弃**——常见词仍然贡献一点分数，只是很少。
           这更合理：一个词常见不代表它没用，只代表它不该单独决定排序。

        2. **tf 项** —— 词频，但边际递减
           一个片段里出现 10 次「南宫婉」，比出现 1 次更可能真的在讲她，
           但相关性不是 10 倍。k1 控制饱和速度（见 config.BM25_K1）。

        3. **|D|/avgdl 长度归一化** —— 消除长片段的系统性优势
           长片段天然更容易碰巧包含查询词。不归一化的话，最长的片段会在
           所有查询里都排前面。b 控制归一化强度（见 config.BM25_B）。

        这修掉了什么真实问题
        --------------------
        改造前这个函数是 `position(词 in 正文) > 0 ... ORDER BY chunk_id`——
        **按片段在书里的先后顺序取前 20 个，完全没有相关性排序**。实测
        《凡人修仙传》19501 个片段里，关键词召回永远只能返回 chunk_id ≤ 10123
        的结果：**后半本书对关键词检索完全不可见**。

        更糟的是这个无序列表会喂给 RRF 融合，而 RRF 的前提是每一路输入都已经
        按相关性排好序——等于给 RRF 喂了噪声。

        only_novels 非空时把搜索范围限定在这些书内。
        """
        terms = query_terms(question)
        if not terms:
            return []

        # BM25_K1 / BM25_B 是从 config 读出来并经过 float() 转换的数值，
        # 不是用户输入，直接内联进 SQL 没有注入风险，可读性比 4 个 %s 好很多。
        k1, b = float(BM25_K1), float(BM25_B)

        # 三处都要按书过滤：语料统计、df 统计、最终打分。范围不一致会让
        # IDF 算错——比如按全库算 df 却只在一本书里打分，稀有词的权重会失真。
        scope_sql = "WHERE novel = ANY(%s)" if only_novels else ""
        scope_and = "AND novel = ANY(%s)" if only_novels else ""
        scope_param = [only_novels] if only_novels else []

        # q(term) 是把查询词做成一张临时表，好跟倒排索引 JOIN
        values_sql = ", ".join(["(%s)"] * len(terms))

        sql = f"""
            WITH q(term) AS (VALUES {values_sql}),
            -- 语料级统计：N（总片段数）和 avgdl（平均片段长度）
            corpus AS (
                SELECT COUNT(*)::float8 AS n,
                       NULLIF(AVG(token_count), 0)::float8 AS avgdl
                FROM novel_chunks
                {scope_sql}
            ),
            -- df：每个查询词各自出现在多少个片段里（IDF 的输入）
            df AS (
                SELECT ct.term, COUNT(*)::float8 AS df
                FROM chunk_terms ct
                JOIN q ON q.term = ct.term
                WHERE TRUE {scope_and.replace("novel", "ct.novel")}
                GROUP BY ct.term
            )
            SELECT nc.novel, nc.chunk_id, nc.chapter_title, nc.text, nc.context,
                   SUM(
                       ln((c.n - d.df + 0.5) / (d.df + 0.5) + 1)
                       * (ct.tf * ({k1} + 1))
                       / (ct.tf + {k1} * (1 - {b} + {b} * nc.token_count / c.avgdl))
                   )::float8 AS score
            FROM chunk_terms ct
            JOIN q ON q.term = ct.term
            JOIN df d ON d.term = ct.term
            JOIN novel_chunks nc
              ON nc.novel = ct.novel AND nc.chunk_id = ct.chunk_id
            CROSS JOIN corpus c
            WHERE TRUE {scope_and.replace("novel", "nc.novel")}
            GROUP BY nc.novel, nc.chunk_id, nc.chapter_title, nc.text, nc.context
            ORDER BY score DESC
            LIMIT %s
        """
        params = [*terms, *scope_param, *scope_param, *scope_param, top_k]

        with connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            SourceChunk(
                novel=row["novel"],
                chunk_id=int(row["chunk_id"]),
                text=row["text"],
                context=row.get("context") or "",
                chapter_title=row.get("chapter_title"),
                # distance 字段在向量检索里是"越小越近"，这里存的是 BM25 分数
                # （越大越相关），语义相反。取负号统一成"越小越好"，避免调用方
                # 按同一个字段排序时把最相关的排到最后。
                distance=-float(row["score"]),
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
                        SELECT novel, chunk_id, chapter_title, text, context
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
                            SELECT novel, chunk_id, chapter_title, text, context
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
                            chapter_title=row.get("chapter_title"),
                            context=row.get("context") or "",
                        )
                    )
        return results

    def _named_novels(self, question: str) -> list[str]:
        """问题里明确提到的书（书名精确或容错匹配）；没提到则返回空列表。"""
        with connect() as conn:
            novels = [
                row["novel"]
                for row in conn.execute(
                    "SELECT DISTINCT novel FROM novel_chunks"
                ).fetchall()
            ]
        return [n for n in novels if _mentions_novel(question, n)]

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
                f"SELECT novel, chunk_id, chapter_title, text, context "
                f"FROM novel_chunks WHERE {conditions}",
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
                chapter_title=row.get("chapter_title"),
                context=row.get("context") or "",
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
