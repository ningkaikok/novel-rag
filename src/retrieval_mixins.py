"""NovelRAG 的单路检索 Mixin：向量、BM25、结构性与邻居扩展。

初学者可以把这里看成 RAG 的“召回层”：每个方法只负责一种召回路径，返回各自
排好序的 ``SourceChunk`` 列表，互相之间不感知——

    retrieve            向量语义检索（pgvector 余弦距离）
    keyword_retrieve    BM25 关键词检索（逐词精确匹配 + 相关性打分）
    positional_retrieve 结构性召回（按 chunk_id 直接取书的首/尾片段）
    expand_neighbors    为命中片段补齐前后相邻片段，供上下文组装使用
    build_answer_context / _expand_chapters
                        M3.4 整章扩展实验：按 CHAPTER_EXPANSION_MODE 决定
                        命中片段以「片段+邻居」还是「所在整章」进入 prompt

多路召回怎么融合、重排、串成流水线，见 ``rag.retrieve_hybrid_stream``；
书名/意图识别的纯函数在 ``novel_match``，数据类在 ``chunk_model``。
"""

import math
import time

from chunk_model import SourceChunk
from config import (
    BM25_B,
    BM25_K1,
    BM25_PER_TERM_LIMIT,
    CHAPTER_EXPANSION_MAX_TOKENS,
    CHAPTER_EXPANSION_MODE,
    CONTEXT_NEIGHBORS,
    TOP_K,
    V2_NATIVE_RETRIEVAL_ENABLED,
    V2_NATIVE_RETRIEVAL_GRAY_PERCENT,
    V2_NATIVE_RETRIEVAL_MODE,
)
from domain_models import RetrievalScope
from index_quality import _token_count
from novel_match import _find_ending_anchor, _mentions_novel
from postgres import (
    connect,
    indexed_novels,
    load_index_manifest,
    vector_literal,
)
from retrieval_scope import select_legacy_novels
from tokenizer import query_terms
from v2_retrieval import V2ReadRepository
from v2_source_adapter import fuse_v2_hits


def _gray_read_enabled(rollout_key: str | None) -> bool:
    if V2_NATIVE_RETRIEVAL_MODE != "gray" or not rollout_key:
        return False
    from hashlib import sha256

    bucket = int(sha256(rollout_key.encode("utf-8")).hexdigest()[:8], 16) % 100
    return bucket < V2_NATIVE_RETRIEVAL_GRAY_PERCENT


class RetrievalMixin:
    @staticmethod
    def _resolve_legacy_scope(
        scope: RetrievalScope | None,
        only_novels: list[str] | None,
    ) -> list[str] | None:
        """把通用范围收敛成 V1 可参数化的小说名单。

        未传 scope 时返回原值，保证旧调用不增加数据库查询；传入 scope 但无法从
        V1 当前索引/manifest 证明映射时返回空列表，由调用方在发 SQL 前短路。
        """

        if scope is None:
            return only_novels
        candidates = only_novels or sorted(indexed_novels())
        source_hashes: dict[str, str | None] | None = None
        if scope.version_id:
            manifests = load_index_manifest()
            source_hashes = {
                novel: str(row["source_hash"]) if row.get("source_hash") else None
                for novel, row in manifests.items()
            }
        return select_legacy_novels(scope, candidates, source_hashes=source_hashes)

    def retrieve(
        self,
        question: str,
        top_k: int = TOP_K,
        only_novels: list[str] | None = None,
        *,
        scope: RetrievalScope | None = None,
    ) -> list[SourceChunk]:
        """向量检索。only_novels 非空时把搜索范围限定在这些书内。"""
        only_novels = self._resolve_legacy_scope(scope, only_novels)
        if scope is not None and not only_novels:
            return []
        query_embedding = self.embedder.encode([question], normalize_embeddings=True)
        query_vector = vector_literal(query_embedding[0])
        scope_sql = "WHERE novel = ANY(%s)" if only_novels else ""
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
                {scope_sql}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                params,
            ).fetchall()
        return [
            SourceChunk.from_legacy_row(row, distance=float(row["distance"])) for row in rows
        ]

    def v2_native_retrieve(
        self,
        question: str,
        top_k: int = TOP_K,
        *,
        rollout_key: str | None = None,
    ) -> list[SourceChunk]:
        """V2 通用文档（非小说）的向量+关键词召回，投影成 SourceChunk 供 RRF 融合。

        开关关闭时零开销直接返回空列表。任何异常都退化为空列表——V2 只是
        锦上添花的一路召回，绝不能让它的故障拖垮 V1 小说问答主链路。

        两路必须**按名次融合**，不能拼起来按原始分数排序
        ------------------------------------------------
        向量那一路的 ``distance`` 是余弦距离（0～2，越小越近）；关键词那一路
        存的是**取负号的 BM25 分数**（量级、符号都不固定，常见 −8～−0.5）。
        旧实现把两个列表拼在一起按 ``hit.distance`` 排序，等于拿余弦距离直接
        和负 BM25 比大小：BM25 分数动辄 −3，于是关键词命中无条件排在语义
        命中前面（即使它在关键词那路里排最后一名），而 BM25 分数的绝对值
        又随语料统计量变化，同一个问题、同样的名次，换一本书就变一个排序。

        现在分开保留两路各自的排名，用 RRF（``1/(k+rank)`` 累加）融合——
        只看“在各路排第几”，与原始分数量纲无关；稳定去重后再取 top_k。
        同一个片段被两路同时命中时会拿到两份 RRF 贡献，自然获得融合加分。
        """
        enabled = V2_NATIVE_RETRIEVAL_ENABLED or V2_NATIVE_RETRIEVAL_MODE == "on"
        enabled = enabled or _gray_read_enabled(rollout_key)
        if not enabled or top_k <= 0:
            return []
        try:
            repository = V2ReadRepository()
            query_embedding = self.embedder.encode([question], normalize_embeddings=True)
        except Exception:
            return []
        # 两路各自单独兜底：关键词那一路挂了不应把已经拿到的语义结果一起丢掉。
        vector_hits = []
        keyword_hits = []
        try:
            vector_hits = repository.vector_search(
                query_embedding[0], top_k=top_k, exclude_legacy_novels=True
            )
        except Exception:
            vector_hits = []
        try:
            keyword_hits = repository.keyword_search(
                query_terms(question), top_k=top_k, exclude_legacy_novels=True
            )
        except Exception:
            keyword_hits = []
        # fuse_v2_hits 按 V2 chunk 的全局唯一 id 去重融合，两路列表各自的
        # 顺序就是各自的排名（repository 已按相关性排好）。
        return [
            SourceChunk.from_v2_hit(hit, distance=hit.distance)
            for hit in fuse_v2_hits(vector_hits, keyword_hits)[:top_k]
        ]

    def keyword_retrieve(
        self,
        question: str,
        top_k: int = TOP_K,
        only_novels: list[str] | None = None,
        *,
        scope: RetrievalScope | None = None,
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

        两阶段聚合改写（M3.4 性能优化，2026-08-23）
        -------------------------------------------
        旧实现是一条 SQL：把所有查询词命中的**全部** (词, 片段) 行聚合求和、
        排序后取 Top-K。EXPLAIN 实测（见 docs/experiments/m34-latency-profile.md），
        常见词（「韩立」命中上万个片段）会让 HashAggregate 处理 15,904 行、
        耗时 ~286ms——而最终只留下 Top-K。索引扫描本身只要 12ms，瓶颈全在
        「为注定进不了 Top-K 的行算分」上。

        现在拆成两阶段：

        1. **SQL 内粗筛**：每个查询词用 LATERAL 子查询按 tf 降序取前
           ``BM25_PER_TERM_LIMIT`` 个候选片段，且候选阶段不携带 text/context
           宽列，只取打分必需的窄字段（tf / token_count / df / 语料统计）。
           聚合行数从上万压到 词数 × N；配合 (term, tf DESC, novel,
           chunk_id) 复合索引（见 postgres._ensure_index_schema），每词的
           Top-N 走 Index Only Scan 提前终止——EXPLAIN 实测常见词场景从
           「堆扫描 15,636 行 + 排序」降到每词 ~0.6ms。
        2. **Python 端精算**：对幸存的每个 (词, 片段) 对用**与 SQL 完全相同**
           的 BM25 公式计算贡献并按 (novel, chunk_id) 求和，排序取前 K；
           最后只为这 K 个片段二次取回正文/章节等宽字段。

        语义近似的 trade-off（必须诚实说明）
        ------------------------------------
        粗筛按「单词条频」截断，可能漏掉这样的极端片段：它在每个命中词上的
        tf 都排不进该词的前 N，但多个词合计的相关性本可以挤进最终 Top-K
        （例如「韩立」只出现 1 次的几万个片段里，「洞府」也出现的那一段）。
        - 每个查询词的命中数 df ≤ N 时，候选集合与旧的全量聚合**完全一致**，
          结果严格等价（稀有词几乎总是如此——df 本身就小）；
        - 只有「多词皆常见且各自 tf 都低」的组合才可能被漏掉，且漏掉的片段
          单词贡献本来就低，实测对 Recall 影响需过夜间检索门禁验证。
        参数化方式：调大 ``BM25_PER_TERM_LIMIT``（config.py，环境变量可覆盖）
        即单调逼近旧行为；N ≥ 所有查询词的 df 时与旧逻辑逐条一致。设为足够大
        的值（如 10**9）即可完全关闭近似。
        """
        terms = query_terms(question)
        if not terms:
            return []
        resolved_novels = self._resolve_legacy_scope(scope, only_novels)
        if scope is not None and not resolved_novels:
            return []
        only_novels = resolved_novels

        # BM25_K1 / BM25_B / BM25_PER_TERM_LIMIT 都是从 config 读出来并经过
        # float()/int() 转换的数值，不是用户输入，直接内联进 SQL 没有注入风险，
        # 可读性比一堆 %s 好很多。
        k1, b = float(BM25_K1), float(BM25_B)
        per_term_limit = int(BM25_PER_TERM_LIMIT)

        # 三处都要按书过滤：语料统计、df 统计、每词候选粗筛。范围不一致会让
        # IDF 算错——比如按全库算 df 却只在一本书里打分，稀有词的权重会失真。
        scope_novels = only_novels or []
        scope_sql = "WHERE novel = ANY(%s)" if scope_novels else ""
        scope_and = "AND novel = ANY(%s)" if scope_novels else ""
        scope_param = [scope_novels] if scope_novels else []

        # q(term) 是把查询词做成一张临时表，好跟倒排索引 JOIN
        values_sql = ", ".join(["(%s)"] * len(terms))

        # 阶段一（SQL）：每个查询词按 tf 取前 per_term_limit 个候选片段。
        # LATERAL 子查询里 ORDER BY 与复合索引 (term, tf DESC, novel,
        # chunk_id) 的键序一致，PostgreSQL 能逐词提前终止扫描；没有该索引时
        # 退化为每词排序，结果不变只是慢。WHERE 过滤先于 LIMIT 生效，所以
        # 「前 N」始终是当前书范围内的前 N。候选行刻意不取 text/context——
        # 它们只在阶段二为 Top-K 取回，让上万个候选行保持窄小正是本次优化
        # 收益的一部分。
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
            SELECT q.term, l.novel, l.chunk_id, l.tf,
                   d.df, nc.token_count, c.n, c.avgdl
            FROM q
            JOIN df d ON d.term = q.term
            CROSS JOIN LATERAL (
                -- 每词 Top-N 候选：tf 相同时按 (novel, chunk_id) 决胜，保证
                -- 截断结果确定可复现（测试与排查都依赖这一点）
                SELECT ct.novel, ct.chunk_id, ct.tf
                FROM chunk_terms ct
                WHERE ct.term = q.term {scope_and.replace("novel", "ct.novel")}
                ORDER BY ct.tf DESC, ct.novel, ct.chunk_id
                LIMIT {per_term_limit}
            ) l
            JOIN novel_chunks nc
              ON nc.novel = l.novel AND nc.chunk_id = l.chunk_id
            CROSS JOIN corpus c
        """
        params = [*terms, *scope_param, *scope_param, *scope_param]

        with connect() as conn:
            candidate_rows = conn.execute(sql, params).fetchall()
            if not candidate_rows:
                return []

            # 语料统计经 CROSS JOIN 广播在每一行上，任意取一行即可。
            n = float(candidate_rows[0]["n"] or 0)
            avgdl = float(candidate_rows[0]["avgdl"] or 0)
            if not n or not avgdl:
                # 空语料或全部 token_count=0 的退化库：旧 SQL 会产出 NULL 分数，
                # 与其返回无意义的排序不如明确空结果。
                return []

            # 阶段二（Python）：对幸存的 (词, 片段) 对逐项套用与旧 SQL 完全相同的
            # BM25 公式（见本函数 docstring 顶部的公式），按片段求和得总分。
            scores: dict[tuple[str, int], float] = {}
            for row in candidate_rows:
                tf = float(row["tf"])
                df = float(row["df"])
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                denom = tf + k1 * (1 - b + b * float(row["token_count"]) / avgdl)
                key = (row["novel"], int(row["chunk_id"]))
                scores[key] = scores.get(key, 0.0) + idf * tf * (k1 + 1) / denom

            # 分数相同时按 (novel, chunk_id) 决胜：排序确定，测试可复现。
            ranked_keys = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]

            # 只为最终 Top-K 取回正文等宽字段（旧实现是聚合全程拖着这些列）。
            conditions = " OR ".join("(novel = %s AND chunk_id = %s)" for _ in ranked_keys)
            meta_rows = conn.execute(
                f"SELECT novel, chunk_id, chapter_title, text, context "
                f"FROM novel_chunks WHERE {conditions}",
                [value for key in ranked_keys for value in key],
            ).fetchall()
            meta = {(row["novel"], int(row["chunk_id"])): row for row in meta_rows}

        results: list[SourceChunk] = []
        for key in ranked_keys:
            row = meta.get(key)
            if row is None:  # 候选到取回之间数据被删的极端兜底
                continue
            results.append(
                SourceChunk.from_legacy_row(
                    row,
                    # distance 字段在向量检索里是"越小越近"，这里存的是 BM25 分数
                    # （越大越相关），语义相反。取负号统一成"越小越好"，避免调用方
                    # 按同一个字段排序时把最相关的排到最后。
                    distance=-scores[key],
                )
            )
        return results

    def positional_retrieve(
        self,
        question: str,
        top_k: int = TOP_K,
        hint_novels: list[str] | None = None,
        *,
        scope: RetrievalScope | None = None,
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

        scoped_novels = self._resolve_legacy_scope(scope, None)
        if scope is not None and not scoped_novels:
            return []

        with connect() as conn:
            novels = [
                row["novel"]
                for row in conn.execute("SELECT DISTINCT novel FROM novel_chunks").fetchall()
            ]
            allowed_novels: list[str] = (
                list(scoped_novels) if scoped_novels is not None else novels
            )
            # 问题里提到了某本书就只查那本，避免把别的书的结尾混进来
            matched = [n for n in allowed_novels if _mentions_novel(text, n)]
            if matched:
                targets = matched
            elif hint_novels:
                # 只保留确实存在于库里的提示书名
                targets = [
                    n for n in allowed_novels if n in set(hint_novels)
                ] or allowed_novels
            else:
                targets = allowed_novels

            # 每本书分配的配额：只查一本时全给它，多本时平摊但至少 1 段
            per_novel = max(1, top_k // len(targets)) if targets else top_k
            order = "DESC" if at_tail else "ASC"
            results: list[SourceChunk] = []
            for novel in targets:
                anchor = _find_ending_anchor(conn, novel) if at_tail else None
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
                    results.append(SourceChunk.from_legacy_row(row))
        return results

    def _named_novels(self, question: str) -> list[str]:
        """问题里明确提到的书（书名精确或容错匹配）；没提到则返回空列表。"""
        with connect() as conn:
            novels = [
                row["novel"]
                for row in conn.execute("SELECT DISTINCT novel FROM novel_chunks").fetchall()
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

        **只给 V1 小说命中补邻居**：下游回 ``novel_chunks`` 靠的是
        ``(novel, chunk_id)``，而 V2 文档片段的这两个字段只是展示投影（文档标题
        + 片段序号）。拿它去查 V1 表会得到两种错误后果，而且都不报错：

        - 存在同名小说时，V2 命中会被同名小说的 V1 正文**静默替换**，最终
          prompt 里的证据变成另一本书的内容；
        - 不存在同名小说时，该 V2 命中根本不在 ``by_key`` 里，被整个丢掉——
          混合检索时表现为"上传的文档明明召回到了却答不出来"。

        所以非 V1 来源原样透传，位置仍按检索相关性顺序，不参与邻居查询。
        """
        if not sources or neighbors <= 0:
            return sources

        legacy_sources = [source for source in sources if source.is_legacy_novel]
        ranges: list[tuple[str, int, int]] = []
        for source in legacy_sources:
            ranges.append(
                (
                    source.novel,
                    max(0, source.chunk_id - neighbors),
                    source.chunk_id + neighbors,
                )
            )

        by_key: dict[tuple[str, int], SourceChunk] = {}
        if ranges:
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
            for row in rows:
                key = (row["novel"], int(row["chunk_id"]))
                by_key[key] = SourceChunk.from_legacy_row(row)

        # 把原始命中也放进 by_key（真实 DB 行优先）。邻居区间查询可能是**部分**的
        # ——索引正在增量重建，或某个邻居片段刚好被删改。那种情况下命中自己没回来，
        # 如果只靠 by_key 组装，它会被静默丢掉：最终 prompt 里少一条检索真正找到的
        # 证据，而且不报错。
        for source in legacy_sources:
            by_key.setdefault((source.novel, source.chunk_id), source)

        expanded: list[SourceChunk] = []
        seen: set[tuple] = set()
        # 按检索相关性保留不同命中簇的顺序；每个命中簇内部按原文顺序排列。
        for source in sources:
            if not source.is_legacy_novel:
                # V2 命中：原文/来源/章节信息全部原样保留，只去重不替换
                if source.identity_key not in seen:
                    seen.add(source.identity_key)
                    expanded.append(source)
                continue
            group = [
                by_key[(source.novel, chunk_id)]
                for chunk_id in range(
                    max(0, source.chunk_id - neighbors), source.chunk_id + neighbors + 1
                )
                if (source.novel, chunk_id) in by_key
            ]
            # 命中自己一定在 group 里（上面已 setdefault 补入），其余能拿到的邻居
            # 照常补上；顺序由 range 决定，即原文 chunk_id 顺序。
            for item in group:
                key = item.identity_key
                if key not in seen:
                    expanded.append(item)
                    seen.add(key)
        return expanded

    def build_answer_context(
        self, sources: list[SourceChunk]
    ) -> tuple[list[SourceChunk], dict | None]:
        """重排结果 → 最终进入 prompt 的证据列表（M3.4 上下文组装的统一入口）。

        这是 expand_neighbors 外面的一层「调度壳」，按 CHAPTER_EXPANSION_MODE 分发：

        - off / neighbors：直接走原有邻居扩展，且**不产生任何 trace 步骤**——
          回归路径必须与改动前逐字节一致，评测脚本和前端都感知不到这层壳；
        - chapter：整章扩展实验（见 _expand_chapters），额外返回一条带模式、
          证据 token 数、是否截断的 trace 步骤，由调用方（backend/main.py）
          并入「思考过程」，供评测复盘。

        返回 ``(最终证据列表, trace 步骤或 None)``。
        """
        if CHAPTER_EXPANSION_MODE != "chapter":
            return self.expand_neighbors(sources), None
        return self._expand_chapters(sources)

    def _expand_chapters(self, sources: list[SourceChunk]) -> tuple[list[SourceChunk], dict]:
        """整章扩展实验档：命中所在章节的全部片段按原文顺序进入 prompt。

        为什么是实验项而不是默认行为
        ----------------------------
        邻居扩展只补 ±CONTEXT_NEIGHBORS 个片段，切分边界仍可能截断证据；
        整章彻底消除边界问题，但一章可能有几十个片段，信噪比急剧下降——
        「迷失在中间」效应会让模型漏读真正相关的段落。所以必须受硬性预算
        约束，并和邻居机制在固定评测集上对照（而不是直接替换）。

        算法分四步：

        1. 把命中按（书名, 章节标题）聚合，组的先后沿用检索相关性顺序；
           跨书天然隔离——书名是分组键的一部分，一本书的章节永远不会把
           另一本书的内容拉进来。只有 V1 小说命中能按章节回 ``novel_chunks``
           补正文；V2 文档命中（``origin="v2_document"``）和没有章节标题的
           命中一样原样透传，不参与章节查询——它们的 novel/chunk_id 只是
           展示投影，拿回去查 V1 表会把同名小说的正文当成 V2 证据。
        2. 每个章节用「书名 + 章节标题」取回全部片段，按 chunk_id 排序
           （与 agent_lab.get_chapter、层级摘要的章节口径一致）。没有章节
           标题的命中（旧索引 / 无标题 txt）无从谈「章」，退化为只保留命中
           片段本身，不猜边界，trace 里如实记录。
        3. token 预算闸门：先无条件保留所有命中片段（它们是检索找到的证据，
           不能被预算挤掉），再从命中的连续区间**向两侧对称生长**——每轮
           比较左右两侧下一个片段的 token 开销，纳入放得下的更便宜的一侧，
           直到两侧都放不下为止。这样截断方向永远是从离命中最远的地方开始丢。
           **不连续命中之间的中间片段不是免费的**：它们同样要占 prompt 预算，
           必须从剩余额度里扣；扣不起就不带入（只留命中），否则「为了维持
           连续区间」白送的中间片段会直接把证据总量顶出预算。
        4. 组装结果：组间按相关性顺序、组内按 chunk_id 原文顺序，并产出
           一条 trace 步骤（mode / evidence_tokens / truncated / 截断原因）。

        计数口径：embedding 模型的真实 tokenizer（index_quality._token_count，
        truncation=False、含特殊 token），与 M3.3 索引质量门禁完全一致。tokenizer
        不可用时**不假装闸门存在**：全量带入但在 trace 里明确记录闸门未执行，
        绝不静默跳过。
        """
        started = time.perf_counter()
        budget = CHAPTER_EXPANSION_MAX_TOKENS

        if not sources:
            # 空输入：不要装作"闸门执行了"，也不要凭空报 tokenizer 不可用。
            return [], {
                "step": "上下文扩展",
                "stage_key": "context_expand",
                "detail": "整章扩展：本轮没有命中片段，无需扩展",
                "ms": int((time.perf_counter() - started) * 1000),
                "expansion_mode": CHAPTER_EXPANSION_MODE,
                "evidence_tokens": 0,
                "truncated": False,
                "truncation_reason": None,
            }

        # ---- 第一步：聚合命中。units 保持相关性顺序，元素两种形状：
        #   ("chapter", (novel, title), [命中 chunk_id])   有章节标题的 V1 命中组
        #   ("single", SourceChunk)                         直接透传的命中（V2 片段 /
        #                                                   无章节标题的 V1 片段）
        units: list[tuple] = []
        unit_index: dict[tuple[str, str], int] = {}
        seen_hit: set[tuple] = set()
        for source in sources:
            key = source.identity_key
            if key in seen_hit:  # 防御：上游理论上已去重，这里兜底
                continue
            seen_hit.add(key)
            if not source.is_legacy_novel or not source.chapter_title:
                units.append(("single", source))
                continue
            group_key = (source.novel, source.chapter_title)
            if group_key in unit_index:
                units[unit_index[group_key]][2].append(source.chunk_id)
            else:
                unit_index[group_key] = len(units)
                units.append(("chapter", group_key, [source.chunk_id]))

        # ---- 第二步：整章取回。一次查询一章，ORDER BY chunk_id 保证原文顺序。
        chapters: dict[tuple[str, str], list[SourceChunk]] = {}
        with connect() as conn:
            for _kind, group_key, _hits in (u for u in units if u[0] == "chapter"):
                novel, title = group_key
                rows = conn.execute(
                    "SELECT novel, chunk_id, chapter_title, text, context "
                    "FROM novel_chunks WHERE novel = %s AND chapter_title = %s "
                    "ORDER BY chunk_id",
                    (novel, title),
                ).fetchall()
                members = [SourceChunk.from_legacy_row(row) for row in rows]
                merged: dict[tuple, SourceChunk] = {}
                for member in members:
                    merged.setdefault(member.identity_key, member)
                # 章节查询可能是**部分**的（增量重建、片段被删改、索引与命中不一致）：
                # 该组里没查回来的原始命中必须按 identity 补回来，再按 chunk_id 排序。
                # 只靠“整章查不到”的分支兜底是不够的：查到一半时那些没回来的命中
                # 会从证据里静默消失，而它们才是检索真正找到的片段。
                # 也只收 **V1** 命中：同名 V2 文档片段的展示标题/章节名可以一模一样，
                # 但它是透传单位（"single"），收进来会重复输出两次。
                for source in sources:
                    if (
                        source.is_legacy_novel
                        and (
                            source.novel,
                            source.chapter_title,
                        )
                        == group_key
                    ):
                        merged.setdefault(source.identity_key, source)
                # 组内顺序稳定：同一章内部永远按原文 chunk_id 排列（与整章查询口径
                # 一致），不依赖 SQL 返回顺序或补入先后。
                chapters[group_key] = sorted(merged.values(), key=lambda chunk: chunk.chunk_id)

        # ---- 第三步：token 计数与预算闸门 --------------------------------
        def _tokens(chunk: SourceChunk) -> int | None:
            return _token_count(self.embedder, chunk.text)

        all_members: dict[tuple, list[SourceChunk]] = {
            **chapters,
            **{
                ("single", id(single)): [single]
                for single in (u[1] for u in units if u[0] == "single")
            },
        }
        # token 计数的覆盖范围必须包含**所有原始命中**，而不只是章节查询回来的
        # 成员：章节表与命中不一致时（兜底分支），命中原文也会真的进 prompt，
        # 不计费就等于白送证据、还把 trace 写成 0 tokens / 未截断。
        counted: dict[tuple, SourceChunk] = {}
        for members in all_members.values():
            for member in members:
                counted.setdefault(member.identity_key, member)
        for source in sources:
            counted.setdefault(source.identity_key, source)
        counts = {key: _tokens(chunk) for key, chunk in counted.items()}
        tokenizer_usable = bool(counts) and all(c is not None for c in counts.values())

        selected: set[tuple] = set()
        truncated = False
        reasons: list[str] = []
        evidence_tokens: int | None
        # 章节查询没能覆盖到的命中组（数据不一致）：不猜章节边界，只保留命中原文。
        fallback_groups: dict[tuple, list[SourceChunk]] = {}

        if not tokenizer_usable:
            # 无法计数就不执行闸门，但必须在 trace 里说清楚——静默跳过长度
            # 检查正是 M3.3 要消灭的行为。
            reasons.append("embedding tokenizer 不可用，token 预算闸门未执行")
            for members in all_members.values():
                selected.update(m.identity_key for m in members)
            evidence_tokens = None
        else:

            def _cost(chunk: SourceChunk) -> int:
                return counts.get(chunk.identity_key) or 0

            hit_total = sum(counts.get(key) or 0 for key in seen_hit)
            if hit_total > budget:
                # 极端情况：光命中片段本身就超预算。证据不能丢，只能全保留
                # 并如实记录——这个信号说明该书的章节粒度远超预期，评测时要看。
                truncated = True
                reasons.append(
                    f"命中片段本身已超预算（{hit_total} > {budget} tokens），仅保留命中"
                )
                remaining = 0
            else:
                remaining = budget - hit_total

            # 逐组从命中区间向两侧对称生长（组间按相关性顺序消耗同一份预算，
            # 靠前的相关章节优先拿到余量）。hlo/hhi 是初始命中边界，用来在
            # 两侧 token 开销打平时优先扩展「伸得更短」的那一侧，保证纳入的
            # 片段始终紧贴命中区间、截断方向从离命中最远处开始丢。
            for _kind, group_key_or_single, hits in (u for u in units if u[0] == "chapter"):
                members = chapters[group_key_or_single]
                positions = {i for i, m in enumerate(members) if m.chunk_id in hits}
                if not positions:
                    # 防御：上面的 identity 补入已经保证该组每个命中都在 members 里。
                    # 真到了这里（数据异常），宁可只保留命中原文，也不丢证据；
                    # 这些命中已在 hit_total 里计过费，trace 会如实报告总量/超预算。
                    fallback: dict[tuple, SourceChunk] = {}
                    for source in sources:
                        if (
                            source.is_legacy_novel
                            and (
                                source.novel,
                                source.chapter_title,
                            )
                            == group_key_or_single
                        ):
                            fallback.setdefault(source.identity_key, source)
                    fallback_groups[group_key_or_single] = list(fallback.values())
                    selected.update(fallback)
                    continue
                lo, hi = min(positions), max(positions)
                hlo, hhi = lo, hi
                # 命中本身已在 hit_total 里付过账，先只把它们标进 selected。
                for index in positions:
                    selected.add(members[index].identity_key)
                # 命中区间里的非命中片段不是免费的：不连续命中之间的中间片段
                # 同样占 prompt 预算。付不起就干脆不带（只保留命中本身）。
                interior = [members[i] for i in range(lo, hi + 1) if i not in positions]
                interior_cost = sum(_cost(member) for member in interior)
                if interior_cost > remaining:
                    truncated = True
                    reasons.append(
                        f"命中之间的 {len(interior)} 段中间片段需 {interior_cost} tokens，"
                        f"超出剩余预算 {remaining}，未带入"
                    )
                    continue
                remaining -= interior_cost
                selected.update(member.identity_key for member in interior)
                while remaining > 0 and (lo > 0 or hi < len(members) - 1):
                    candidates = []
                    if lo > 0:
                        left = members[lo - 1]
                        candidates.append((_cost(left), hlo - lo, lo - 1))
                    if hi < len(members) - 1:
                        right = members[hi + 1]
                        candidates.append((_cost(right), hi - hhi, hi + 1))
                    fit = [c for c in candidates if c[0] <= remaining]
                    if not fit:
                        break
                    cost, _pending, position = min(fit)
                    chosen = members[position]
                    selected.add(chosen.identity_key)
                    remaining -= cost
                    lo, hi = min(lo, position), max(hi, position)
                if lo > 0 or hi < len(members) - 1:
                    truncated = True
            if truncated and not reasons:
                reasons.append(f"超出 token 预算 {budget}，已从命中片段向两侧截断")

            for single in (u[1] for u in units if u[0] == "single"):
                selected.add(single.identity_key)

            evidence_tokens = sum(counts.get(key) or 0 for key in selected)

        # ---- 第四步：组装结果与 trace ------------------------------------
        result: list[SourceChunk] = []
        emitted: set[tuple] = set()
        for unit in units:
            if unit[0] == "single":
                result.append(unit[1])
                continue
            if unit[1] in fallback_groups:
                for source in fallback_groups[unit[1]]:
                    if source.identity_key not in emitted:
                        result.append(source)
                        emitted.add(source.identity_key)
                continue
            for member in chapters[unit[1]]:
                key = member.identity_key
                if key in selected and key not in emitted:
                    result.append(member)
                    emitted.add(key)

        titled_groups = sum(1 for u in units if u[0] == "chapter")
        detail = (
            f"整章扩展：{len(sources)} 个命中聚合成 "
            f"{titled_groups} 个章节组，证据 "
            f"{evidence_tokens if evidence_tokens is not None else '?'}/{budget} tokens"
        )
        step = {
            "step": "上下文扩展",
            "stage_key": "context_expand",
            "detail": detail,
            "ms": int((time.perf_counter() - started) * 1000),
            "expansion_mode": CHAPTER_EXPANSION_MODE,
            "evidence_tokens": evidence_tokens,
            "truncated": truncated,
            "truncation_reason": "；".join(reasons) if reasons else None,
        }
        return result, step
