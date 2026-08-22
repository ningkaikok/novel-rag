"""RAG 编排层：多路召回的流水线调度与对外兼容入口。

这个模块现在是**编排层**：各职责已经拆到专职模块，这里负责把它们串成一条
完整的检索-生成流水线，并保持 ``from rag import ...`` 的历史用法不变——

    novel_match.py       书名识别与问题意图的纯函数（错字容错、结构问题、目录题）
    chunk_model.py       SourceChunk 数据类与候选 trace 记录
    retrieval_mixins.py  单路召回：向量 / BM25 / 结构性 / 邻居扩展（RetrievalMixin）
    generation_mixin.py  prompt 组装、图线索、Ollama 生成（GenerationMixin）

留在本模块的方法有两类：一是流水线编排本身（``retrieve_hybrid_stream`` 及其
一次性/简化包装），二是被测试以 ``patch.object(rag, "connect")`` 等方式打桩、
因此必须从本模块全局取名字的方法（``library_answer``、``_full_text_chunks``、
``hierarchy_retrieve``）。

这个模块刻意用普通 Python 函数显式编排，而不是交给 LangGraph：标准 RAG 请求是一条
短生命周期、方向固定的流水线，没有工具调用循环、人工审批或失败后跨进程恢复的
需求。学习时可以直接沿着 ``retrieve_hybrid_stream`` 阅读每个阶段的数据变化。

核心对象在各阶段的变化如下，阅读时注意“召回候选”和“最终上下文”不是一回事：

    question
      → 全局问题可选：全书/章节摘要定位 → 映射回 SourceChunk 原文
      → semantic / BM25 / positional 等多路 SourceChunk 候选
      → RRF 去重融合后的候选池
      → CrossEncoder 重排后的 top-k
      → expand_neighbors 补齐相邻片段
      → 带 [n] 编号的 prompt
      → 模型 token 流

层级摘要只负责导航，不能充当原文证据；召回负责“别漏掉”，重排负责“把正确答案
提到前面”，邻居扩展负责“别让切分边界截断证据”。把这些步骤混为一个阶段，会很难
判断检索质量究竟坏在哪一层。

Web 层和云端模型路由在 ``backend/main.py``；这里不依赖 FastAPI，因此评测脚本
可以直接调用检索逻辑。完整选型理由见 ``docs/architecture-decisions.md``。
"""
import time

from sentence_transformers import SentenceTransformer

from embedder import load_embedder
from config import (
    FULL_TEXT_MAX_CHARS,
    HIERARCHY_ENABLED,
    HIERARCHY_TOP_K,
    QUERY_EXPAND_ENABLED,
    QUERY_EXPAND_MAX_VARIANTS,
    RERANK_CANDIDATE_MULTIPLIER,
    RERANK_ENABLED,
    RECALL_K,
    TOP_K,
)
from hierarchy import is_global_question
from postgres import (
    connect,
    has_index,
    search_hierarchy,
    vector_literal,
)
from reranker import rerank_with_scores
from confidence import compute_confidence
from query_expander import expand_query_variants
from chunk_model import SourceChunk, _trace_candidates
from novel_match import (
    _display_title,
    _dominant_novels,
    _edit_distance,
    _find_ending_anchor,
    _fuzzy_contains,
    _is_library_question,
    _mentions_novel,
    _named_via_typo,
    _novel_titles,
    _strip_novel_titles,
    _structural_kind,
)
from retrieval_mixins import RetrievalMixin
from generation_mixin import (
    PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_VERSION,
    GenerationMixin,
    generate_ollama_prompt_stream,
)


class NovelRAG(RetrievalMixin, GenerationMixin):
    # 自适应查询扩展（M3.4）用的生成函数，由 Web 层注入：本模块刻意不依赖
    # FastAPI / 云端 SDK，评测脚本和测试里这个属性是 None，扩展自动跳过。
    # backend/main.py 启动时按 QUERY_EXPAND_MODEL 的前缀路由到 zhipu/claude_cli。
    expand_generate_fn = None

    def __init__(self, embedder: SentenceTransformer | None = None):
        self.embedder = embedder or load_embedder()
        if not has_index():
            raise RuntimeError("PostgreSQL novel_chunks 表不存在，请先重建索引")

    def library_answer(self, question: str) -> str | None:
        """回答书架目录问题，避免把 top-k 召回范围误当成全集。

        这是结构化元数据查询，不经过 embedding、BM25、RRF 或 LLM。完整目录问题
        的正确性来自数据库的 ``DISTINCT novel``，而不是来自某一批碰巧召回的片段。
        非目录问题返回 ``None``，继续走普通 RAG 流程。
        """
        if not _is_library_question(question):
            return None
        with connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT novel FROM novel_chunks ORDER BY novel"
            ).fetchall()
        novels = [str(row["novel"]) for row in rows]
        if not novels:
            return "当前书架中没有已建立索引的小说。"
        titles = [_display_title(novel) for novel in novels]
        return f"当前书架一共有 {len(titles)} 部小说：" + "、".join(titles) + "。"

    def _full_text_chunks(self, named_novels: list[str]) -> list[SourceChunk] | None:
        """书够小就返回它的全部片段（按原文顺序），否则返回 None。

        这是 RAG 四杠杆里「长上下文取舍」的另一面：**不是所有场景都需要 RAG**。
        当整份材料能塞进模型的上下文窗口时，检索反而是在给自己制造风险——
        召回可能漏掉关键信息，而全文给过去则零信息损失。

        触发条件刻意收得很紧：
        - **必须只点名了一本书**。跨书问题不能这么做——两本书各自很小，
          合起来也可能超窗口，而且混在一起会干扰模型判断。
        - **必须没点名也不行**。没点名时无法确定范围，只能老老实实检索。
        - 全文字数在 FULL_TEXT_MAX_CHARS 以内。

        现状：当前语料里只有《雾隐山庄》（1229 字）会触发。这个短路主要服务于
        "用户上传一份小文档"的场景。
        """
        if len(named_novels) != 1:
            return None
        novel = named_novels[0]
        with connect() as conn:
            row = conn.execute(
                "SELECT SUM(LENGTH(text)) AS chars FROM novel_chunks WHERE novel = %s",
                (novel,),
            ).fetchone()
            total = int(row["chars"] or 0)
            if not total or total > FULL_TEXT_MAX_CHARS:
                return None
            rows = conn.execute(
                "SELECT novel, chunk_id, chapter_title, text, context FROM novel_chunks "
                "WHERE novel = %s ORDER BY chunk_id",
                (novel,),
            ).fetchall()
        return [
            SourceChunk(
                novel=r["novel"],
                chunk_id=int(r["chunk_id"]),
                text=r["text"],
                distance=0.0,
                chapter_title=r.get("chapter_title"),
                context=r.get("context") or "",
            )
            for r in rows
        ]

    def hierarchy_retrieve(
        self,
        question: str,
        *,
        named_novels: list[str] | None = None,
        top_k: int = HIERARCHY_TOP_K,
    ) -> tuple[list[SourceChunk], list[dict]]:
        """先搜索摘要节点，再把命中章节映射回可引用的原文代表片段。

        全书节点只用于判断“应该看哪本书”，章节节点负责定位具体范围。每个命中
        章节取开头/中间/结尾三处原文，随后仍会进入 RRF 和交叉编码器重排。
        """
        query_embedding = self.embedder.encode([question], normalize_embeddings=True)
        query_vector = vector_literal(query_embedding[0])

        targets = list(named_novels or [])
        book_hits: list[dict] = []
        if not targets:
            book_hits = search_hierarchy(
                query_vector,
                level="novel",
                limit=2,
            )
            targets = list(dict.fromkeys(hit["novel"] for hit in book_hits))
        if not targets:
            return [], []

        # 跨书比较必须让每本书都有候选，不能全局 LIMIT 后被其中一本包揽。
        per_novel = max(2, (top_k + len(targets) - 1) // len(targets))
        chapter_hits: list[dict] = []
        for novel in targets:
            chapter_hits.extend(
                search_hierarchy(
                    query_vector,
                    level="chapter",
                    limit=per_novel,
                    novels=[novel],
                )
            )
        chapter_hits.sort(key=lambda hit: float(hit["distance"]))

        sources: list[SourceChunk] = []
        seen: set[tuple[str, int]] = set()
        with connect() as conn:
            for hit in chapter_hits[: max(top_k, len(targets) * 2)]:
                start = int(hit["start_chunk_id"])
                end = int(hit["end_chunk_id"])
                representative_ids = sorted({start, (start + end) // 2, end})
                rows = conn.execute(
                    "SELECT novel, chunk_id, chapter_title, text, context "
                    "FROM novel_chunks WHERE novel = %s AND chunk_id = ANY(%s) "
                    "ORDER BY chunk_id",
                    (hit["novel"], representative_ids),
                ).fetchall()
                for row in rows:
                    key = (row["novel"], int(row["chunk_id"]))
                    if key in seen:
                        continue
                    seen.add(key)
                    sources.append(
                        SourceChunk(
                            novel=row["novel"],
                            chunk_id=int(row["chunk_id"]),
                            text=row["text"],
                            # 摘要距离只表示这个章节整体与问题的相关性；后面重排会
                            # 重新判断具体原文片段，因此这里只保留为候选排序信号。
                            distance=float(hit["distance"]),
                            chapter_title=row.get("chapter_title") or hit["title"],
                            context=row.get("context") or "",
                        )
                    )
        return sources, [*book_hits, *chapter_hits]

    def retrieve_hybrid(self, question: str, top_k: int = TOP_K) -> list[SourceChunk]:
        """统一的两阶段召回：候选池合并后用轻量 RRF 排序，最终取 top-k。"""
        sources, _ = self.retrieve_hybrid_traced(question, top_k)
        return sources

    def retrieve_hybrid_traced(
        self, question: str, top_k: int = TOP_K
    ) -> tuple[list[SourceChunk], list[dict]]:
        """同 retrieve_hybrid，但额外返回一份「思考过程」trace。

        一次性拿到全部结果的版本，给评测脚本和测试用。
        接口端走 retrieve_hybrid_stream()，好把步骤边跑边推给前端。
        """
        trace: list[dict] = []
        sources: list[SourceChunk] = []
        for kind, payload in self.retrieve_hybrid_stream(question, top_k):
            if kind == "step":
                trace.append(payload)
            else:
                sources = payload
        return sources, trace

    def retrieve_hybrid_stream(
        self, question: str, top_k: int = TOP_K, _allow_expand: bool = True
    ):
        """检索流水线的生成器版本：每完成一个阶段就 yield 一次，最后 yield 结果。

        **为什么要做成生成器**：整条流水线要 2 秒左右（交叉编码器重排占大头），
        一次性返回的话用户在这 2 秒里什么都看不到，只能干等。改成边跑边报之后，
        界面可以像成熟的 AI 应用那样把步骤一条条点亮——**等待时间没变，但
        用户知道系统在干什么**，感受完全不同。

        顺带解决了一个真实的性能问题：原来接口端在 `async def` 里同步调用检索，
        这 2 秒会**阻塞整个事件循环**（其他请求全卡住，连断连检查都跑不了）。
        改成生成器后，接口端可以用 `run_in_threadpool` 逐步取，每取一次就是
        一次让出控制权的机会——和下面消费模型 token 用的是同一套模式。

        ``_allow_expand``：查询扩展（M3.4）的防循环闸门。补救时对每个变体
        会再次调用本方法，变体的检索**禁止再触发扩展**——否则低置信度问题
        可能无限递归烧钱。外部调用一律用默认值 True。

        yield 的形状：
            ("step",   {"step": 阶段名, "detail": 说明, "ms": 本阶段耗时})
            ("result", [SourceChunk, ...])   ← 最后一条，只有一条
        """
        # 每个阶段各自计时：光看总耗时不知道慢在哪，分段之后一眼能看出
        # 重排占了大头（也正是靠这个数据才确定了「重排让你可以少送」这个结论）。
        mark = time.perf_counter()

        def took() -> int:
            """返回距上一个阶段的毫秒数，并重置计时起点。"""
            nonlocal mark
            now = time.perf_counter()
            ms = int((now - mark) * 1000)
            mark = now
            return ms

        # 候选池要明显大于最终要的条数，重排才有东西可挑。
        # 不开重排时退回原来的规模，避免白白多召回、多花时间。
        candidate_k = (
            max(top_k * RERANK_CANDIDATE_MULTIPLIER, RECALL_K)
            if RERANK_ENABLED
            else max(top_k, RECALL_K)
        )
        # 阶段一：理解问题——点没点书名（含错字容错）、是不是问结构（结局/开头）
        named_novels = self._named_novels(question)
        global_question = is_global_question(question)

        # 「长上下文取舍」：书小到能整本塞进模型窗口时，检索本身就是多余的。
        # 与其检索出几段（可能漏掉关键信息），不如直接给全文——零信息损失。
        # 只在问题明确点名了某本书、且那本书足够小时触发（见 _full_text_chunks）。
        full_text = self._full_text_chunks(named_novels)
        if full_text is not None:
            title = _display_title(named_novels[0])
            chars = sum(len(c.text) for c in full_text)
            yield "step", {"step": "理解问题", "detail": f"识别到你在问{title}", "ms": took()}
            yield "step", {
                "step": "跳过检索",
                "detail": f"{title}全文仅 {chars} 字，直接把整本给模型，不做检索（零信息损失）",
                "ms": took(),
            }
            yield "result", full_text
            return

        structural = _structural_kind(question)  # "结局" / "开头" / None
        if named_novels:
            named_desc = "、".join(_display_title(n) for n in named_novels)
            corrected = [n for n in named_novels if _named_via_typo(question, n)]
            hint = "（书名有错别字，已自动纠正）" if corrected else ""
            detail = f"识别到你在问{named_desc}{hint}"
        else:
            detail = "未点明书名，稍后根据检索内容自动判断属于哪本书"
        if structural:
            detail += f"，且在问「{structural}」这类结构性问题"
        if global_question:
            detail += "，识别为需要跨章节观察的全局问题"
        yield "step", {"step": "理解问题", "detail": detail, "ms": took()}

        # 阶段二：确定检索范围
        if named_novels:
            scope_detail = f"只在{'、'.join(_display_title(n) for n in named_novels)}内检索"
        else:
            scope_detail = "在全部书里检索"
        yield "step", {"step": "检索范围", "detail": scope_detail, "ms": took()}

        hierarchy_sources: list[SourceChunk] = []
        hierarchy_hits: list[dict] = []
        if HIERARCHY_ENABLED and global_question:
            try:
                hierarchy_sources, hierarchy_hits = self.hierarchy_retrieve(
                    question,
                    named_novels=named_novels,
                )
                chapter_count = sum(hit.get("level") == "chapter" for hit in hierarchy_hits)
                novels = list(dict.fromkeys(hit["novel"] for hit in hierarchy_hits))
                yield "step", {
                    "step": "层级检索",
                    "stage_key": "hierarchy",
                    "detail": (
                        f"先用全书/章节摘要定位到 {chapter_count} 个章节，"
                        f"再回到 {'、'.join(_display_title(n) for n in novels)} 的原文取证"
                    ),
                    "ms": took(),
                    "candidates": _trace_candidates(
                        hierarchy_sources,
                        score_label="章节摘要相似度",
                        score_of=lambda source, _key: 1 - source.distance,
                    ),
                }
            except Exception as exc:
                # 层级表尚未迁移或暂时不可用时，退回原有片段检索，不让升级过程阻断问答。
                yield "step", {
                    "step": "层级检索",
                    "stage_key": "hierarchy",
                    "detail": f"层级摘要暂不可用，退回片段检索（{exc}）",
                    "ms": took(),
                }

        # 阶段三：多路召回
        semantic_sources = self.retrieve(
            question, top_k=candidate_k, only_novels=named_novels
        )
        yield "step", {
            "step": "向量召回",
            "stage_key": "vector",
            "detail": f"按语义相似度召回 {len(semantic_sources)} 个片段",
            "ms": took(),
            "candidates": _trace_candidates(
                semantic_sources,
                score_label="余弦相似度",
                score_of=lambda source, _key: 1 - source.distance,
            ),
        }
        # 关键词检索（分词后逐词匹配）如果不限定书的范围，像"伴侣"这种两本书
        # 都会用到的常见词，会把不相关小说的片段也拉进来。这里先用语义召回的
        # 结果猜一次书（没点名书名时），再拿这个猜测去收窄关键词检索——语义
        # 检索本身不受这个范围限制，全书候选池仍然完整，只是关键词这一路
        # 收窄了范围。
        keyword_scope = named_novels or _dominant_novels(semantic_sources)
        # 书名已经用来确定检索范围了，不该再作为内容词参与 BM25 打分
        # （详见 _strip_novel_titles 的说明——这个 bug 实测能让无关片段反超正确答案）
        keyword_sources = self.keyword_retrieve(
            _strip_novel_titles(question, keyword_scope),
            top_k=candidate_k,
            only_novels=keyword_scope,
        )
        yield "step", {
            "step": "BM25 召回",
            "stage_key": "bm25",
            "detail": f"按关键词相关性召回 {len(keyword_sources)} 个片段",
            "ms": took(),
            "candidates": _trace_candidates(
                keyword_sources,
                score_label="BM25",
                score_of=lambda source, _key: -source.distance,
            ),
        }
        hint_novels = named_novels or _dominant_novels(
            hierarchy_sources + semantic_sources + keyword_sources
        )
        positional_sources = self.positional_retrieve(
            question, top_k=candidate_k, hint_novels=hint_novels
        )
        if positional_sources:
            yield "step", {
                "step": "结构性召回",
                "stage_key": "position",
                "detail": f"按原文位置召回 {len(positional_sources)} 个片段",
                "ms": took(),
                "candidates": _trace_candidates(
                    positional_sources,
                    score_label="原文位置",
                    score_of=lambda _source, _key: None,
                ),
            }
        recall_detail = f"语义召回 {len(semantic_sources)} 条 · 关键词召回 {len(keyword_sources)} 条"
        if positional_sources:
            ids = sorted(s.chunk_id for s in positional_sources)
            span = f"#{ids[0]}" if len(ids) == 1 else f"#{ids[0]}–{ids[-1]}"
            where = "结尾" if structural == "结局" else "开头" if structural == "开头" else "位置"
            recall_detail += f" · 结构性召回 {len(positional_sources)} 条（定位到{where} {span}）"
        if hierarchy_sources:
            recall_detail += f" · 层级召回映射原文 {len(hierarchy_sources)} 条"
        if not named_novels and hint_novels:
            recall_detail += f"；据此判断问题属于{'、'.join(_display_title(n) for n in hint_novels)}"
        yield "step", {
            "step": "多路召回",
            "stage_key": "recall_summary",
            "detail": recall_detail,
            "ms": 0,
        }

        # Reciprocal Rank Fusion：把多路召回合并成一个候选池。
        # RRF 只看「在各路里排第几」，不看各路的原始分数——因为语义距离和 BM25
        # 分数量纲完全不同，没法直接相加。这让它简单又稳健，但也意味着它对
        # 「这段话到底有没有回答问题」一无所知，那是下面重排要做的事。
        rrf_k = 60
        scores: dict[tuple[str, int], float] = {}
        items: dict[tuple[str, int], SourceChunk] = {}
        for ranked_sources in (
            semantic_sources,
            keyword_sources,
            positional_sources,
            hierarchy_sources,
        ):
            for rank, source in enumerate(ranked_sources, start=1):
                key = (source.novel, source.chunk_id)
                scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + rank)
                items.setdefault(key, source)

        ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
        fused_sources = [items[key] for key in ranked_keys]

        # 阶段四：融合排序
        #
        # detail 里要把「取前 candidate_k 个」写出来。之前只写了候选总数，
        # 界面上就出现了「合并去重后共 38 个候选」紧跟着「对 20 个候选重新打分」
        # 这种数字断层——少掉的 18 个去哪了没人知道。**思考过程的价值就在于
        # 可解释，出现解释不了的数字反而比不展示更糟。**
        yield "step", {
            "step": "融合排序",
            "detail": (
                f"合并去重后共 {len(scores)} 个候选"
                + (
                    f"，按 RRF 分数取前 {candidate_k} 个进入精排"
                    if len(scores) > candidate_k
                    else ""
                )
            ),
            "ms": took(),
            "stage_key": "rrf",
            "candidates": _trace_candidates(
                fused_sources,
                score_label="RRF",
                score_of=lambda _source, key: scores[key],
                selected_count=min(candidate_k, len(fused_sources)),
            ),
        }

        # 阶段五：交叉编码器重排。
        # 前面几路召回都是「粗筛」——快，但只能判断主题相近，判断不了「这段话
        # 是不是真的在回答这个问题」。这里对候选池跑交叉编码器做精排：
        # 它把问题和片段拼在一起送进模型，让两边的词直接做注意力交互。
        # 详见 src/reranker.py 里双编码器 vs 交叉编码器的说明。
        #
        # 注意重排**只能改善排序，救不回没召回到的东西**——如果正确答案根本
        # 不在这 20 个候选里，重排也无能为力。
        candidates = [items[key] for key in ranked_keys[:candidate_k]]
        # 只有重排真正跑成才有归一化分数可用——低置信度信号绝不用向量/BM25
        # 原始分凑数（量纲不同，混用必然得出错误阈值，见 confidence.py）。
        scored_final = None
        if RERANK_ENABLED and len(candidates) > top_k:
            try:
                scored = rerank_with_scores(question, candidates, len(candidates))
                scored_final = scored
                reranked = [source for source, _score in scored]
                rerank_scores = {
                    (source.novel, source.chunk_id): score for source, score in scored
                }
                previous_ranks = {
                    (source.novel, source.chunk_id): rank
                    for rank, source in enumerate(candidates, start=1)
                }
                result = reranked[:top_k]
                yield "step", {
                    "step": "精排",
                    "stage_key": "rerank",
                    "detail": f"用交叉编码器对 {len(candidates)} 个候选重新打分，取最相关的 {len(result)} 段",
                    "ms": took(),
                    "candidates": _trace_candidates(
                        reranked,
                        score_label="CrossEncoder",
                        score_of=lambda _source, key: rerank_scores[key],
                        previous_ranks=previous_ranks,
                        selected_count=len(result),
                    ),
                }
            except Exception as exc:
                # 重排是锦上添花，模型加载失败/推理出错都不该让整个问答挂掉，
                # 退回融合排序的结果即可（质量差一点，但功能可用）。
                result = candidates[:top_k]
                yield "step", {
                    "step": "精排",
                    "stage_key": "rerank",
                    "detail": f"重排不可用，按融合排序取前 {len(result)} 段（{exc}）",
                    "ms": took(),
                    "candidates": _trace_candidates(
                        candidates,
                        score_label="RRF（降级）",
                        score_of=lambda _source, key: scores[key],
                        selected_count=len(result),
                    ),
                }
        else:
            result = candidates[:top_k]

        # ---- M3.4 自适应查询扩展：低置信度时的唯一一次补救 ----------------
        # 挂在重排完成之后：此时才拿得到重排归一化分数，置信度信号才有意义。
        # 开关关闭（默认）时这段完全不执行，主链路行为与从前逐字节一致。
        if QUERY_EXPAND_ENABLED and _allow_expand:
            rescue = self._maybe_expand(
                question, top_k, candidates, scored_final, took
            )
            try:
                while True:
                    yield "step", next(rescue)
            except StopIteration as stop:
                # 生成器 return 的值放在 StopIteration.value 里——补救成功时
                # 是重排后的最终结果；没触发/失败时是 None，保持原 result。
                if stop.value is not None:
                    result = stop.value

        yield "result", result

    def _maybe_expand(
        self,
        question: str,
        top_k: int,
        candidates: list[SourceChunk],
        scored_final: list[tuple[SourceChunk, float]] | None,
        took,
    ):
        """低置信度补救：生成变体 → 逐个检索 → 合并去重 → 重排一次。

        写成生成器而不是普通方法：中间要往 trace 里推步骤（触发原因、变体、
        耗时），而最终结果通过 ``return`` 交给调用方（见上面的 StopIteration
        消费模式），避免用可变容器在两层之间传来传去。

        **整个问答最多补救一次**由两道闸保证：
        - 本方法只在 ``_allow_expand=True`` 时被调用；
        - 变体检索走 retrieve_hybrid_stream(..., _allow_expand=False)，
          内部不可能再进入本方法。
        """
        # 没有重排分数就没有可信信号（重排被关/失败/候选不足时）。宁可放弃
        # 补救也不用向量/BM25 原始分凑合——错误地触发扩展比不触发更糟。
        if scored_final is None:
            return None

        signals = compute_confidence(question, scored_final[:top_k])
        if not signals["is_low_confidence"]:
            return None  # 置信度正常：不花一次 LLM 调用，不加任何延迟

        generate_fn = getattr(self, "expand_generate_fn", None)
        if generate_fn is None:
            # 开了开关但 Web 层没注入生成函数（比如独立脚本环境）：明确记一条，
            # 不静默——「为什么开了没生效」这类问题必须能从 trace 里看出来。
            yield {
                "step": "查询扩展",
                "stage_key": "expand",
                "stage": "expand",
                "detail": (
                    f"检测到低置信度（{('、'.join(signals['low_signals']))}），"
                    "但当前环境没有可用的生成后端，跳过补救"
                ),
                "ms": took(),
                "reasons": signals["low_signals"],
                "variants": [],
            }
            return None

        errors: list[str] = []
        variants = expand_query_variants(
            question,
            generate_fn,
            max_variants=QUERY_EXPAND_MAX_VARIANTS,
            errors=errors,
        )
        yield {
            "step": "查询扩展",
            "stage_key": "expand",
            "stage": "expand",
            "detail": (
                f"低置信度（{'、'.join(signals['low_signals'])}），"
                + (
                    f"生成 {len(variants)} 个改写变体补充检索：{'；'.join(variants)}"
                    if variants
                    else f"未能生成可用变体（{'; '.join(errors) or '模型输出为空'}）"
                )
            ),
            "ms": took(),
            "reasons": signals["low_signals"],
            "variants": variants,
        }
        if not variants:
            return None

        # 对每个变体复用完整混合检索主链路。top_k 取候选池规模而不是最终条数：
        # 反正后面还要对合并后的池子整体重排一次，多捞几个给重排留挑选余地。
        merged: dict[tuple[str, int], SourceChunk] = {
            (c.novel, c.chunk_id): c for c in candidates
        }
        for variant in variants:
            for kind, payload in self.retrieve_hybrid_stream(
                variant, top_k=RECALL_K, _allow_expand=False
            ):
                if kind == "result":
                    for chunk in payload:
                        merged.setdefault((chunk.novel, chunk.chunk_id), chunk)
                    break  # 变体的 trace 步骤不并入主 trace，避免刷屏

        pool = list(merged.values())
        final_scored: list[tuple[SourceChunk, float]] | None = None
        detail_extra = ""
        try:
            # 合并去重后只重排这一次。原始问题的候选也在池子里，所以坏变体
            # 顶多稀释候选池，不可能把原始结果挤出最终 top-k 之外。
            final_scored = rerank_with_scores(question, pool, len(pool))
            rescued = [chunk for chunk, _score in final_scored[:top_k]]
            detail_extra = f"合并去重后共 {len(pool)} 个候选，重排取前 {len(rescued)} 段"
        except Exception as exc:
            # 补救链路上任何一步失败都退回原始结果——补救不能让回答变得
            # 比不做补救更差。
            rescued = None
            detail_extra = f"补救重排失败，保留原结果（{exc}）"

        still_no_evidence: bool | None = None
        if final_scored is not None:
            # 在线没有标准答案，「是否仍无证据」只能近似：补救后信号仍然低
            # 就如实记录 True，供评测脚本和排查参考。仍无证据时不做任何额外
            # 动作，与现有的无证据拒答逻辑保持一致。
            still_no_evidence = compute_confidence(
                question, final_scored[:top_k]
            )["is_low_confidence"]

        yield {
            "step": "扩展重排",
            "stage_key": "expand_rerank",
            "stage": "expand",
            "detail": (
                detail_extra
                + (
                    "；仍未找到可靠证据，按现有拒答逻辑处理"
                    if still_no_evidence
                    else ""
                )
            ),
            "ms": took(),
            "still_no_evidence": still_no_evidence,
        }
        if rescued is None:
            return None
        return rescued
