"""RAG 核心：多路召回、融合、重排、上下文组装和本地生成。

这个模块刻意用普通 Python 函数显式编排，而不是交给 LangGraph：当前请求是一条
短生命周期、方向固定的流水线，没有工具调用循环、人工审批或失败后跨进程恢复的
需求。学习时可以直接沿着 ``retrieve_hybrid_stream`` 阅读每个阶段的数据变化。

核心对象在各阶段的变化如下，阅读时注意“召回候选”和“最终上下文”不是一回事：

    question
      → semantic / BM25 / positional 三路 SourceChunk 候选
      → RRF 去重融合后的候选池
      → CrossEncoder 重排后的 top-k
      → expand_neighbors 补齐相邻片段
      → 带 [n] 编号的 prompt
      → 模型 token 流

召回负责“别漏掉”，重排负责“把正确答案提到前面”，邻居扩展负责“别让切分边界
截断证据”。把三者混为一个步骤，会很难判断检索质量究竟坏在哪一层。

Web 层和云端模型路由在 ``backend/main.py``；这里不依赖 FastAPI，因此评测脚本
可以直接调用检索逻辑。完整选型理由见 ``docs/architecture-decisions.md``。
"""
import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass

import requests
from sentence_transformers import SentenceTransformer

from embedder import load_embedder
from config import (
    BM25_B,
    BM25_K1,
    CONTEXT_NEIGHBORS,
    FULL_TEXT_MAX_CHARS,
    RERANK_CANDIDATE_MULTIPLIER,
    RERANK_ENABLED,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    RECALL_K,
    TOP_K,
)
from graph import detect_relation_question, format_graph_hint
from postgres import connect, has_index, query_relations, vector_literal
from reranker import rerank
from tokenizer import query_terms


PROMPT_TEMPLATE = """你是一个小说问答助手。请仅根据下面提供的编号原文片段回答问题。
如果片段中没有足够信息回答，请明确说“根据提供的片段无法确定”，不要编造内容。

引用要求：
- 每个来自原文的关键事实后标注支持它的片段编号，例如“顾长风中了蚀骨散[2]”。
- 只能使用下面真实存在的编号，不能编造引用。
- 如果一句话由多个片段共同支持，可以写成[1][3]。
- 不要把编号写成脚注列表；直接放在对应事实后，方便用户点击核对。

原文片段：
{context}

问题：{question}

回答："""


def generate_ollama_prompt_stream(
    prompt: str, model: str = OLLAMA_MODEL
) -> Iterator[str]:
    """把已经构造好的 prompt 交给 Ollama，并逐 token 返回。

    独立成模块函数是为了让“自由问答”在书架尚未建立索引、无法创建 NovelRAG
    实例时仍能使用本地模型。NovelRAG.generate_stream 也复用它，避免维护两套协议。
    """
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


def _novel_titles(novel: str) -> list[str]:
    """从库里的书名（文件名）提取用户可能说出的标题。

    文件名形如"《凡人修仙传》（校对版全本+番外）作者：忘语"，用户只会说"凡人修仙传"。
    """
    inner = re.findall(r"《([^》]+)》", novel)
    titles = [name.strip() for name in inner if name.strip()]
    if not titles:
        # 没有书名号就退化为用文件名主体（截断，避免整串带作者名匹配不上）
        titles = [novel.split("（")[0].split("作者")[0].strip()]
    return [t for t in titles if t]


def _edit_distance(a: str, b: str, limit: int) -> int:
    """两字符串的 Levenshtein 距离；一旦确定超过 limit 就提前返回 limit + 1。

    书名很短（通常 3~6 字），这里用滚动数组的朴素实现足够快，不引入额外依赖。
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            current[j] = min(
                previous[j] + 1,  # 删除
                current[j - 1] + 1,  # 插入
                previous[j - 1] + (ca != cb),  # 替换
            )
        # 整行都超过阈值，后面只会更大，可以提前结束
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _fuzzy_contains(question: str, title: str) -> bool:
    """问题里是否有一个与 title 近似的片段（容忍少量错字）。

    用户常打出同音错字，例如把"诡秘之主"打成"闺蜜之主"（guǐ mì / guī mì）。
    精确子串匹配会失败，进而把问题归到别的书上。这里在问题里滑动一个与书名
    等长的窗口，只要某个窗口与书名的编辑距离在容差内就算提到了这本书。

    容差按标题长度取：3 字标题最多错 1 个字，4 字及以上最多错 2 个字
    （"闺蜜之主"→"诡秘之主"就错了 2 个字）。距离上限约为标题长度的一半，
    不同的书之间差异远大于此，不会互相误判。
    """
    n = len(title)
    if n < 3:
        # 标题太短，模糊匹配极易误判，只接受精确包含
        return title in question
    tolerance = 1 if n == 3 else 2
    # 窗口长度允许有 ±tolerance 的浮动，覆盖多字/漏字的情况
    for width in range(max(3, n - tolerance), n + tolerance + 1):
        for start in range(0, len(question) - width + 1):
            window = question[start : start + width]
            if _edit_distance(window, title, tolerance) <= tolerance:
                return True
    return False


def _mentions_novel(question: str, novel: str) -> bool:
    """判断问题里是否提到了这本书（先精确匹配，失败再容错匹配错字）。"""
    titles = _novel_titles(novel)
    if any(title in question for title in titles):
        return True
    return any(_fuzzy_contains(question, title) for title in titles)


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


def _strip_novel_titles(question: str, novels: list[str]) -> str:
    """把已经识别出来的书名从问题里去掉，只留下真正要检索的内容。

    **为什么只对 BM25 这一路做，不对语义检索做**：书名的职责是「路由」——
    确定要在哪本书里搜。一旦已经靠它把范围限定到《凡人修仙传》，再拿「凡人」
    「修仙」这两个词去这本书内部做关键词匹配就是纯噪声：整本书都在讲凡人修仙，
    这两个词对区分书内的哪一段毫无价值。

    实测过这个 bug 的代价：问「《凡人修仙传》里，韩立小时候的绰号是什么」时，
    书名切出的「凡人」「修仙」两个词给某个无关片段白送了 14.1 分
    （凡人 7.73 + 修仙 6.40），而真正的关键词「绰号」只贡献 7.24 分——
    结果无关片段以 17.63 : 10.13 压过了正确答案所在的片段。

    语义检索不受这个影响：它编码的是整句话的含义，书名只是让语义更完整的
    上下文，不会像 BM25 那样被拆成独立的词各自累加分数。
    """
    stripped = question
    for novel in novels:
        for title in _novel_titles(novel):
            stripped = stripped.replace(f"《{title}》", " ").replace(title, " ")
    return stripped


def _display_title(novel: str) -> str:
    """把库里的文件名式书名压成用户认得的短标题，如《诡秘之主》。"""
    titles = _novel_titles(novel)
    return f"《{titles[0]}》" if titles else novel


def _named_via_typo(question: str, novel: str) -> bool:
    """这本书是靠错字容错匹配上的（而非精确出现在问题里）——用于思考过程里提示'已纠正错字'。"""
    titles = _novel_titles(novel)
    exact = any(title in question for title in titles)
    return (not exact) and any(_fuzzy_contains(question, t) for t in titles)


def _structural_kind(question: str) -> str | None:
    """判断是不是在问书的结构位置：返回 '结局' / '开头' / None。

    与 positional_retrieve 里的词表保持一致，只是这里对外给出可读的类别名。
    """
    text = question.strip()
    if any(w in text for w in ("结局", "结尾", "最后", "最终", "收尾", "大结局", "结束")):
        return "结局"
    if any(w in text for w in ("开头", "开篇", "最初", "一开始", "起初", "开始时")):
        return "开头"
    return None


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
    # 章节识别是增强元数据：旧索引和无规范标题的 txt 都可能为空。
    chapter_title: str | None = None
    # Contextual Retrieval 生成的上下文说明（没做增强时是空串）。
    # 重排要用它（见 reranker.rerank 里 indexed_text 的说明），
    # 但 build_prompt 只用 text——不把 AI 生成的说明当原文依据给模型。
    context: str = ""

    @property
    def indexed_text(self) -> str:
        """建索引时用的文本，也是重排该看到的文本。

        必须和索引保持一致：索引的是「说明 + 原文」，重排如果只看原文，
        就会把上下文增强的效果整个抵消掉。
        """
        return f"{self.context}\n{self.text}" if self.context else self.text


class NovelRAG:
    def __init__(self, embedder: SentenceTransformer | None = None):
        self.embedder = embedder or load_embedder()
        if not has_index():
            raise RuntimeError("PostgreSQL novel_chunks 表不存在，请先重建索引")

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

    def retrieve_hybrid_stream(self, question: str, top_k: int = TOP_K):
        """检索流水线的生成器版本：每完成一个阶段就 yield 一次，最后 yield 结果。

        **为什么要做成生成器**：整条流水线要 2 秒左右（交叉编码器重排占大头），
        一次性返回的话用户在这 2 秒里什么都看不到，只能干等。改成边跑边报之后，
        界面可以像成熟的 AI 应用那样把步骤一条条点亮——**等待时间没变，但
        用户知道系统在干什么**，感受完全不同。

        顺带解决了一个真实的性能问题：原来接口端在 `async def` 里同步调用检索，
        这 2 秒会**阻塞整个事件循环**（其他请求全卡住，连断连检查都跑不了）。
        改成生成器后，接口端可以用 `run_in_threadpool` 逐步取，每取一次就是
        一次让出控制权的机会——和下面消费模型 token 用的是同一套模式。

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
        yield "step", {"step": "理解问题", "detail": detail, "ms": took()}

        # 阶段二：确定检索范围
        if named_novels:
            scope_detail = f"只在{'、'.join(_display_title(n) for n in named_novels)}内检索"
        else:
            scope_detail = "在全部书里检索"
        yield "step", {"step": "检索范围", "detail": scope_detail, "ms": took()}

        # 阶段三：多路召回
        semantic_sources = self.retrieve(
            question, top_k=candidate_k, only_novels=named_novels
        )
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
        hint_novels = named_novels or _dominant_novels(
            semantic_sources + keyword_sources
        )
        positional_sources = self.positional_retrieve(
            question, top_k=candidate_k, hint_novels=hint_novels
        )
        recall_detail = f"语义召回 {len(semantic_sources)} 条 · 关键词召回 {len(keyword_sources)} 条"
        if positional_sources:
            ids = sorted(s.chunk_id for s in positional_sources)
            span = f"#{ids[0]}" if len(ids) == 1 else f"#{ids[0]}–{ids[-1]}"
            where = "结尾" if structural == "结局" else "开头" if structural == "开头" else "位置"
            recall_detail += f" · 结构性召回 {len(positional_sources)} 条（定位到{where} {span}）"
        if not named_novels and hint_novels:
            recall_detail += f"；据此判断问题属于{'、'.join(_display_title(n) for n in hint_novels)}"
        yield "step", {"step": "多路召回", "detail": recall_detail, "ms": took()}

        # Reciprocal Rank Fusion：把多路召回合并成一个候选池。
        # RRF 只看「在各路里排第几」，不看各路的原始分数——因为语义距离和 BM25
        # 分数量纲完全不同，没法直接相加。这让它简单又稳健，但也意味着它对
        # 「这段话到底有没有回答问题」一无所知，那是下面重排要做的事。
        rrf_k = 60
        scores: dict[tuple[str, int], float] = {}
        items: dict[tuple[str, int], SourceChunk] = {}
        for ranked_sources in (semantic_sources, keyword_sources, positional_sources):
            for rank, source in enumerate(ranked_sources, start=1):
                key = (source.novel, source.chunk_id)
                scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + rank)
                items.setdefault(key, source)

        ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)

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
        if RERANK_ENABLED and len(candidates) > top_k:
            try:
                result = rerank(question, candidates, top_k)
                yield "step", {
                    "step": "精排",
                    "detail": f"用交叉编码器对 {len(candidates)} 个候选重新打分，取最相关的 {len(result)} 段",
                    "ms": took(),
                }
            except Exception as exc:
                # 重排是锦上添花，模型加载失败/推理出错都不该让整个问答挂掉，
                # 退回融合排序的结果即可（质量差一点，但功能可用）。
                result = candidates[:top_k]
                yield "step", {
                    "step": "精排",
                    "detail": f"重排不可用，按融合排序取前 {len(result)} 段（{exc}）",
                    "ms": took(),
                }
        else:
            result = candidates[:top_k]

        yield "result", result

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

    def build_prompt(self, question: str, sources: list[SourceChunk]) -> str:
        """拼装检索片段 + 问题成完整 prompt。Ollama 和其他生成后端（如 Claude CLI）共用。

        问到人物关系时，会在原文片段前面加一段「图线索」——那是从全书共现统计
        推断出来的关系列表，用来补足 top-k 片段覆盖不到的部分（见 graph.py）。
        线索明确标注了"是统计推断不是确定事实"，让模型拿它当线索去核对原文，
        而不是直接照抄。
        """
        blocks = []
        for index, source in enumerate(sources, start=1):
            location = f"《{_display_title(source.novel).strip('《》')}》"
            if source.chapter_title:
                location += f" · {source.chapter_title}"
            location += f" · 片段 #{source.chunk_id}"
            blocks.append(f"[{index}] {location}\n{source.text}")
        context = "\n\n---\n\n".join(blocks)
        hint = self._graph_hint(question)
        if hint:
            context = f"{hint}\n\n---\n\n{context}"
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def _graph_hint(self, question: str) -> str:
        """问到人物关系时，从图里查一份补充线索；其余情况返回空串。

        图检索是**补充而不是替代**：普通问题走原来的多路召回就好，
        没必要多查一次图。任何一步失败都退回空串，不影响正常问答。
        """
        relation = detect_relation_question(question)
        if not relation:
            return ""
        try:
            # 问题里提到的人物名——从图里已有的人物名反查，避免再做一次分词
            with connect() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT person_a AS name FROM character_relations "
                    "UNION SELECT DISTINCT person_b FROM character_relations"
                ).fetchall()
            subjects = [r["name"] for r in rows if r["name"] in question]
            if not subjects:
                return ""
            # 问题里可能提到多个人名，取最长的那个（最具体）
            subject = max(subjects, key=len)
            neighbors = query_relations(subject, relation)
            return format_graph_hint(subject, relation, neighbors)
        except Exception:
            # 图表可能不存在（没开 GRAPH_ENABLED 建过图），静默跳过即可——
            # 这是纯增强功能，缺了只是回到没有图检索的状态
            return ""

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
        yield from generate_ollama_prompt_stream(self.build_prompt(question, sources), model)

    def query(
        self, question: str, top_k: int = TOP_K, model: str = OLLAMA_MODEL
    ) -> tuple[str, list[SourceChunk]]:
        """最小化的“纯向量检索 → Ollama 生成”示例。

        这是方便在 REPL 里讲解基础 RAG 的入口，不是 Web 应用的生产调用链。
        Web 接口使用 ``retrieve_hybrid_stream``，还会经过 BM25、结构性召回、
        RRF、重排和邻居扩展。学习者若从这里调试，要注意两条路径的能力不同。
        """
        sources = self.retrieve(question, top_k)
        answer = self.generate(question, sources, model=model)
        return answer, sources
