"""Agent Lab：不用 LangGraph 实现一个可观察、有限步、只读的工具循环。

这份代码的教学重点不是“让模型随便做事”，而是看清 Agent 的四个组成部分：

    状态（问题、观察、已收集证据）
      → 决策（模型只输出一个 JSON action）
      → 工具执行（Python 白名单 + 参数校验）
      → 新观察写回状态，再进入下一步

循环最多 3～5 步；最后必须调用 ``answer_with_citations``。工具层只读 PostgreSQL，
不会上传、删除或重建索引，因此适合初学者观察 Agent 行为而不承担写操作风险。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from config import AGENT_TOOL_MAX_CHARS
from postgres import connect
from rag import NovelRAG, SourceChunk, _mentions_novel

Planner = Callable[[str], str]
Answerer = Callable[[str], Iterator[str]]


@dataclass
class ToolResult:
    summary: str
    sources: list[SourceChunk] = field(default_factory=list)
    # `summary` 给规划器快速阅读；`facts` 保存可机器校验的事实。
    # 二者不能混为一谈：检索摘要通常只是局部召回，而结构化事实可以声明
    # 自己的覆盖范围（complete / bounded / partial）。
    facts: dict[str, object] = field(default_factory=dict)


def _apply_output_budget(
    sources: list[SourceChunk],
    max_chars: int = AGENT_TOOL_MAX_CHARS,
    center_chunk_id: int | None = None,
) -> tuple[list[SourceChunk], dict]:
    """工具**读到内容之后**的第二道闸：限制单次工具输出的体积（M3.6）。

    第一道闸是参数上限（``radius`` ≤3、``limit`` ≤12），它限制的是"取几段"。
    但片段长度本身是变量——大部头的一章、或几段特别长的原文，条数合法、体积
    照样能撑爆 prompt。步数上限（3~5 步）同样拦不住这个：它管的是次数不是大小。

    ``center_chunk_id`` 决定牺牲顺序：

    - 传了（``read_neighbors``）：中心片段**无条件保留**，再向两侧对称生长，
      每轮纳入更便宜的一侧。截断永远从离中心最远处开始——这和整章扩展
      （``retrieval_mixins._expand_chapters``）是同一套策略，理由也一样：
      用户要的是"这一段前后"，最远端的邻居本来就是最可有可无的
    - 没传（``get_chapter``）：按原文顺序保留前面的，从末尾开始丢

    返回 ``(保留的片段, trace)``；``trace`` 如实记录丢了几段、为什么丢——
    截断不写进 trace 就是静默丢证据，比不截断更危险。
    """

    def _cost(source: SourceChunk) -> int:
        return len(source.text or "")

    total = sum(_cost(s) for s in sources)
    if total <= max_chars or not sources:
        return sources, {
            "budget_chars": max_chars,
            "chars": total,
            "dropped": 0,
            "truncated": False,
            "reason": "未截断",
        }

    if center_chunk_id is None:
        kept: list[SourceChunk] = []
        used = 0
        for source in sources:
            if kept and used + _cost(source) > max_chars:
                break
            kept.append(source)  # 至少留一段，压成空的等于这次工具调用白跑
            used += _cost(source)
    else:
        ids = [s.chunk_id for s in sources]
        center = ids.index(center_chunk_id) if center_chunk_id in ids else len(ids) // 2
        kept = [sources[center]]
        used = _cost(sources[center])
        left, right = center - 1, center + 1
        while left >= 0 or right < len(sources):
            left_cost = _cost(sources[left]) if left >= 0 else None
            right_cost = _cost(sources[right]) if right < len(sources) else None
            # 每轮纳入更便宜的一侧，直到两侧都放不下
            if left_cost is not None and (right_cost is None or left_cost <= right_cost):
                pick, left = left, left - 1
            elif right_cost is not None:
                pick, right = right, right + 1
            else:
                break
            if used + _cost(sources[pick]) > max_chars:
                break
            kept.append(sources[pick])
            used += _cost(sources[pick])
        kept.sort(key=lambda s: s.chunk_id)

    dropped = len(sources) - len(kept)
    return kept, {
        "budget_chars": max_chars,
        "chars": used,
        "dropped": dropped,
        "truncated": True,
        "reason": (
            f"原文共 {total} 字，超过单次工具输出 {max_chars} 字预算，"
            f"丢弃{'离中心最远的' if center_chunk_id is not None else '末尾的'} {dropped} 段"
        ),
    }


def _summarize_items(items: list[str], max_chars: int = 800) -> str:
    """目录类输出给规划器看的那一行：太长就只列前面几项，并说清楚省了多少。

    只压 ``summary``，**不动 ``facts["items"]``**。这个区分是 ToolResult 的核心
    约定：summary 是给规划器快速阅读的，facts 是可机器校验的完整事实——目录问题
    的确定性回答（``_catalog_answer``）依赖后者，压了它就会把"共有几部"答错。
    """
    if not items:
        return "（空）"
    kept: list[str] = []
    used = 0
    for item in items:
        if kept and used + len(item) + 1 > max_chars:
            break
        kept.append(item)
        used += len(item) + 1
    text = "、".join(kept)
    if len(kept) < len(items):
        text += f"…（共 {len(items)} 项，这里只列出前 {len(kept)} 项）"
    return text


class AgentToolbox:
    """五个只读工具的显式注册表；工具名之外的 action 一律拒绝。"""

    def __init__(self, rag: NovelRAG):
        self.rag = rag

    def query_library(
        self,
        *,
        domain: str = "books",
        operation: str = "list",
        novel: str | None = None,
        chapter: str | None = None,
        limit: int = 100,
    ) -> ToolResult:
        """查询小说的结构化目录；参数是受限 DSL，不接受任意 SQL。

        同一个能力接口可以覆盖书籍、章节和片段统计，避免为每一种问法增加工具。
        返回的事实声明为 complete，表示查询范围内的数据来自数据库聚合结果，而不是
        top-k 召回片段。
        """
        allowed_domains = {"books", "chapters", "chunks"}
        allowed_operations = {"list", "count"}
        if domain not in allowed_domains or operation not in allowed_operations:
            raise ValueError("query_library 的 domain/operation 不受支持")
        limit = max(1, min(int(limit), 100))
        with connect() as conn:
            if domain == "books":
                if operation == "count":
                    row = conn.execute(
                        "SELECT COUNT(DISTINCT novel) AS total FROM novel_chunks"
                    ).fetchone()
                    total = int(row["total"]) if row else 0
                    return ToolResult(
                        f"书架共有 {total} 部小说",
                        facts={
                            "kind": "library_query",
                            "coverage": "complete",
                            "domain": domain,
                            "operation": operation,
                            "total": total,
                            "items": [],
                        },
                    )
                rows = conn.execute(
                    "SELECT DISTINCT novel FROM novel_chunks ORDER BY novel LIMIT %s",
                    (limit + 1,),
                ).fetchall()
                complete = len(rows) <= limit
                items = [row["novel"] for row in rows[:limit]]
                return ToolResult(
                    "书架包含：" + _summarize_items(items),
                    facts={
                        "kind": "library_query",
                        "coverage": "complete" if complete else "bounded",
                        "domain": domain,
                        "operation": operation,
                        "total": len(items),
                        "items": items,
                    },
                )

            clauses: list[str] = []
            params: list[object] = []
            if novel:
                clauses.append("novel = %s")
                params.append(novel)
            if chapter and domain == "chunks":
                clauses.append("chapter_title = %s")
                params.append(chapter)
            where = " WHERE " + " AND ".join(clauses) if clauses else ""

            if domain == "chapters":
                if operation == "count":
                    row = conn.execute(
                        f"SELECT COUNT(DISTINCT chapter_title) AS total FROM novel_chunks{where}",
                        tuple(params),
                    ).fetchone()
                    total = int(row["total"]) if row else 0
                    return ToolResult(
                        f"符合条件的章节共有 {total} 章",
                        facts={
                            "kind": "library_query",
                            "coverage": "complete",
                            "domain": domain,
                            "operation": operation,
                            "total": total,
                            "items": [],
                            "novel": novel,
                        },
                    )
                rows = conn.execute(
                    f"SELECT novel, chapter_title, COUNT(*) AS chunk_count "
                    f"FROM novel_chunks{where} GROUP BY novel, chapter_title "
                    f"ORDER BY novel, MIN(chunk_id) LIMIT %s",
                    tuple(params) + (limit + 1,),
                ).fetchall()
                complete = len(rows) <= limit
                items = [
                    {
                        "novel": row["novel"],
                        "chapter_title": row["chapter_title"],
                        "chunk_count": int(row["chunk_count"]),
                    }
                    for row in rows[:limit]
                ]
                return ToolResult(
                    f"读取到 {len(items)} 个章节",
                    facts={
                        "kind": "library_query",
                        "coverage": "complete" if complete else "bounded",
                        "domain": domain,
                        "operation": operation,
                        "total": len(items),
                        "items": items,
                        "novel": novel,
                    },
                )

            if operation == "count":
                row = conn.execute(
                    f"SELECT COUNT(*) AS total FROM novel_chunks{where}",
                    tuple(params),
                ).fetchone()
                total = int(row["total"]) if row else 0
                return ToolResult(
                    f"符合条件的片段共有 {total} 段",
                    facts={
                        "kind": "library_query",
                        "coverage": "complete",
                        "domain": domain,
                        "operation": operation,
                        "total": total,
                        "items": [],
                        "novel": novel,
                        "chapter": chapter,
                    },
                )
            raise ValueError("chunks 目前只支持 count 操作")

    def list_books(self) -> ToolResult:
        """兼容旧 action；新规划应使用 query_library(domain=books)。"""
        result = self.query_library(domain="books", operation="list")
        items = list(result.facts.get("items", []))
        result.facts = {
            "kind": "book_catalog",
            "coverage": result.facts.get("coverage", "complete"),
            "book_count": len(items),
            "books": items,
        }
        return result

    def _resolve_novel(self, requested: str) -> str:
        """把规划器给的书名字符串解析成库里真实的 novel 主键。

        规划器经常把用户原话里的书名（可能是错字）原样填进 `novel` 参数，
        比如用户问"闺蜜之主"，规划器就会传 `novel="闺蜜之主"`。子串匹配对这种
        情况必然失败——"闺蜜之主"不是"《诡秘之主》…"的子串。主对话链路
        （`rag.py` 的 `_named_novels`）用带编辑距离容差的 `_mentions_novel`
        处理过同样的问题，这里复用同一套逻辑，而不是让工具直接报错——
        两套代码解决同一个"书名打错字"问题却给出不同结果，会让 Agent
        没道理地卡在这一步（实测：规划器反复重试同一个坏参数，白白烧掉
        步数预算，见 tests/backend/test_agent_lab.py 里的对应用例）。
        """
        with connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT novel FROM novel_chunks ORDER BY novel"
            ).fetchall()
        novels = [row["novel"] for row in rows]
        exact = [novel for novel in novels if novel == requested]
        if exact:
            return exact[0]
        fuzzy = [
            novel
            for novel in novels
            if requested.casefold() in novel.casefold()
            or novel.casefold() in requested.casefold()
        ]
        if not fuzzy:
            fuzzy = [novel for novel in novels if _mentions_novel(requested, novel)]
        if len(fuzzy) == 1:
            return fuzzy[0]
        raise ValueError(f"无法唯一确定小说：{requested}")

    def search_novels(
        self, query: str, novel: str | None = None, limit: int = 5
    ) -> ToolResult:
        scoped_query = f"《{novel}》{query}" if novel else query
        sources = self.rag.retrieve_hybrid(scoped_query, top_k=max(1, min(limit, 8)))
        return ToolResult(
            f"检索到 {len(sources)} 个相关原文片段",
            sources,
            facts={
                "kind": "novel_passages",
                "coverage": "partial",
                "matched_count": len(sources),
            },
        )

    def read_neighbors(self, novel: str, chunk_id: int, radius: int = 1) -> ToolResult:
        resolved = self._resolve_novel(novel)
        radius = max(0, min(int(radius), 3))
        with connect() as conn:
            rows = conn.execute(
                "SELECT novel, chunk_id, chapter_title, text, context "
                "FROM novel_chunks WHERE novel = %s AND chunk_id BETWEEN %s AND %s "
                "ORDER BY chunk_id",
                (resolved, max(0, int(chunk_id) - radius), int(chunk_id) + radius),
            ).fetchall()
        sources, budget = _apply_output_budget(
            [_row_to_source(row) for row in rows], center_chunk_id=int(chunk_id)
        )
        summary = f"读取片段 #{chunk_id} 前后共 {len(sources)} 段"
        if budget["truncated"]:
            summary += f"（{budget['reason']}）"
        return ToolResult(
            summary,
            sources,
            facts={
                "kind": "neighbor_context",
                "coverage": "bounded",
                "novel": resolved,
                "center_chunk_id": int(chunk_id),
                "returned_count": len(sources),
                "budget": budget,
            },
        )

    def get_chapter(self, novel: str, chapter_title: str, limit: int = 8) -> ToolResult:
        resolved = self._resolve_novel(novel)
        with connect() as conn:
            rows = conn.execute(
                "SELECT novel, chunk_id, chapter_title, text, context "
                "FROM novel_chunks WHERE novel = %s AND chapter_title ILIKE %s "
                "ORDER BY chunk_id LIMIT %s",
                (resolved, f"%{chapter_title}%", max(1, min(int(limit), 12))),
            ).fetchall()
        sources, budget = _apply_output_budget([_row_to_source(row) for row in rows])
        summary = f"章节“{chapter_title}”读取到 {len(sources)} 段原文"
        if budget["truncated"]:
            summary += f"（{budget['reason']}）"
        return ToolResult(
            summary,
            sources,
            facts={
                "kind": "chapter_passages",
                "coverage": "bounded",
                "novel": resolved,
                "chapter_title": chapter_title,
                "returned_count": len(sources),
                "limit": max(1, min(int(limit), 12)),
                "budget": budget,
            },
        )

    def execute(self, name: str, args: dict) -> ToolResult:
        if name == "query_library":
            return self.query_library(
                domain=str(args.get("domain", "books")),
                operation=str(args.get("operation", "list")),
                novel=str(args["novel"]) if args.get("novel") else None,
                chapter=str(args["chapter"]) if args.get("chapter") else None,
                limit=int(args.get("limit", 100)),
            )
        if name == "list_books":
            return self.list_books()
        if name == "search_novels":
            return self.search_novels(
                str(args.get("query", "")),
                str(args["novel"]) if args.get("novel") else None,
                int(args.get("limit", 5)),
            )
        if name == "read_neighbors":
            return self.read_neighbors(
                str(args.get("novel", "")),
                int(args.get("chunk_id", 0)),
                int(args.get("radius", 1)),
            )
        if name == "get_chapter":
            return self.get_chapter(
                str(args.get("novel", "")),
                str(args.get("chapter_title", "")),
                int(args.get("limit", 8)),
            )
        raise ValueError(f"未知或不允许的工具：{name}")


def _row_to_source(row: dict) -> SourceChunk:
    return SourceChunk(
        novel=row["novel"],
        chunk_id=int(row["chunk_id"]),
        text=row["text"],
        distance=0.0,
        chapter_title=row.get("chapter_title"),
        context=row.get("context") or "",
    )


_PLANNER_PROMPT = """你是小说 RAG 的工具规划器。一次只选择一个工具，不要直接回答。
可用工具：
- query_library: {{"domain": "books|chapters|chunks", "operation": "list|count", "novel": "可选", "chapter": "可选", "limit": 1到100}}
- list_books: {{}}（兼容旧 action，优先使用 query_library）
- search_novels: {{"query": "检索问题", "novel": "可选书名", "limit": 1到8}}
- read_neighbors: {{"novel": "书名", "chunk_id": 片段号, "radius": 0到3}}
- get_chapter: {{"novel": "书名", "chapter_title": "章节名", "limit": 1到12}}
- answer_with_citations: {{"source_ids": ["S1", "S2"]}}

规则：
1. 没有证据时不能 answer；先搜索或列书。
2. 证据上下文不完整时可以读相邻片段或整章。
3. 证据足够就选择 answer_with_citations，不要无意义重复搜索。
4. “一共/全部/有哪些/多少”等全集问题必须选择能声明 `coverage=complete` 的工具；
   搜索片段是 partial，不能用召回数量推断全集数量。
5. 只输出一个 JSON 对象：
   {{"reason": "一句可展示的理由", "tool": "工具名", "args": {{...}}}}

用户问题：{question}
已有观察：
{observations}
"""


# 这里识别的是“问题需要什么覆盖范围”，不是为某一句用户话术写分支。
# 后续增加人物目录、章节目录等完整工具时，只需给工具返回对应 kind/coverage，
# 不必再把每一种自然语言问法硬编码进最终回答逻辑。
_CATALOG_QUESTION_RE = re.compile(
    r"(?:一共有|共有|总共有|多少部|几部|多少本|几本|有哪些小说|哪些书|所有小说|全部小说|书架)"
)
_EXHAUSTIVE_MARKER_RE = re.compile(
    r"(?:全部|所有|完整|一共|总共|共有|有哪些|多少|几部|几本|列出)"
)


def _question_scope(question: str) -> str:
    """返回问题所需的证据覆盖范围：catalog / library / exhaustive / open。"""
    text = question.strip()
    if _CATALOG_QUESTION_RE.search(text) and not re.search(r"章节|片段|段落", text):
        return "catalog"
    if re.search(r"章节|片段|段落", text) and _EXHAUSTIVE_MARKER_RE.search(text):
        return "library"
    if _EXHAUSTIVE_MARKER_RE.search(text):
        return "exhaustive"
    return "open"


def _library_action_for_question(question: str) -> dict[str, object]:
    """为结构化问题选择查询维度，不绑定某个具体自然语言问法。"""
    text = question.strip()
    if re.search(r"片段|段落", text) and re.search(r"多少|几|数量|总", text):
        return {"domain": "chunks", "operation": "count"}
    if re.search(r"章节|章", text):
        return {"domain": "chapters", "operation": "list"}
    return {"domain": "books", "operation": "list"}


def _catalog_answer(question: str, facts: list[dict[str, object]]) -> str | None:
    """用完整目录事实回答目录问题；没有完整事实就返回 None，禁止猜测。"""
    if _question_scope(question) != "catalog":
        return None
    catalog = next(
        (
            item
            for item in reversed(facts)
            if item.get("kind") in {"book_catalog", "library_query"}
            and item.get("coverage") == "complete"
            and (item.get("kind") == "book_catalog" or item.get("domain") == "books")
        ),
        None,
    )
    if catalog is None:
        return None
    books = [str(book) for book in catalog.get("books", catalog.get("items", []))]
    count = int(catalog.get("book_count", catalog.get("total", len(books))))
    if not books:
        return "当前书架中没有已建立索引的小说。"
    return f"当前书架一共有 {count} 部小说：" + "、".join(books) + "。"


def _library_answer(question: str, facts: list[dict[str, object]]) -> str | None:
    """对可直接由完整目录事实计算的问题给出确定答案。"""
    if _question_scope(question) != "library":
        return None
    result = next(
        (
            item
            for item in reversed(facts)
            if item.get("kind") == "library_query" and item.get("coverage") == "complete"
        ),
        None,
    )
    if result is None:
        return None
    domain = result.get("domain")
    operation = result.get("operation")
    total = int(result.get("total", 0))
    if domain == "chunks" and operation == "count":
        return f"符合当前条件的片段共有 {total} 段。"
    if domain == "chapters" and operation == "count":
        return f"符合当前条件的章节共有 {total} 章。"
    if domain == "chapters" and operation == "list":
        items = result.get("items", [])
        names = [
            f"《{item['novel']}》{item['chapter_title']}"
            for item in items
            if isinstance(item, dict) and item.get("chapter_title")
        ]
        return f"共找到 {len(names)} 个章节：" + "、".join(names) + "。"
    return None


def _facts_prompt(facts: list[dict[str, object]]) -> str:
    """把结构化事实附加给回答模型，并明确每条事实的覆盖边界。"""
    if not facts:
        return ""
    payload = json.dumps(facts[-8:], ensure_ascii=False)
    return (
        "\n\n【结构化工具事实】\n"
        f"{payload}\n"
        "事实的 coverage=complete 才能支持全集、总数和全部列表；"
        "partial/bounded 只能支持召回到的局部内容。若问题要求完整范围但没有"
        "complete 事实，必须明确说明无法从当前证据确定，不要把片段数量当总数。"
    )


class ActionParseError(ValueError):
    """动作解析失败，带一个可聚合的失败类别（见 _parse_action 的埋点说明）。"""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def _parse_action(raw: str) -> tuple[dict, str]:
    """容忍模型带 Markdown 围栏，但最终必须得到白名单 action 对象。

    返回 ``(action, parse_mode)``。``parse_mode`` 就是路线图 M3.2.1 要求的那条
    「动作解析失败 / 走正则兜底」埋点：

        strict  裸 JSON，一次解析成功——这是理想情况
        fenced  剥掉 ```json 围栏之后才成功
        regex   连剥围栏都不行，靠正则抓第一个 {...} 兜底
        failed  彻底失败（category 记录失败类型）

    **为什么先埋点而不是直接改协议**：M3.2.1 想把 JSON 动作换成首行标签协议，
    理由是"JSON 收完才能解析、正则兜底是猜、解析失败没有回路"。这三条在道理上
    都成立，但「在自家模型上到底多久出一次」从来没测过。路线图对此写得很明白：
    先用真实失败率决定这件事的排期，不要凭"这个设计更好"就抢跑。

    regex 这一档是最关键的信号：它意味着**解析成功了，但成功得很可疑**——正则
    抓的是第一个花括号，模型在 JSON 前后多写一句带花括号的话就可能抓错对象，
    而且抓错了不会报错，会安安静静地执行一个错误的动作。它的占比比 failed 更
    值得看：failed 至少还会走降级并在界面上写明原因。

    聚合方式见 scripts/agent_parse_stats.py（读 chat_turns.agent_steps）。
    """
    text = raw.strip()
    stripped = text.replace("```json", "").replace("```", "").strip()
    for mode, candidate in (("strict", text), ("fenced", stripped)):
        try:
            return _validated(json.loads(candidate)), mode
        except json.JSONDecodeError:
            continue
        except ActionParseError:
            raise  # JSON 合法但形状不对：换个剥法也不会变对，别掩盖它

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ActionParseError("no_json", "规划器没有返回 JSON")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ActionParseError("invalid_json", f"正则兜底仍不是合法 JSON：{exc}") from None
    return _validated(data), "regex"


def _validated(data: object) -> dict:
    if not isinstance(data, dict) or not isinstance(data.get("args", {}), dict):
        raise ActionParseError("bad_shape", "规划器 action 形状不正确")
    return data


def _observation_text(observations: list[dict]) -> str:
    if not observations:
        return "（尚无）"
    lines = []
    for item in observations[-4:]:
        lines.append(
            f"步骤{item['step']} {item['tool']}：{item['observation']}；"
            f"证据ID={','.join(item.get('source_ids', [])) or '无'}"
        )
    return "\n".join(lines)


def run_agent(
    question: str,
    *,
    rag: NovelRAG,
    planner: Planner,
    answerer: Answerer,
    max_steps: int = 5,
) -> Iterator[tuple[str, object]]:
    """运行有限步工具循环，逐步产出 ``agent_step/sources/token/done`` 事件。"""
    max_steps = max(3, min(int(max_steps), 5))
    toolbox = AgentToolbox(rag)
    observations: list[dict] = []
    source_registry: dict[str, SourceChunk] = {}
    # 与原文证据分开保存；原文可能是 top-k 局部召回，不能代表数据库全集。
    fact_registry: list[dict[str, object]] = []
    seen_actions: set[str] = set()
    # 同一个工具连续失败的次数，只按工具名计数、不看具体参数。
    #
    # 下面的“重复动作”检测按完整 (tool, args) 精确匹配去重，但实测发现规划器
    # 会在同一个坏参数上反复重试，每次只改一个无关紧要的参数（比如
    # read_neighbors 的 radius 从 1 改成 3 再改成 0），核心的坏参数（比如打错的
    # 书名）从没变过。args 一变，signature 就不同，精确匹配的重复检测完全失效，
    # 三步预算里能白白烧掉两三步在一个注定失败的调用上。这里换一个更粗但更管用
    # 的信号：只要同一个工具名连续失败两次，就不再信任这个工具，不管第三次的
    # 参数长什么样。
    tool_failure_streak: dict[str, int] = {}

    for step in range(1, max_steps + 1):
        # 这一步的动作是怎么来的（M3.2.1 埋点）。规划器没有参与的步骤（强制收尾、
        # 目录门禁、重复动作拦截）保持 None——把它们算进失败率会把统计冲淡。
        parse_mode: str | None = None
        if step == max_steps and source_registry:
            action = {
                "reason": "已到最大步骤，使用现有证据回答",
                "tool": "answer_with_citations",
                "args": {"source_ids": list(source_registry)},
            }
        else:
            prompt = _PLANNER_PROMPT.format(
                question=question,
                observations=_observation_text(observations),
            )
            try:
                action, parse_mode = _parse_action(planner(prompt))
            except Exception as exc:
                parse_mode = f"failed:{getattr(exc, 'category', 'unknown')}"
                # 规划输出偶尔不是合法 JSON。无证据先走搜索，有证据就结束回答，
                # 让教学 demo 可用，同时在 reason 中如实显示降级原因。
                action = (
                    {
                        "reason": f"规划格式无效，降级为检索（{exc}）",
                        "tool": "search_novels",
                        "args": {"query": question, "limit": 5},
                    }
                    if not source_registry
                    else {
                        "reason": f"规划格式无效，使用已有证据回答（{exc}）",
                        "tool": "answer_with_citations",
                        "args": {"source_ids": list(source_registry)},
                    }
                )

            # 对需要全集范围的目录问题，第一步固定读取完整目录。
            # 这是通用的“覆盖范围门禁”，不是对某个最终答案字符串做特判。
            if step == 1 and _question_scope(question) in {"catalog", "library"}:
                library_args = _library_action_for_question(question)
                action = {
                    "reason": "这是结构化范围问题，先读取 coverage=complete 的目录事实",
                    "tool": "query_library",
                    "args": library_args,
                }

        tool = str(action.get("tool", ""))
        args = action.get("args") or {}
        reason = str(action.get("reason", "未提供理由"))[:240]
        signature = json.dumps([tool, args], sort_keys=True, ensure_ascii=False)
        stuck = signature in seen_actions or tool_failure_streak.get(tool, 0) >= 2
        if stuck and tool != "answer_with_citations":
            if source_registry:
                tool = "answer_with_citations"
                args = {"source_ids": list(source_registry)}
                reason = "检测到反复失败或重复动作，停止循环并使用已有证据回答"
            else:
                args = {"query": question, "limit": 5}
                tool = "search_novels"
                reason = "检测到反复失败或重复动作，改用原问题检索"
        seen_actions.add(signature)

        if tool == "answer_with_citations":
            requested = [str(value) for value in args.get("source_ids", [])]
            selected = [source_registry[sid] for sid in requested if sid in source_registry]
            if not selected:
                selected = list(source_registry.values())
            if not selected:
                # 没有证据时禁止模型凭空回答；把动作改成搜索并继续循环。
                tool = "search_novels"
                args = {"query": question, "limit": 5}
                reason = "尚无可引用证据，先检索小说原文"
            else:
                yield (
                    "agent_step",
                    {
                        "step": step,
                        "reason": reason,
                        "tool": "answer_with_citations",
                        "args": {"source_ids": requested or list(source_registry)},
                        "observation": f"使用 {len(selected)} 个原文片段生成带引用答案",
                        "source_ids": requested or list(source_registry),
                        "parse_mode": parse_mode,
                    },
                )
                yield "sources", selected
                deterministic = _catalog_answer(question, fact_registry) or _library_answer(
                    question, fact_registry
                )
                if deterministic is not None:
                    # 数据库目录是确定性元数据，不再让模型从 top-k 片段猜总数。
                    yield "token", deterministic
                else:
                    prompt = rag.build_prompt(question, selected) + _facts_prompt(
                        fact_registry
                    )
                    for token in answerer(prompt):
                        yield "token", token
                yield "done", {}
                return

        try:
            result = toolbox.execute(tool, args)
            source_ids: list[str] = []
            for source in result.sources:
                key = next(
                    (
                        sid
                        for sid, existing in source_registry.items()
                        if (existing.novel, existing.chunk_id)
                        == (source.novel, source.chunk_id)
                    ),
                    None,
                )
                if key is None:
                    key = f"S{len(source_registry) + 1}"
                    source_registry[key] = source
                source_ids.append(key)
            if result.facts:
                fact_registry.append(dict(result.facts))
            observation = result.summary
            tool_failure_streak[tool] = 0
        except Exception as exc:
            source_ids = []
            observation = f"工具执行失败：{type(exc).__name__}: {exc}"
            tool_failure_streak[tool] = tool_failure_streak.get(tool, 0) + 1

        event = {
            "step": step,
            "reason": reason,
            "tool": tool,
            "args": args,
            "observation": observation,
            "source_ids": source_ids,
            "parse_mode": parse_mode,
        }
        observations.append(event)
        yield "agent_step", event

    # 理论上只有连续工具失败且没有证据才会到这里。不要再额外虚构一个第 N+1 步；
    # 前端已经展示了最后一次失败观察，这里只给出明确拒答并结束 SSE。
    yield "token", "经过最多五步检索仍没有找到足够的小说原文，因此无法给出有依据的回答。"
    yield "done", {}
