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
from dataclasses import dataclass, field
from typing import Callable, Iterator

from postgres import connect
from rag import NovelRAG, SourceChunk

Planner = Callable[[str], str]
Answerer = Callable[[str], Iterator[str]]


@dataclass
class ToolResult:
    summary: str
    sources: list[SourceChunk] = field(default_factory=list)


class AgentToolbox:
    """五个只读工具的显式注册表；工具名之外的 action 一律拒绝。"""

    def __init__(self, rag: NovelRAG):
        self.rag = rag

    def list_books(self) -> ToolResult:
        with connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT novel FROM novel_chunks ORDER BY novel"
            ).fetchall()
        books = [row["novel"] for row in rows]
        return ToolResult("书架包含：" + ("、".join(books) if books else "（空）"))

    def _resolve_novel(self, requested: str) -> str:
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
        if len(fuzzy) == 1:
            return fuzzy[0]
        raise ValueError(f"无法唯一确定小说：{requested}")

    def search_novels(
        self, query: str, novel: str | None = None, limit: int = 5
    ) -> ToolResult:
        scoped_query = f"《{novel}》{query}" if novel else query
        sources = self.rag.retrieve_hybrid(scoped_query, top_k=max(1, min(limit, 8)))
        return ToolResult(f"检索到 {len(sources)} 个相关原文片段", sources)

    def read_neighbors(
        self, novel: str, chunk_id: int, radius: int = 1
    ) -> ToolResult:
        resolved = self._resolve_novel(novel)
        radius = max(0, min(int(radius), 3))
        with connect() as conn:
            rows = conn.execute(
                "SELECT novel, chunk_id, chapter_title, text, context "
                "FROM novel_chunks WHERE novel = %s AND chunk_id BETWEEN %s AND %s "
                "ORDER BY chunk_id",
                (resolved, max(0, int(chunk_id) - radius), int(chunk_id) + radius),
            ).fetchall()
        sources = [_row_to_source(row) for row in rows]
        return ToolResult(f"读取片段 #{chunk_id} 前后共 {len(sources)} 段", sources)

    def get_chapter(
        self, novel: str, chapter_title: str, limit: int = 8
    ) -> ToolResult:
        resolved = self._resolve_novel(novel)
        with connect() as conn:
            rows = conn.execute(
                "SELECT novel, chunk_id, chapter_title, text, context "
                "FROM novel_chunks WHERE novel = %s AND chapter_title ILIKE %s "
                "ORDER BY chunk_id LIMIT %s",
                (resolved, f"%{chapter_title}%", max(1, min(int(limit), 12))),
            ).fetchall()
        sources = [_row_to_source(row) for row in rows]
        return ToolResult(f"章节“{chapter_title}”读取到 {len(sources)} 段原文", sources)

    def execute(self, name: str, args: dict) -> ToolResult:
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
- list_books: {{}}
- search_novels: {{"query": "检索问题", "novel": "可选书名", "limit": 1到8}}
- read_neighbors: {{"novel": "书名", "chunk_id": 片段号, "radius": 0到3}}
- get_chapter: {{"novel": "书名", "chapter_title": "章节名", "limit": 1到12}}
- answer_with_citations: {{"source_ids": ["S1", "S2"]}}

规则：
1. 没有证据时不能 answer；先搜索或列书。
2. 证据上下文不完整时可以读相邻片段或整章。
3. 证据足够就选择 answer_with_citations，不要无意义重复搜索。
4. 只输出一个 JSON 对象：
   {{"reason": "一句可展示的理由", "tool": "工具名", "args": {{...}}}}

用户问题：{question}
已有观察：
{observations}
"""


def _parse_action(raw: str) -> dict:
    """容忍模型带 Markdown 围栏，但最终必须得到白名单 action 对象。"""
    text = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("规划器没有返回 JSON")
        data = json.loads(match.group(0))
    if not isinstance(data, dict) or not isinstance(data.get("args", {}), dict):
        raise ValueError("规划器 action 形状不正确")
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
    seen_actions: set[str] = set()

    for step in range(1, max_steps + 1):
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
                action = _parse_action(planner(prompt))
            except Exception as exc:
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

        tool = str(action.get("tool", ""))
        args = action.get("args") or {}
        reason = str(action.get("reason", "未提供理由"))[:240]
        signature = json.dumps([tool, args], sort_keys=True, ensure_ascii=False)
        if signature in seen_actions and tool != "answer_with_citations":
            if source_registry:
                tool = "answer_with_citations"
                args = {"source_ids": list(source_registry)}
                reason = "检测到重复动作，停止循环并使用已有证据回答"
            else:
                args = {"query": question, "limit": 5}
                tool = "search_novels"
                reason = "检测到重复动作，改用原问题检索"
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
                yield "agent_step", {
                    "step": step,
                    "reason": reason,
                    "tool": "answer_with_citations",
                    "args": {"source_ids": requested or list(source_registry)},
                    "observation": f"使用 {len(selected)} 个原文片段生成带引用答案",
                    "source_ids": requested or list(source_registry),
                }
                yield "sources", selected
                prompt = rag.build_prompt(question, selected)
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
            observation = result.summary
        except Exception as exc:
            source_ids = []
            observation = f"工具执行失败：{type(exc).__name__}: {exc}"

        event = {
            "step": step,
            "reason": reason,
            "tool": tool,
            "args": args,
            "observation": observation,
            "source_ids": source_ids,
        }
        observations.append(event)
        yield "agent_step", event

    # 理论上只有连续工具失败且没有证据才会到这里。不要再额外虚构一个第 N+1 步；
    # 前端已经展示了最后一次失败观察，这里只给出明确拒答并结束 SSE。
    yield "token", "经过最多五步检索仍没有找到足够的小说原文，因此无法给出有依据的回答。"
    yield "done", {}
