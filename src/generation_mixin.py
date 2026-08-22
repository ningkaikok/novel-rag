"""NovelRAG 的生成 Mixin：prompt 组装、图线索和 Ollama 调用。

初学者可以把这里看成 RAG 的“生成层”：检索阶段产出的 ``SourceChunk`` 列表
在这里被拼装成带 [n] 编号的 prompt，再交给本地 Ollama 生成回答——

    PROMPT_TEMPLATE                 引用规范与上下文/问题的骨架文本
    generate_ollama_prompt_stream   把现成 prompt 逐 token 流式发给 Ollama
    build_prompt                    片段 → 编号上下文块（+ 可选人物关系图线索）
    generate / generate_stream      阻塞 / 流式两种调用方式
    query                           最小化的“向量检索 → 生成”教学入口

多路召回、融合与重排的流水线编排见 ``rag.retrieve_hybrid_stream``；
数据类在 ``chunk_model``，书名识别纯函数在 ``novel_match``。
"""
import json
from collections.abc import Iterator

import requests

from config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    TOP_K,
)
from graph import detect_relation_question, format_graph_hint
from postgres import (
    connect,
    query_relations,
)
from chunk_model import SourceChunk
from novel_match import _display_title


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


class GenerationMixin:
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
