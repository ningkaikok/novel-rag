"""统一的模型路由、显式降级与生成观测。

这层只负责把任务交给 Ollama、Claude CLI 或智谱，不把 prompt、回答或原文
写入日志。回答模型仍由用户在界面选择；查询改写、摘要、查询扩展和 Judge
可以通过 ``MODEL_*`` 环境变量独立选用便宜模型。
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

from backend import claude_cli, zhipu
from config import (
    HISTORY_SUMMARY_MODEL,
    QUERY_EXPAND_MODEL,
    QUERY_REWRITE_MODEL,
)
from generation_mixin import generate_ollama_prompt_stream

ModelTask = Literal["answer", "query_rewrite", "summary", "query_expand", "judge"]
ModelFactory = Callable[[str, str], Iterator[str]]

MODEL_ROUTING_ENABLED = os.environ.get("MODEL_ROUTING_ENABLED", "1") != "0"
MODEL_FALLBACK_MODEL = os.environ.get("MODEL_FALLBACK_MODEL", "").strip()
TASK_MODELS: dict[str, str] = {
    "query_rewrite": os.environ.get("MODEL_QUERY_REWRITE", QUERY_REWRITE_MODEL),
    "summary": os.environ.get("MODEL_SUMMARY", HISTORY_SUMMARY_MODEL),
    "query_expand": os.environ.get("MODEL_QUERY_EXPAND", QUERY_EXPAND_MODEL),
    "judge": os.environ.get("MODEL_JUDGE", ""),
}


@dataclass
class GenerationStats:
    """不含 prompt/回答正文的单次模型调用摘要。"""

    task: str
    requested_model: str
    selected_model: str
    provider: str
    fallback_model: str | None = None
    fallback_used: bool = False
    chunks: int = 0
    characters: int = 0
    elapsed_ms: int | None = None
    completed: bool = False
    error_type: str | None = None

    def snapshot(self) -> dict:
        return {
            "task": self.task,
            "requested_model": self.requested_model,
            "selected_model": self.selected_model,
            "provider": self.provider,
            "fallback_model": self.fallback_model,
            "fallback_used": self.fallback_used,
            "chunks": self.chunks,
            "characters": self.characters,
            "elapsed_ms": self.elapsed_ms,
            "completed": self.completed,
            "error_type": self.error_type,
        }


def resolve_model(task: ModelTask, requested_model: str) -> str:
    """按任务选择模型；回答任务尊重用户当前选择。"""
    if task == "answer" or not MODEL_ROUTING_ENABLED:
        return requested_model
    configured = TASK_MODELS.get(task, "").strip()
    return configured or requested_model


def provider_name(model: str) -> str:
    if model.startswith(claude_cli.MODEL_PREFIX):
        return "claude"
    if model.startswith(zhipu.MODEL_PREFIX):
        return "zhipu"
    return "ollama"


def routing_snapshot(requested_model: str) -> dict:
    """返回可安全落库的路由配置，不包含密钥或正文。"""
    return {
        "enabled": MODEL_ROUTING_ENABLED,
        "requested_answer_model": requested_model,
        "task_models": dict(TASK_MODELS),
        "fallback_model": MODEL_FALLBACK_MODEL or None,
    }


def _default_factory(model: str, prompt: str) -> Iterator[str]:
    if model.startswith(claude_cli.MODEL_PREFIX):
        return claude_cli.generate_stream(prompt, model)
    if model.startswith(zhipu.MODEL_PREFIX):
        return zhipu.generate_stream(prompt, model)
    return generate_ollama_prompt_stream(prompt, model=model)


def generate_stream(
    prompt: str,
    requested_model: str,
    *,
    task: ModelTask = "answer",
    stats: list[GenerationStats] | None = None,
    ollama_factory: Callable[[str, str], Iterator[str]] | None = None,
    claude_factory: Callable[[str, str], Iterator[str]] | None = None,
    zhipu_factory: Callable[[str, str], Iterator[str]] | None = None,
) -> Iterator[str]:
    """生成流；仅在首个 token 之前失败时才使用显式备用模型。

    流已经产出部分内容后不能偷偷切模型，否则会造成重复或语义断裂，因此
    后置异常直接抛出。备用模型为空时行为与原先完全一致。
    """
    selected = resolve_model(task, requested_model)
    fallback = MODEL_FALLBACK_MODEL or None
    if fallback == selected:
        fallback = None
    record = GenerationStats(
        task=task,
        requested_model=requested_model,
        selected_model=selected,
        provider=provider_name(selected),
        fallback_model=fallback,
    )
    if stats is not None:
        stats.append(record)

    factories: dict[str, Callable[[str, str], Iterator[str]]] = {
        "ollama": ollama_factory or _default_factory,
        "claude": claude_factory or _default_factory,
        "zhipu": zhipu_factory or _default_factory,
    }
    started = time.monotonic()
    candidates = [selected, *([fallback] if fallback else [])]
    try:
        for attempt, model in enumerate(candidates):
            produced = False
            try:
                iterator = factories[provider_name(model)](model, prompt)
                for chunk in iterator:
                    if chunk:
                        produced = True
                        record.chunks += 1
                        record.characters += len(chunk)
                    yield chunk
                record.selected_model = model
                record.provider = provider_name(model)
                record.fallback_used = attempt > 0
                record.completed = True
                return
            except Exception as exc:
                record.error_type = type(exc).__name__
                if attempt == 0 and not produced and len(candidates) > 1:
                    record.fallback_used = True
                    continue
                raise
    finally:
        record.elapsed_ms = round((time.monotonic() - started) * 1000)
