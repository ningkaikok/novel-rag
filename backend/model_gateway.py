"""统一的模型路由、显式降级与生成观测。

这层只负责把任务交给 Ollama、Claude CLI、Codex CLI 或智谱，不把 prompt、回答
或原文写入日志。回答模型仍由用户在界面选择；查询改写、摘要、查询扩展和 Judge
可以通过 ``MODEL_*`` 环境变量独立选用便宜模型。
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from math import ceil
from typing import Literal

from backend import claude_cli, codex_cli, zhipu
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
MODEL_CLOUD_ALLOWED = os.environ.get("MODEL_CLOUD_ALLOWED", "1") != "0"
MODEL_MAX_OUTPUT_CHARACTERS = int(os.environ.get("MODEL_MAX_OUTPUT_CHARACTERS", 0))
MODEL_MAX_ESTIMATED_COST_USD = float(os.environ.get("MODEL_MAX_ESTIMATED_COST_USD", 0))
MODEL_CHARS_PER_TOKEN = float(os.environ.get("MODEL_CHARS_PER_TOKEN", 1.5))
MODEL_INPUT_USD_PER_MILLION_TOKENS = float(
    os.environ.get("MODEL_INPUT_USD_PER_MILLION_TOKENS", 0)
)
MODEL_OUTPUT_USD_PER_MILLION_TOKENS = float(
    os.environ.get("MODEL_OUTPUT_USD_PER_MILLION_TOKENS", 0)
)
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
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_cost_usd: float | None = None
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
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
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
    if model.startswith(codex_cli.MODEL_PREFIX):
        return "codex"
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
        "cloud_allowed": MODEL_CLOUD_ALLOWED,
        "max_output_characters": MODEL_MAX_OUTPUT_CHARACTERS or None,
        "max_estimated_cost_usd": MODEL_MAX_ESTIMATED_COST_USD or None,
        "cost_rate_configured": bool(
            MODEL_INPUT_USD_PER_MILLION_TOKENS or MODEL_OUTPUT_USD_PER_MILLION_TOKENS
        ),
    }


def _default_factory(model: str, prompt: str) -> Iterator[str]:
    if model.startswith(claude_cli.MODEL_PREFIX):
        return claude_cli.generate_stream(prompt, model)
    if model.startswith(codex_cli.MODEL_PREFIX):
        return codex_cli.generate_stream(prompt, model)
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
    codex_factory: Callable[[str, str], Iterator[str]] | None = None,
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
        estimated_input_tokens=max(0, ceil(len(prompt) / max(MODEL_CHARS_PER_TOKEN, 0.1))),
    )
    if stats is not None:
        stats.append(record)

    factories: dict[str, Callable[[str, str], Iterator[str]]] = {
        "ollama": ollama_factory or _default_factory,
        "claude": claude_factory or _default_factory,
        "codex": codex_factory or _default_factory,
        "zhipu": zhipu_factory or _default_factory,
    }
    started = time.monotonic()
    candidates = [selected, *([fallback] if fallback else [])]
    try:
        for attempt, model in enumerate(candidates):
            produced = False
            try:
                if (
                    provider_name(model) in {"claude", "codex", "zhipu"}
                    and not MODEL_CLOUD_ALLOWED
                ):
                    raise RuntimeError("云端模型调用已被 MODEL_CLOUD_ALLOWED=0 禁止")
                iterator = factories[provider_name(model)](model, prompt)
                for chunk in iterator:
                    if chunk:
                        produced = True
                        record.chunks += 1
                        record.characters += len(chunk)
                        record.estimated_output_tokens = max(
                            0,
                            ceil(record.characters / max(MODEL_CHARS_PER_TOKEN, 0.1)),
                        )
                        if (
                            MODEL_MAX_OUTPUT_CHARACTERS
                            and record.characters > MODEL_MAX_OUTPUT_CHARACTERS
                        ):
                            raise RuntimeError("超过单次模型输出字符预算")
                        record.estimated_cost_usd = _estimated_cost(record)
                        if (
                            MODEL_MAX_ESTIMATED_COST_USD
                            and record.estimated_cost_usd is not None
                            and record.estimated_cost_usd > MODEL_MAX_ESTIMATED_COST_USD
                        ):
                            raise RuntimeError("超过单次模型估算成本预算")
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


def _estimated_cost(record: GenerationStats) -> float | None:
    if not (MODEL_INPUT_USD_PER_MILLION_TOKENS or MODEL_OUTPUT_USD_PER_MILLION_TOKENS):
        return None
    amount = (
        record.estimated_input_tokens * MODEL_INPUT_USD_PER_MILLION_TOKENS
        + record.estimated_output_tokens * MODEL_OUTPUT_USD_PER_MILLION_TOKENS
    ) / 1_000_000
    return round(amount, 8)
