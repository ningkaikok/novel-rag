"""不可绕过的 Agent 工具执行入口。

ToolSpec/Registry 描述“工具是什么”，本模块负责“这次调用能不能执行”。当前
Agent Lab 只有只读工具，因此权限默认是 ``novel:read``；接口已经把启停、schema、
调用次数、重复动作、超时观测和审计摘要集中起来，后续增加写工具时不需要把安全
逻辑散回规划循环。
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Protocol, cast
from urllib.parse import urlparse

from tool_spec import ToolSpec, get_tool_spec


class ToolGatewayError(ValueError):
    """可供 Agent 观察和审计聚合的结构化工具边界错误。"""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class ToolGatewayPolicy:
    """一次运行共享的边界策略。

    ``max_calls`` 是按运行计数的限流；其余策略默认关闭或取保守值，避免在
    现有只读工具上偷偷改变调用次数。真正接入外部/写工具时，由宿主显式传入
    允许的 host、重试次数和熔断窗口。
    """

    max_calls: int = 5
    max_retries: int = 0
    retry_backoff_s: float = 0.0
    failure_threshold: int = 3
    circuit_cooldown_s: float = 30.0
    allowed_hosts: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ToolAudit:
    tool: str
    version: str
    status: str
    elapsed_ms: int
    attempts: int = 1


class ToolboxProtocol(Protocol):
    def execute(self, name: str, args: dict[str, Any]) -> Any: ...


def isolate_untrusted_text(text: str, *, label: str = "tool_output") -> str:
    """把外部/检索文本包成只读数据区，降低提示注入被当成策略的机会。

    这不是权限边界：最终是否能调用工具仍由 ``ToolGateway`` 决定。它只是让
    Planner/回答模型看到清晰的信任边界，且不复制或改写原文内容。
    """

    safe_label = "".join(char if char.isalnum() or char == "_" else "_" for char in label)
    return f'<{safe_label} untrusted="true">\n{text}\n</{safe_label}>'


def _type_matches(value: Any, expected: str | list[str]) -> bool:
    expected_types = [expected] if isinstance(expected, str) else expected
    for kind in expected_types:
        if kind == "null" and value is None:
            return True
        if kind == "string" and isinstance(value, str):
            return True
        if kind == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if (
            kind == "number"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            return True
        if kind == "array" and isinstance(value, list):
            return True
        if kind == "object" and isinstance(value, dict):
            return True
    return False


def validate_tool_args(spec: ToolSpec, args: Mapping[str, Any]) -> None:
    schema = cast(dict[str, Any], spec.params_json_schema)
    properties = cast(dict[str, dict[str, Any]], schema.get("properties", {}))
    required = cast(list[str], schema.get("required", []))
    missing = [name for name in required if name not in args]
    if missing:
        raise ToolGatewayError("invalid_args", f"缺少必填参数：{', '.join(missing)}")
    unknown = sorted(set(args) - set(properties))
    if unknown:
        raise ToolGatewayError("invalid_args", f"未知参数：{', '.join(unknown)}")
    for name, value in args.items():
        rule = properties[name]
        if not _type_matches(value, rule.get("type", "object")):
            raise ToolGatewayError("invalid_args", f"参数 {name} 类型不正确")
        if "enum" in rule and value not in rule["enum"]:
            raise ToolGatewayError("invalid_args", f"参数 {name} 不在允许值范围内")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in rule and value < rule["minimum"]:
                raise ToolGatewayError("invalid_args", f"参数 {name} 小于最小值")
            if "maximum" in rule and value > rule["maximum"]:
                raise ToolGatewayError("invalid_args", f"参数 {name} 大于最大值")
        if rule.get("type") == "array" and "items" in rule:
            item_type = rule["items"].get("type", "object")
            if not isinstance(value, list) or not all(
                _type_matches(item, item_type) for item in value
            ):
                raise ToolGatewayError("invalid_args", f"参数 {name} 包含非法数组元素")


class ToolGateway:
    """为一个 Agent 运行执行工具调用并保留不含正文的审计摘要。"""

    def __init__(
        self,
        toolbox: ToolboxProtocol,
        *,
        permissions: set[str] | frozenset[str] | None = None,
        max_calls: int = 5,
        policy: ToolGatewayPolicy | None = None,
    ):
        self.toolbox = toolbox
        self.permissions = frozenset({"novel:read"} if permissions is None else permissions)
        base_policy = policy or ToolGatewayPolicy(max_calls=max_calls)
        self.policy = ToolGatewayPolicy(
            max_calls=max(1, int(base_policy.max_calls)),
            max_retries=max(0, int(base_policy.max_retries)),
            retry_backoff_s=max(0.0, float(base_policy.retry_backoff_s)),
            failure_threshold=max(1, int(base_policy.failure_threshold)),
            circuit_cooldown_s=max(0.0, float(base_policy.circuit_cooldown_s)),
            allowed_hosts=frozenset(base_policy.allowed_hosts),
        )
        self.max_calls = self.policy.max_calls
        self._calls = 0
        self._seen: set[str] = set()
        self._completed: dict[str, Any] = {}
        self._audits: list[ToolAudit] = []
        self._failures: dict[str, int] = {}
        self._circuit_opened_at: dict[str, float] = {}

    def _authorize(
        self,
        name: str,
        args: Mapping[str, Any],
        *,
        allow_duplicate: bool = False,
    ) -> ToolSpec:
        try:
            spec = get_tool_spec(name)
        except KeyError as exc:
            raise ToolGatewayError("unknown_tool", str(exc)) from None
        if not spec.enabled:
            raise ToolGatewayError("tool_disabled", f"工具已停用：{name}")
        if spec.permission not in self.permissions:
            raise ToolGatewayError("permission_denied", f"没有权限：{spec.permission}")
        if self._calls >= self.max_calls:
            raise ToolGatewayError("rate_limited", "本次运行已达到工具调用上限")
        validate_tool_args(spec, args)
        self._validate_outbound(spec, args)
        signature = json.dumps([name, dict(args)], sort_keys=True, ensure_ascii=False)
        if signature in self._seen and not allow_duplicate:
            raise ToolGatewayError("duplicate_call", "本次运行禁止重复执行相同工具调用")
        self._seen.add(signature)
        self._calls += 1
        return spec

    def execute(
        self,
        name: str,
        args: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ):
        spec = self._authorize(name, args, allow_duplicate=idempotency_key is not None)
        if idempotency_key and idempotency_key in self._completed:
            self._audits.append(ToolAudit(name, spec.version, "idempotent_replay", 0))
            return self._completed[idempotency_key]
        self._ensure_circuit_closed(name)
        started = time.perf_counter()
        attempts = 0
        try:
            while True:
                attempts += 1
                try:
                    result = self._execute_once(spec, name, args)
                except ToolGatewayError as exc:
                    if (
                        spec.readonly
                        and exc.category in {"downstream_error", "timeout"}
                        and attempts <= self.policy.max_retries
                    ):
                        if self.policy.retry_backoff_s:
                            time.sleep(self.policy.retry_backoff_s)
                        continue
                    self._record_failure(name)
                    self._record(spec, exc.category, started, attempts)
                    raise
                self._record_success(name)
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                self._audits.append(
                    ToolAudit(name, spec.version, "complete", elapsed_ms, attempts)
                )
                if idempotency_key:
                    self._completed[idempotency_key] = result
                return result
        except ToolGatewayError:
            raise
        except Exception as exc:
            self._record_failure(name)
            self._record(spec, "downstream_error", started, attempts)
            raise ToolGatewayError("downstream_error", f"{type(exc).__name__}: {exc}") from exc

    def accept(self, name: str, args: Mapping[str, Any]) -> ToolSpec:
        """校验不由 Toolbox 执行的终止动作（如 answer_with_citations）。"""
        spec = self._authorize(name, args)
        self._audits.append(ToolAudit(name, spec.version, "accepted", 0))
        return spec

    def snapshot(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for audit in self._audits:
            counts[audit.status] = counts.get(audit.status, 0) + 1
        return {
            "calls": self._calls,
            "max_calls": self.max_calls,
            "policy": {
                "max_retries": self.policy.max_retries,
                "failure_threshold": self.policy.failure_threshold,
                "circuit_cooldown_s": self.policy.circuit_cooldown_s,
                "allowed_hosts": sorted(self.policy.allowed_hosts),
            },
            "permissions": sorted(self.permissions),
            "statuses": counts,
            "audits": [audit.__dict__ for audit in self._audits],
        }

    def _execute_once(
        self,
        spec: ToolSpec,
        name: str,
        args: Mapping[str, Any],
    ) -> Any:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.toolbox.execute, name, dict(args))
        try:
            result = future.result(timeout=spec.timeout_s)
        except FutureTimeoutError:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise ToolGatewayError(
                "timeout", f"工具执行超过 {spec.timeout_s}s：{name}"
            ) from None
        except Exception as exc:
            executor.shutdown(wait=False, cancel_futures=True)
            raise ToolGatewayError("downstream_error", f"{type(exc).__name__}: {exc}") from exc
        else:
            executor.shutdown(wait=True, cancel_futures=True)
            return result

    def _validate_outbound(self, spec: ToolSpec, args: Mapping[str, Any]) -> None:
        """所有未来出站参数都必须命中宿主显式提供的 host 白名单。"""

        for key, value in args.items():
            if key.lower() not in {"url", "uri", "endpoint", "host", "hostname"}:
                continue
            if not isinstance(value, str):
                continue
            parsed = urlparse(value if "://" in value else f"//{value}")
            host = parsed.hostname
            if not host or host not in self.policy.allowed_hosts:
                raise ToolGatewayError("outbound_denied", f"出站地址未获允许：{value}")

    def _ensure_circuit_closed(self, name: str) -> None:
        opened_at = self._circuit_opened_at.get(name)
        if opened_at is None:
            return
        if time.monotonic() - opened_at < self.policy.circuit_cooldown_s:
            raise ToolGatewayError("circuit_open", f"工具熔断中：{name}")
        self._circuit_opened_at.pop(name, None)
        self._failures[name] = 0

    def _record_success(self, name: str) -> None:
        self._failures.pop(name, None)
        self._circuit_opened_at.pop(name, None)

    def _record_failure(self, name: str) -> None:
        failures = self._failures.get(name, 0) + 1
        self._failures[name] = failures
        if failures >= self.policy.failure_threshold:
            self._circuit_opened_at[name] = time.monotonic()

    def _record(
        self,
        spec: ToolSpec,
        status: str,
        started: float,
        attempts: int = 1,
    ) -> None:
        self._audits.append(
            ToolAudit(
                spec.name,
                spec.version,
                status,
                int((time.perf_counter() - started) * 1000),
                attempts,
            )
        )
