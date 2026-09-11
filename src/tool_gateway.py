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
from dataclasses import dataclass
from typing import Any

from tool_spec import ToolSpec, get_tool_spec


class ToolGatewayError(ValueError):
    """可供 Agent 观察和审计聚合的结构化工具边界错误。"""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class ToolAudit:
    tool: str
    version: str
    status: str
    elapsed_ms: int


def _type_matches(value: object, expected: str | list[str]) -> bool:
    expected_types = [expected] if isinstance(expected, str) else expected
    for kind in expected_types:
        if kind == "null" and value is None:
            return True
        if kind == "string" and isinstance(value, str):
            return True
        if kind == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if kind == "array" and isinstance(value, list):
            return True
        if kind == "object" and isinstance(value, dict):
            return True
    return False


def validate_tool_args(spec: ToolSpec, args: Mapping[str, object]) -> None:
    schema = spec.params_json_schema
    properties = schema.get("properties", {})
    required = schema.get("required", [])
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
            if not all(_type_matches(item, item_type) for item in value):
                raise ToolGatewayError("invalid_args", f"参数 {name} 包含非法数组元素")


class ToolGateway:
    """为一个 Agent 运行执行工具调用并保留不含正文的审计摘要。"""

    def __init__(
        self,
        toolbox: object,
        *,
        permissions: set[str] | frozenset[str] | None = None,
        max_calls: int = 5,
    ):
        self.toolbox = toolbox
        self.permissions = frozenset(
            {"novel:read"} if permissions is None else permissions
        )
        self.max_calls = max(1, int(max_calls))
        self._calls = 0
        self._seen: set[str] = set()
        self._audits: list[ToolAudit] = []

    def _authorize(self, name: str, args: Mapping[str, object]) -> ToolSpec:
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
        signature = json.dumps([name, dict(args)], sort_keys=True, ensure_ascii=False)
        if signature in self._seen:
            raise ToolGatewayError("duplicate_call", "本次运行禁止重复执行相同工具调用")
        self._seen.add(signature)
        self._calls += 1
        return spec

    def execute(self, name: str, args: Mapping[str, object]):
        spec = self._authorize(name, args)
        started = time.perf_counter()
        try:
            result = self.toolbox.execute(name, dict(args))
        except Exception:
            self._record(spec, "error", started)
            raise
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self._audits.append(ToolAudit(name, spec.version, "complete", elapsed_ms))
        if elapsed_ms > spec.timeout_s * 1000:
            raise ToolGatewayError("timeout", f"工具执行超过 {spec.timeout_s}s：{name}")
        return result

    def accept(self, name: str, args: Mapping[str, object]) -> ToolSpec:
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
            "permissions": sorted(self.permissions),
            "statuses": counts,
            "audits": [audit.__dict__ for audit in self._audits],
        }

    def _record(self, spec: ToolSpec, status: str, started: float) -> None:
        self._audits.append(
            ToolAudit(
                spec.name,
                spec.version,
                status,
                int((time.perf_counter() - started) * 1000),
            )
        )
