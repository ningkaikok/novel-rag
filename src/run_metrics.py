"""从安全 Agent 事件计算运行级质量、可靠性和成本观测指标。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def summarize_run_events(events: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """聚合事件元数据，不读取也不推断 Prompt、回答或原文内容。"""

    rows = list(events)
    route = next(
        (str(event["route"]) for event in rows if event.get("event_type") == "route_selected"),
        None,
    )
    finished = next(
        (event for event in reversed(rows) if event.get("event_type") == "run_finished"),
        None,
    )
    tool_events = [event for event in rows if event.get("event_type") == "tool_finished"]
    tool_successes = sum(
        1
        for event in tool_events
        if str(event.get("status", "")) in {"complete", "success", "observed"}
    )
    tool_failures = sum(
        1
        for event in tool_events
        if str(event.get("status", "")) in {"error", "failed", "timeout", "downstream_error"}
    )
    elapsed = finished.get("elapsed_ms") if finished else None
    return {
        "event_count": len(rows),
        "route": route,
        "status": finished.get("status") if finished else None,
        "elapsed_ms": int(elapsed) if isinstance(elapsed, int) else None,
        "tool_calls": len(tool_events),
        "tool_successes": tool_successes,
        "tool_failures": tool_failures,
        "tool_success_rate": tool_successes / len(tool_events) if tool_events else None,
        "evidence_events": sum(
            1 for event in rows if event.get("event_type") == "evidence_added"
        ),
        "answer_generated": any(
            event.get("event_type") == "answer_generated" for event in rows
        ),
        "answer_failed": any(event.get("event_type") == "answer_failed" for event in rows),
    }
