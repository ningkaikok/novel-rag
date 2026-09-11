from run_metrics import summarize_run_events


def test_summarize_run_events_distinguishes_tool_failures_and_answer_failure():
    summary = summarize_run_events(
        [
            {"event_type": "route_selected", "route": "grounded"},
            {"event_type": "tool_finished", "tool": "search_novels", "status": "complete"},
            {"event_type": "tool_finished", "tool": "read_neighbors", "status": "timeout"},
            {"event_type": "answer_failed"},
            {"event_type": "run_finished", "status": "error", "elapsed_ms": 123},
        ]
    )

    assert summary == {
        "event_count": 5,
        "route": "grounded",
        "status": "error",
        "elapsed_ms": 123,
        "tool_calls": 2,
        "tool_successes": 1,
        "tool_failures": 1,
        "tool_success_rate": 0.5,
        "evidence_events": 0,
        "answer_generated": False,
        "answer_failed": True,
    }


def test_summarize_empty_run_has_no_division_by_zero():
    summary = summarize_run_events([])

    assert summary["event_count"] == 0
    assert summary["tool_success_rate"] is None
    assert summary["elapsed_ms"] is None
