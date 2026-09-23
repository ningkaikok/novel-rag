#!/usr/bin/env python3
"""比较 V2 原生文档 read path 开关前后的固定问答集结果。

脚本只输出排名、来源类型、候选变化和耗时，不输出文档正文，适合把结果保存为
基线并在开启灰度前后复盘。它需要当前 PostgreSQL 中已有 V2 原生文档索引。
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import rag  # noqa: E402
import retrieval_mixins  # noqa: E402
from embedder import load_embedder  # noqa: E402

DEFAULT_TEST_SET = ROOT / "tests" / "v2_native_qa_test_set.json"


def _load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"测试集必须是非空数组：{path}")
    return cases


def _hit_rank(sources: list, keywords: list[str]) -> int | None:
    if not keywords:
        return None
    for rank, source in enumerate(sources, start=1):
        if any(keyword in source.text for keyword in keywords):
            return rank
    return None


def _run(rag_service, cases: list[dict], *, native_enabled: bool, top_k: int) -> dict:
    """在同一进程内切换两组开关，避免评测依赖重启服务。"""
    old_rag = rag.V2_NATIVE_RETRIEVAL_ENABLED
    old_mixin = retrieval_mixins.V2_NATIVE_RETRIEVAL_ENABLED
    old_mode = rag.V2_NATIVE_RETRIEVAL_MODE
    old_mixin_mode = retrieval_mixins.V2_NATIVE_RETRIEVAL_MODE
    try:
        rag.V2_NATIVE_RETRIEVAL_ENABLED = native_enabled
        retrieval_mixins.V2_NATIVE_RETRIEVAL_ENABLED = native_enabled
        rag.V2_NATIVE_RETRIEVAL_MODE = "off"
        retrieval_mixins.V2_NATIVE_RETRIEVAL_MODE = "off"
        rows = []
        for case in cases:
            started = time.perf_counter()
            sources, trace = rag_service.retrieve_hybrid_traced(case["question"], top_k=top_k)
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            native_count = sum(source.origin == "v2_document" for source in sources)
            trace_native = next(
                (step for step in trace if step.get("stage_key") == "v2_native"), None
            )
            rows.append(
                {
                    "id": case["id"],
                    "hit_rank": _hit_rank(sources, case.get("expect_keywords", [])),
                    "native_count": native_count,
                    "v2_candidate_count": len((trace_native or {}).get("candidates", [])),
                    "result_origins": sorted({source.origin for source in sources}),
                    "result_keys": [
                        f"{source.origin}:{source.novel}:{source.chunk_id}"
                        for source in sources
                    ],
                    "elapsed_ms": elapsed_ms,
                }
            )
        return {"enabled": native_enabled, "cases": rows}
    finally:
        rag.V2_NATIVE_RETRIEVAL_ENABLED = old_rag
        retrieval_mixins.V2_NATIVE_RETRIEVAL_ENABLED = old_mixin
        rag.V2_NATIVE_RETRIEVAL_MODE = old_mode
        retrieval_mixins.V2_NATIVE_RETRIEVAL_MODE = old_mixin_mode


def _summary(report: dict, cases: list[dict], top_k: int) -> dict:
    rows = report["cases"]
    with_gold = [
        row for row, case in zip(rows, cases, strict=True) if case.get("expect_keywords")
    ]
    hits = [row["hit_rank"] for row in with_gold]
    return {
        "cases": len(rows),
        "gold_cases": len(with_gold),
        f"recall@{top_k}": round(
            sum(rank is not None and rank <= top_k for rank in hits) / len(hits), 3
        )
        if hits
        else None,
        "mrr": round(sum(1 / rank for rank in hits if rank) / len(hits), 3) if hits else None,
        "native_result_cases": sum(row["native_count"] > 0 for row in rows),
        "avg_ms": round(sum(row["elapsed_ms"] for row in rows) / len(rows)) if rows else 0,
    }


def _compare(baseline: dict, candidate: dict) -> dict:
    before = {row["id"]: row for row in baseline["cases"]}
    changed = [
        row["id"]
        for row in candidate["cases"]
        if row["result_keys"] != before[row["id"]]["result_keys"]
    ]
    return {
        "changed_cases": changed,
        "changed_case_count": len(changed),
        "avg_ms_delta": candidate["summary"]["avg_ms"] - baseline["summary"]["avg_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V1/V2 原生文档 read path 固定集评测")
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save", type=Path, help="保存 JSON 报告")
    parser.add_argument("--compare", type=Path, help="与之前保存的报告比较")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k 必须为正数")

    cases = _load_cases(args.test_set)
    service = rag.NovelRAG(embedder=load_embedder())
    baseline = _run(service, cases, native_enabled=False, top_k=args.top_k)
    candidate = _run(service, cases, native_enabled=True, top_k=args.top_k)
    baseline["summary"] = _summary(baseline, cases, args.top_k)
    candidate["summary"] = _summary(candidate, cases, args.top_k)
    report = {
        "schema_version": 1,
        "test_set": str(args.test_set.relative_to(ROOT)),
        "top_k": args.top_k,
        "baseline": baseline,
        "candidate": candidate,
        "comparison": _compare(baseline, candidate),
    }
    if args.compare:
        previous = json.loads(args.compare.read_text(encoding="utf-8"))
        report["previous_report_comparison"] = {
            "candidate_avg_ms_delta": candidate["summary"]["avg_ms"]
            - previous["candidate"]["summary"]["avg_ms"]
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.save:
        args.save.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
