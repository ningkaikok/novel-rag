#!/usr/bin/env python3
"""检索延迟画像：按阶段统计真实大部头索引上的耗时分布（路线图 M3.4 性能预算）。

背景
----
大部头（《凡人修仙传》19901 片段）实测单次混合检索 2.3~3.3s，但不知道时间
花在哪个阶段。本脚本对一组真实问题逐条跑 retrieve_hybrid_traced，从 trace 里
提取每阶段的 ms，输出各阶段的 P50/P95/均值占比——瓶颈是谁一目了然。

用法
----
    uv run python scripts/profile_latency.py                 # 默认问题集 × 3 轮
    uv run python scripts/profile_latency.py --rounds 5      # 更多轮取更稳分位

只做只读检索：不连 Ollama、不写任何表。数据库用 DATABASE_URL 指向的真实库。
"""
import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from embedder import load_embedder  # noqa: E402
from rag import NovelRAG  # noqa: E402

# 用真实大部头的评测问题，保证查询分布与线上场景一致
QUESTIONS = [
    "韩立小时候的绰号是什么？",
    "韩立的名字是谁给取的？",
    "韩立家里都有谁？",
    "这几本书里，谁的绰号叫二愣子？",
    "韩立有哪些伴侣？",
    "凡人修仙传的结局是怎样的？",
    "韩立修炼的功法叫什么名字？",
    "韩立在七玄门时最好的朋友是谁？",
]


def percentile(values: list[float], pct: float) -> float:
    """简单线性插值分位；样本量小，够用即可。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100
    low, high = int(idx), min(int(idx) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (idx - low)


def main() -> None:
    parser = argparse.ArgumentParser(description="检索延迟分阶段画像")
    parser.add_argument("--rounds", type=int, default=3, help="每个问题重复轮数")
    args = parser.parse_args()

    rag = NovelRAG(embedder=load_embedder())

    stage_ms: dict[str, list[float]] = defaultdict(list)
    totals: list[float] = []

    for round_no in range(args.rounds):
        for question in QUESTIONS:
            started = time.perf_counter()
            _sources, trace = rag.retrieve_hybrid_traced(question)
            wall_ms = (time.perf_counter() - started) * 1000
            totals.append(wall_ms)
            for step in trace:
                ms = step.get("ms")
                if ms is not None:
                    # 同名阶段可能多次出现（如多路召回各自的行），全部计入
                    stage_ms[step["step"]].append(float(ms))
        print(f"第 {round_no + 1}/{args.rounds} 轮完成", file=sys.stderr)

    print()
    print(f"{'阶段':<14} {'次数':>5} {'P50(ms)':>9} {'P95(ms)':>9} {'均值':>8} {'占总时长%':>9}")
    print("-" * 62)
    total_mean = statistics.mean(totals)
    rows = sorted(stage_ms.items(), key=lambda kv: -statistics.mean(kv[1]))
    for stage, values in rows:
        print(
            f"{stage:<14} {len(values):>5} {percentile(values, 50):>9.0f} "
            f"{percentile(values, 95):>9.0f} {statistics.mean(values):>8.0f} "
            f"{statistics.mean(values) / total_mean * 100:>8.1f}%"
        )
    print("-" * 62)
    print(
        f"{'端到端总计':<13} {len(totals):>5} {percentile(totals, 50):>9.0f} "
        f"{percentile(totals, 95):>9.0f} {total_mean:>8.0f}"
    )

    out = ROOT / "docs" / "experiments" / "latency-profile.json"
    out.write_text(
        json.dumps(
            {
                "rounds": args.rounds,
                "questions": len(QUESTIONS),
                "totals_ms": {"p50": percentile(totals, 50), "p95": percentile(totals, 95), "mean": total_mean},
                "stages": {
                    stage: {
                        "count": len(v),
                        "p50": percentile(v, 50),
                        "p95": percentile(v, 95),
                        "mean": statistics.mean(v),
                        "share_of_total": statistics.mean(v) / total_mean,
                    }
                    for stage, v in rows
                },
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\n原始数据已写入 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
