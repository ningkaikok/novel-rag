#!/usr/bin/env python3
"""汇总一次问答结果文件中的引用合法率和预期证据覆盖率。

用法：
    python scripts/eval_citations.py tests/results_7b.json

新结果由 ``tests/run_qa_tests.py`` 直接写入 citation_metrics；旧结果没有该字段时，
本脚本会基于保存的 80 字来源摘录重新计算，因此覆盖率可能偏低，但编号合法率准确。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from citation_eval import evaluate_citations  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path, help="run_qa_tests.py 生成的 JSON 文件")
    args = parser.parse_args()

    payload = json.loads(args.result.read_text(encoding="utf-8"))
    rows = []
    for item in payload.get("results", []):
        metrics = item.get("citation_metrics")
        if metrics is None:
            keywords = (item.get("retrieval") or {}).get("expect_keywords") or []
            metrics = evaluate_citations(
                item.get("answer", ""), item.get("sources", []), keywords
            )
        rows.append((item.get("id", "?"), metrics))

    if not rows:
        print("结果文件中没有问答记录", file=sys.stderr)
        return 1

    valid_mentions = sum(len(m["valid_citation_numbers"]) for _, m in rows)
    all_mentions = sum(len(m["citation_numbers"]) for _, m in rows)
    cited_answers = sum(bool(m["citation_numbers"]) for _, m in rows)
    coverage_values = [
        m["expected_evidence_coverage"]
        for _, m in rows
        if m["expected_evidence_coverage"] is not None
    ]

    print(f"含引用的回答：{cited_answers}/{len(rows)}")
    print(
        f"引用编号合法率：{valid_mentions}/{all_mentions} = "
        f"{(valid_mentions / all_mentions if all_mentions else 0):.1%}"
    )
    if coverage_values:
        print(f"预期证据关键词平均覆盖率：{sum(coverage_values) / len(coverage_values):.1%}")

    problems = [
        (case_id, metrics)
        for case_id, metrics in rows
        if metrics["invalid_citation_numbers"] or metrics["expected_evidence_coverage"] == 0
    ]
    if problems:
        print("\n需要复核：")
        for case_id, metrics in problems:
            print(
                f"  {case_id}: invalid={metrics['invalid_citation_numbers']} "
                f"evidence_coverage={metrics['expected_evidence_coverage']}"
            )
    print("\n注意：规则指标不能证明语义蕴含，最终仍需人工检查引用是否支持对应句子。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
