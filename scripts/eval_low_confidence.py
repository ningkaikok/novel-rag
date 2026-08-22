#!/usr/bin/env python3
"""低置信度信号离线校准：给 confidence.py 的阈值选择提供数据依据。

为什么需要这个脚本
------------------
confidence.py 里的 SCORE_GAP_LOW / TERM_COVERAGE_MIN 目前是**经验占位值**
（见该文件的注释），直接拿去影响线上行为等于拍脑袋。正确的顺序是：

    1. 跑本脚本，对评测集逐条记录「信号新值 + 是否命中」；
    2. 看「置信度分组 × 命中率」对照表——好的阈值应该让低置信组的命中率
       **显著低于**正常组（比如 0.3 vs 0.9）；
    3. 据此回填阈值常量，再跑一遍确认分组效果。

如果某个信号的各分箱命中率没有明显梯度，说明这个信号对当前语料没有区分度，
不该用它触发补救。

本脚本**只读不写**：不改任何检索行为、不碰正式索引之外的数据，跑多少遍都
不会污染线上。它复用 eval_retrieval.py 的关键词命中逻辑（任一期望词出现
即算命中）和同一份评测集，保证校准结论和检索基线可比。

用法
----
    # 用默认评测集（tests/qa_test_set.json）
    python scripts/eval_low_confidence.py

    # 指定评测集 / top-k
    python scripts/eval_low_confidence.py --test-set tests/qa_test_set.json --top-k 5

输出解读
--------
- 「当前阈值」表：is_low_confidence 判定与实际命中的混淆矩阵。理想状态是
  低置信组的命中率远低于正常组；若低置信组里也有大量命中，说明阈值太松、
  会误触发无谓的补救。
- 分箱表：每个信号按取值分箱。阈值应画在「命中率开始明显下滑」的箱边界上。
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# 直接复用 eval_retrieval 的评测集加载与命中判定，避免两套口径各自漂移。
sys.path.insert(0, str(ROOT / "scripts"))

from eval_retrieval import FETCH_K, first_hit_rank, load_cases  # noqa: E402

from confidence import SCORE_GAP_LOW, TERM_COVERAGE_MIN, compute_confidence  # noqa: E402
from embedder import load_embedder  # noqa: E402
from rag import NovelRAG  # noqa: E402
from reranker import rerank_with_scores  # noqa: E402


def collect(rag: NovelRAG, cases: list[dict]) -> list[dict]:
    """逐条跑检索，记录信号新值 + 命中情况。不改变任何检索行为。"""
    rows = []
    for case in cases:
        expect = case["retrieval"]
        question = case["question"]
        sources, _trace = rag.retrieve_hybrid_traced(question, top_k=FETCH_K)
        # 重排分数不在 retrieve_hybrid_traced 的返回里，这里用同一把重排器
        # 对最终结果补一次打分——纯函数、离线运行，代价可接受。
        scored = rerank_with_scores(question, sources, len(sources))
        signals = compute_confidence(question, scored)
        rows.append(
            {
                "id": case["id"],
                "question": question,
                "hit": first_hit_rank(sources, expect["expect_keywords"]) is not None,
                **signals,
            }
        )
    return rows


def _rate(rows: list[dict]) -> tuple[int, float]:
    n = len(rows)
    hit = sum(1 for r in rows if r["hit"])
    return n, (hit / n if n else 0.0)


def print_threshold_table(rows: list[dict]) -> None:
    low = [r for r in rows if r["is_low_confidence"]]
    normal = [r for r in rows if not r["is_low_confidence"]]
    print()
    print("=" * 72)
    print(
        f"当前阈值判定（score_gap<{SCORE_GAP_LOW} 叠加跨书≤1，或 term_coverage<{TERM_COVERAGE_MIN}）"
    )
    print("=" * 72)
    for label, group in (("低置信", low), ("正常", normal)):
        n, rate = _rate(group)
        print(f"  {label:<8} {n:>4} 条   命中率 {rate:.3f}")
    if low:
        print("  → 若低置信组命中率不明显更低，说明阈值会误触发补救，应收紧")


def print_bins(title: str, key_of, rows: list[dict], labels: list[str]) -> None:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[key_of(row)].append(row)
    print()
    print("-" * 72)
    print(f"{title}（分箱 × 命中率）")
    for label in labels:
        n, rate = _rate(buckets.get(label, []))
        bar = "#" * round(rate * 30)
        print(f"  {label:<14} {n:>4} 条   {rate:.3f}  {bar}")


def gap_bin(row: dict) -> str:
    gap = row["score_gap"]
    if gap < 0.02:
        return "<0.02"
    if gap < 0.05:
        return "0.02~0.05"
    if gap < 0.10:
        return "0.05~0.10"
    if gap < 0.20:
        return "0.10~0.20"
    return ">=0.20"


def coverage_bin(row: dict) -> str:
    cov = row["term_coverage"]
    if cov == 0:
        return "0"
    if cov < 0.34:
        return "(0,0.34)"
    if cov < 0.67:
        return "[0.34,0.67)"
    if cov < 1:
        return "[0.67,1)"
    return "1.0"


def main() -> None:
    parser = argparse.ArgumentParser(description="低置信度信号离线校准")
    parser.add_argument(
        "--test-set",
        metavar="FILE",
        help="评测集路径（默认 tests/qa_test_set.json）",
    )
    args = parser.parse_args()

    test_set = Path(args.test_set) if args.test_set else None
    if test_set and not test_set.is_absolute():
        test_set = ROOT / test_set
    cases = load_cases(test_set)
    if not cases:
        print("测试集里没有标注 retrieval 期望的用例", file=sys.stderr)
        sys.exit(1)

    print(f"加载 {len(cases)} 条检索用例，逐条记录置信度信号…")
    rag = NovelRAG(embedder=load_embedder())
    rows = collect(rag, cases)

    print()
    print("=" * 72)
    print("逐条信号明细")
    print("=" * 72)
    print(f"{'ID':<5} {'gap':>7} {'覆盖':>6} {'跨书':>4} {'低置信':<6} {'命中':<4} 问题")
    for r in rows:
        flag = "⚠" if r["is_low_confidence"] else ""
        mark = "✅" if r["hit"] else "❌"
        print(
            f"{r['id']:<5} {r['score_gap']:>7.4f} {r['term_coverage']:>6.2f} "
            f"{r['cross_book_dispersion']:>4} {flag:<6} {mark:<4} {r['question'][:30]}"
        )

    print_threshold_table(rows)
    print_bins(
        "score_gap（第1名与第2名归一化分差）",
        gap_bin,
        rows,
        ["<0.02", "0.02~0.05", "0.05~0.10", "0.10~0.20", ">=0.20"],
    )
    print_bins(
        "term_coverage（问题关键词覆盖率）",
        coverage_bin,
        rows,
        ["0", "(0,0.34)", "[0.34,0.67)", "[0.67,1)", "1.0"],
    )
    print_bins(
        "cross_book_dispersion（候选跨越几本书）",
        lambda r: (
            f"{min(r['cross_book_dispersion'], 4)} 本"
            if r["cross_book_dispersion"] < 4
            else "≥4 本"
        ),
        rows,
        ["1 本", "2 本", "3 本", "≥4 本"],
    )
    print()
    print("用法提示：把命中率开始明显下滑的箱边界回填到 src/confidence.py 的")
    print("SCORE_GAP_LOW / TERM_COVERAGE_MIN，再跑一遍本脚本确认分组效果。")


if __name__ == "__main__":
    main()
