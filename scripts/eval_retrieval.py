#!/usr/bin/env python3
"""检索质量评测：跑一遍测试集，算出 Recall@k 和 MRR。

为什么需要这个脚本
------------------
改检索策略时最容易骗自己的地方是：随手试两个问题，感觉"好像变好了"，就认为
改进成功。但检索是个统计问题——某个改动可能让 A 类问题变好、同时让 B 类问题
变差，只试几个问题根本看不出来。

业界共识是 RAG 失败时约七成的失败点在检索环节而不是生成环节，所以检索指标
是最该先立起来的东西。**没有这个脚本，后面每一项改进都无法判断是真的变好、
还是只是换了一批失败案例。**

指标怎么定义的
--------------
理想情况下应该人工标注「每个问题的正确答案在哪几个片段里」，但那个工作量很大。
这里用一个更省事、但足够有效的代理指标：**期望关键词命中**。

    对每个问题，标注一组「正确答案的原文里必然出现的词」（比如问韩立的绰号，
    答案原文里必然有「二愣子」三个字）。如果某个被召回的片段包含了任一期望
    关键词，就算这个片段「命中」。

这个代理指标的**局限**要说清楚：
- 它只能证明「召回了含正确答案的片段」，不能证明「模型会用好这个片段」；
- 片段里出现关键词也可能只是碰巧提到，不代表真的回答了问题。

但它有个决定性优点：**完全客观、可自动重复跑**。对「比较改动前后哪个更好」
这个用途来说足够了，这也正是我们需要它的场景。

两个指标各自回答什么问题
------------------------
- **Recall@k**：前 k 个结果里，有没有至少一个命中？
  → 回答"能不能找到"。这是及格线：找不到，后面全免谈。

- **MRR（Mean Reciprocal Rank，平均倒数排名）**：第一个命中的片段排第几名，
  取排名的倒数（排第 1 得 1.0，排第 2 得 0.5，排第 10 得 0.1），再对所有
  问题求平均。
  → 回答"找到的东西排得够不够前"。这个指标之所以重要：最终只有 top-k 会被
    塞进 prompt 送给模型，一个排在第 20 名的正确片段，实际效果等同于没找到。
    Recall@20 很高但 MRR 很低，说明"东西在候选池里但排序很差"——这正是
    重排（reranker）要解决的问题。

用法
----
    # 跑一遍，打印结果
    python scripts/eval_retrieval.py

    # 存下这次结果，作为后续改动的对照基线
    python scripts/eval_retrieval.py --save baseline.json

    # 改完代码后，跟基线对比，直接看每条用例是变好还是变差
    python scripts/eval_retrieval.py --compare baseline.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from embedder import load_embedder  # noqa: E402
from rag import NovelRAG  # noqa: E402

TEST_SET = ROOT / "tests" / "qa_test_set.json"

# 在这几个 k 上分别算 Recall。取到 20 是因为召回候选池就是 20（RECALL_K），
# 看 Recall@20 能区分出"根本没召回到"和"召回了但排得太靠后"这两种不同的失败。
RECALL_AT = [1, 3, 5, 10, 20]
# 一次检索取多少个候选。必须 >= max(RECALL_AT)，否则算不出大 k 的 Recall。
FETCH_K = max(RECALL_AT)


def load_cases() -> list[dict]:
    """读测试集，只保留标了 retrieval 期望的用例。

    有些用例（抗幻觉、纯生成层回归）天生没有"应该被召回的片段"，
    它们在 JSON 里 retrieval 字段是 null，并写明了跳过原因。
    """
    cases = json.loads(TEST_SET.read_text(encoding="utf-8"))
    return [c for c in cases if c.get("retrieval")]


def first_hit_rank(sources: list, expect_keywords: list[str]) -> int | None:
    """返回第一个命中片段的排名（从 1 开始）；一个都没命中返回 None。

    命中判定用「任一关键词出现即命中」而不是「全部关键词都要出现」：
    一个问题的答案可能跨多个片段（比如"名字谁取的、用什么换的"，
    「老张叔」和「窝头」完全可能分布在不同片段里），要求单个片段
    包含全部关键词会把本来正确的召回误判成失败。
    """
    for rank, source in enumerate(sources, start=1):
        if any(kw in source.text for kw in expect_keywords):
            return rank
    return None


def evaluate(rag: NovelRAG, cases: list[dict]) -> dict:
    results = []
    for case in cases:
        expect = case["retrieval"]
        started = time.time()
        sources = rag.retrieve_hybrid(case["question"], top_k=FETCH_K)
        elapsed_ms = (time.time() - started) * 1000

        rank = first_hit_rank(sources, expect["expect_keywords"])
        # 路由是否正确：召回的片段里，排第一的那个是不是期望的那本书。
        # 用「书名里是否包含期望的短标题」判断，因为库里存的是文件名
        # （形如"《凡人修仙传》（校对版全本+番外）作者：忘语"）。
        top_novel = sources[0].novel if sources else ""
        routed_right = expect.get("expect_novel", "") in top_novel

        results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "difficulty": expect.get("difficulty", "?"),
                "hit_rank": rank,
                "routed_right": routed_right,
                "elapsed_ms": round(elapsed_ms),
            }
        )

    # Recall@k：命中排名 <= k 的用例占比
    recall = {}
    for k in RECALL_AT:
        hit = sum(1 for r in results if r["hit_rank"] is not None and r["hit_rank"] <= k)
        recall[f"recall@{k}"] = round(hit / len(results), 3) if results else 0.0

    # MRR：命中排名的倒数取平均；完全没命中的按 0 计入（不是跳过——
    # 跳过会让"召回率低但命中的都排第一"的系统拿到虚高的 MRR）
    mrr = sum(1 / r["hit_rank"] for r in results if r["hit_rank"]) / len(results) if results else 0.0

    routing_acc = sum(1 for r in results if r["routed_right"]) / len(results) if results else 0.0

    return {
        "summary": {
            **recall,
            "mrr": round(mrr, 3),
            "routing_accuracy": round(routing_acc, 3),
            "cases": len(results),
            "avg_ms": round(sum(r["elapsed_ms"] for r in results) / len(results)) if results else 0,
        },
        "cases": results,
    }


def print_report(report: dict, baseline: dict | None = None) -> None:
    print()
    print("=" * 78)
    print("逐条用例")
    print("=" * 78)
    print(f"{'ID':<5} {'难度':<8} {'命中排名':<10} {'路由':<6} {'耗时':<8} 问题")
    print("-" * 78)

    base_by_id = {c["id"]: c for c in baseline["cases"]} if baseline else {}
    for case in report["cases"]:
        rank = case["hit_rank"]
        rank_text = f"#{rank}" if rank else "未命中"

        # 跟基线比：命中排名变小=变好，变大=变差
        if base_by_id:
            old = base_by_id.get(case["id"], {}).get("hit_rank")
            if old != rank:
                old_text = f"#{old}" if old else "未命中"
                if rank and (old is None or rank < old):
                    rank_text = f"{old_text}→{rank_text} ↑"
                else:
                    rank_text = f"{old_text}→{rank_text} ↓"

        print(
            f"{case['id']:<5} {case['difficulty']:<8} {rank_text:<12} "
            f"{'✅' if case['routed_right'] else '❌':<5} "
            f"{case['elapsed_ms']:>5}ms  {case['question'][:28]}"
        )

    s = report["summary"]
    print()
    print("=" * 78)
    print("汇总指标")
    print("=" * 78)
    for k in RECALL_AT:
        key = f"recall@{k}"
        line = f"  {key:<12} {s[key]:.3f}"
        if baseline:
            delta = s[key] - baseline["summary"][key]
            if abs(delta) > 1e-9:
                line += f"   ({'+' if delta > 0 else ''}{delta:.3f})"
        print(line)
    for key, label in [("mrr", "MRR"), ("routing_accuracy", "路由准确率")]:
        line = f"  {label:<12} {s[key]:.3f}"
        if baseline:
            delta = s[key] - baseline["summary"][key]
            if abs(delta) > 1e-9:
                line += f"   ({'+' if delta > 0 else ''}{delta:.3f})"
        print(line)
    print(f"  {'平均耗时':<12} {s['avg_ms']}ms")
    print(f"  {'用例数':<12} {s['cases']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="检索质量评测")
    parser.add_argument("--save", metavar="FILE", help="把本次结果存成基线文件")
    parser.add_argument("--compare", metavar="FILE", help="跟指定的基线文件对比")
    args = parser.parse_args()

    cases = load_cases()
    if not cases:
        print("测试集里没有标注 retrieval 期望的用例", file=sys.stderr)
        sys.exit(1)

    print(f"加载 {len(cases)} 条可自动判定的检索用例，正在评测…")
    rag = NovelRAG(embedder=load_embedder())
    report = evaluate(rag, cases)

    baseline = None
    if args.compare:
        baseline_path = Path(args.compare)
        if not baseline_path.is_absolute():
            baseline_path = ROOT / baseline_path
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    print_report(report, baseline)

    if args.save:
        save_path = Path(args.save)
        if not save_path.is_absolute():
            save_path = ROOT / save_path
        save_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"已保存基线到 {save_path}")


if __name__ == "__main__":
    main()
