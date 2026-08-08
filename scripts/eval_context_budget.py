#!/usr/bin/env python3
"""长上下文取舍：测「送多少片段给模型最划算」。

为什么需要这个脚本
------------------
`TOP_K` 一直是拍脑袋定的 5。但这个参数直接决定了送进 prompt 的上下文量，
是业界公认的 RAG 四杠杆之一（切分、混合检索、重排、**长上下文取舍**），
不该没有依据。

这个杠杆的两难：

- **送太少**：正确答案可能压根没进 prompt，模型无米下炊
- **送太多**：① 花钱变多、变慢；② **"迷失在中间"**——上下文过长时模型对
  中间部分的注意力会下降，反而可能忽略掉正确答案；③ 无关片段会稀释信号，
  增加模型被带偏、产生幻觉的概率

所以存在一个最优点，而且它跟具体的语料、模型、问题类型都有关——**只能测**。

这个脚本测什么
--------------
对每个 `TOP_K` 取值，跑一遍检索并统计：

- **命中率**：正确片段有没有进 prompt（用 tests/qa_test_set.json 的期望关键词判定）
- **上下文字数**：实际拼进 prompt 的字数——直接对应成本和延迟
- **信噪比**：命中的片段占送进去的片段的比例。这个指标最容易被忽略：
  `TOP_K=20` 的命中率一定不低于 `TOP_K=5`，但如果 19 段都是噪声，
  模型反而更容易被带偏。

**注意这里测的是"检索层面的性价比"，不是"回答质量"。** 真正的回答质量要靠
人工评估或 LLM-as-judge，成本高得多。但检索层面的数据已经足以回答
"TOP_K 该设多少"这个具体问题——这也是这个脚本的定位。

用法
----
    python scripts/eval_context_budget.py
    python scripts/eval_context_budget.py --values 1,3,5,10,20
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from embedder import load_embedder  # noqa: E402
from rag import NovelRAG  # noqa: E402

TEST_SET = ROOT / "tests" / "qa_test_set.json"


def load_cases() -> list[dict]:
    cases = json.loads(TEST_SET.read_text(encoding="utf-8"))
    return [c for c in cases if c.get("retrieval")]


def measure(rag: NovelRAG, cases: list[dict], top_k: int) -> dict:
    hits = 0
    total_chars = 0
    total_chunks = 0
    hit_chunks = 0

    for case in cases:
        expect = case["retrieval"]["expect_keywords"]
        sources = rag.retrieve_hybrid(case["question"], top_k=top_k)
        # 和线上一致：检索完还会补相邻片段，所以要按扩展后的结果算成本
        context_sources = rag.expand_neighbors(sources)
        prompt = rag.build_prompt(case["question"], context_sources)

        total_chars += len(prompt)
        total_chunks += len(context_sources)
        matched = [s for s in context_sources if any(kw in s.text for kw in expect)]
        if matched:
            hits += 1
        hit_chunks += len(matched)

    n = len(cases)
    return {
        "top_k": top_k,
        "hit_rate": round(hits / n, 3),
        "avg_prompt_chars": round(total_chars / n),
        "avg_chunks": round(total_chunks / n, 1),
        # 信噪比：送进去的片段里有多大比例真的相关
        "signal_ratio": round(hit_chunks / total_chunks, 3) if total_chunks else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="测不同 TOP_K 的性价比")
    parser.add_argument(
        "--values",
        default="1,3,5,10,20",
        help="要测的 TOP_K 取值，逗号分隔（默认 1,3,5,10,20）",
    )
    args = parser.parse_args()
    values = [int(v) for v in args.values.split(",")]

    cases = load_cases()
    print(f"用 {len(cases)} 条可自动判定的用例，测 TOP_K = {values}")
    rag = NovelRAG(embedder=load_embedder())

    results = [measure(rag, cases, k) for k in values]

    print()
    print("=" * 66)
    print(f"{'TOP_K':>6} {'命中率':>8} {'prompt字数':>12} {'实际片段数':>10} {'信噪比':>8}")
    print("-" * 66)
    for r in results:
        print(
            f"{r['top_k']:>6} {r['hit_rate']:>8.3f} {r['avg_prompt_chars']:>12} "
            f"{r['avg_chunks']:>10.1f} {r['signal_ratio']:>8.3f}"
        )
    print("=" * 66)
    print()
    print("怎么读这张表：")
    print("  · 命中率上不去 → TOP_K 太小，正确答案没进 prompt")
    print("  · 命中率不再涨但字数还在涨 → 已经过了最优点，纯粹在浪费成本")
    print("  · 信噪比持续下降 → 噪声占比变高，模型更容易被带偏、产生幻觉")


if __name__ == "__main__":
    main()
