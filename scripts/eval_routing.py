#!/usr/bin/env python3
"""离线评测自动问答路由，不启动数据库、后端或任何模型。

运行：
    python scripts/eval_routing.py

新增/修改规则前先补 ``tests/routing_test_set.json``，再运行本脚本。路由的目标不是
猜中所有自然语言，而是：明确开放的问题可以跳过检索，拿不准的小说人物问题宁可
保守检索，并且用户始终能用模式选择器覆盖自动判断。
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from query_router import AnswerMode, choose_answer_route  # noqa: E402


def main() -> int:
    cases = json.loads(
        (ROOT / "tests" / "routing_test_set.json").read_text(encoding="utf-8")
    )
    errors = []
    category_total: Counter[str] = Counter()
    category_correct: Counter[str] = Counter()

    for case in cases:
        actual = choose_answer_route(case["question"], AnswerMode.auto).route.value
        category = case["category"]
        category_total[category] += 1
        if actual == case["expected"]:
            category_correct[category] += 1
        else:
            errors.append({**case, "actual": actual})

    correct = len(cases) - len(errors)
    print(f"路由准确率：{correct}/{len(cases)} = {correct / len(cases):.1%}")
    for category in sorted(category_total):
        print(
            f"  {category}: {category_correct[category]}/{category_total[category]}"
        )

    if errors:
        print("\n误判：")
        for case in errors:
            print(
                f"  [{case['id']}] {case['question']} "
                f"expected={case['expected']} actual={case['actual']}"
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
