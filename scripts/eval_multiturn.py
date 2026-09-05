#!/usr/bin/env python3
"""离线评测多轮追问的查询改写（M3.6 第一步）。

运行：
    # 只跑不需要模型的部分（触发判断），零成本
    python scripts/eval_multiturn.py

    # 跑完整改写评测（需要生成后端）
    python scripts/eval_multiturn.py --model glm:glm-4-flash

评的是一件事：**带着上文去理解这句追问时，系统有没有指到正确的对象。**

为什么断言是确定性的字符串包含/排除，而不是 LLM 评委：改写的产物是"拿去检索的
问题"，它的成败可以被机械验证——「他后来怎么样了」改写后必须出现「陆知微」，
问另一本书时必须不出现上一本书的实体。引入评委只会把一个可判定的问题变成又一个
需要校准的问题（M3.5 的教训：Judge 自己的一致率只有 67.9%）。

两类断言：
    expect_contains —— 改写后必须出现（通常是被解析出来的实体）
    expect_absent   —— 必须不出现（通常是上文里那个错误的对象）

「不该改写」类别的用例考察相反的方向：问题本身已自足时，硬改会把上文的无关实体
拼进检索词，反而污染结果。这类用例靠 expect_absent 抓。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from query_rewriter import needs_rewrite, rewrite_query  # noqa: E402

TEST_SET = ROOT / "tests" / "multiturn_test_set.json"


def build_generate_fn(model: str):
    """按模型前缀路由到生成后端。与影子评测脚本同一套模式。"""
    from backend.dotenv_lite import load_env

    load_env(ROOT / ".env")
    if model.startswith("glm:"):
        from backend import zhipu

        return lambda prompt: zhipu.generate_stream(prompt, model)
    if model.startswith("claude:"):
        from backend import claude_cli

        return lambda prompt: claude_cli.generate_stream(prompt, model)
    raise SystemExit(f"不支持的模型前缀：{model}（只支持 glm:/claude:）")


def check(rewritten: str, case: dict) -> list[str]:
    """返回这条用例的失败原因列表；空列表表示通过。"""
    problems = []
    for token in case.get("expect_contains", []):
        if token not in rewritten:
            problems.append(f"缺少「{token}」")
    for token in case.get("expect_absent", []):
        if token in rewritten:
            problems.append(f"不该出现「{token}」")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="改写用的模型（glm:/claude: 前缀）。不传则只检查触发判断，不调模型",
    )
    args = parser.parse_args()

    data = json.loads(TEST_SET.read_text(encoding="utf-8"))
    cases = data["cases"]

    if args.model is None:
        return report_trigger_only(cases)

    generate_fn = build_generate_fn(args.model)
    rows = []
    for case in cases:
        errors: list[str] = []
        rewritten = rewrite_query(case["question"], case["history"], generate_fn, errors)
        rows.append(
            {
                "case": case,
                "rewritten": rewritten,
                "unchanged": rewritten == case["question"],
                "problems": check(rewritten, case),
                "errors": errors,
            }
        )
    return report(rows, args.model)


def report_trigger_only(cases: list[dict]) -> int:
    """不调模型时的轻量报告：只看 needs_rewrite 的触发判断。

    触发判断本身就是一道闸——该改写的没触发，后面再好的模型也救不回来；
    不该改写的触发了，则是白花一次调用并引入跑偏风险。
    """
    print("只检查触发判断（未调用模型）。加 --model 跑完整改写评测。\n")
    by_category: Counter[str] = Counter()
    ok_by_category: Counter[str] = Counter()
    wrong: list[dict] = []
    for case in cases:
        fires = needs_rewrite(case["question"], has_history=bool(case["history"]))
        expected = case["should_rewrite"]
        by_category[case["category"]] += 1
        if fires == expected:
            ok_by_category[case["category"]] += 1
        else:
            wrong.append(case)

    total = len(cases)
    correct = total - len(wrong)
    print(f"触发判断准确率：{correct}/{total} = {correct / total:.1%}\n")
    print("分类：")
    for category in sorted(by_category):
        print(f"  {category:<10} {ok_by_category[category]}/{by_category[category]}")

    if wrong:
        print(f"\n判断错误（{len(wrong)} 条）——注意这只是必要条件，改得对不对还要跑 --model：")
        for case in wrong:
            direction = "该改写却没触发" if case["should_rewrite"] else "不该改写却触发了"
            print(f"\n  [{case['id']}] {case['category']}：{direction}")
            print(f"    {case['question']}")
            print(f"    为什么该这样：{case['note']}")
    return 0


def report(rows: list[dict], model: str) -> int:
    passed = [r for r in rows if not r["problems"]]
    by_category: Counter[str] = Counter()
    ok_by_category: Counter[str] = Counter()
    for row in rows:
        category = row["case"]["category"]
        by_category[category] += 1
        if not row["problems"]:
            ok_by_category[category] += 1

    print(f"改写模型：{model}")
    print(f"通过率：{len(passed)}/{len(rows)} = {len(passed) / len(rows):.1%}\n")

    print("分类通过率：")
    for category in sorted(by_category):
        total = by_category[category]
        ok = ok_by_category[category]
        print(f"  {category:<10} {ok}/{total} = {ok / total:.0%}")

    failures = [r for r in rows if r["problems"]]
    if failures:
        print(f"\n未通过（{len(failures)} 条）：")
        for row in failures:
            case = row["case"]
            print(f"\n  [{case['id']}] {case['category']}")
            print(f"    原问题：{case['question']}")
            print(f"    改写后：{row['rewritten']}")
            print(f"    问题：{'；'.join(row['problems'])}")
            print(f"    这条在测什么：{case['note']}")
            if row["errors"]:
                print(f"    调用错误：{row['errors']}")

    call_errors = [r for r in rows if r["errors"]]
    if call_errors:
        print(f"\n⚠️ {len(call_errors)} 条发生过调用/解析失败（已降级为原问题）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
