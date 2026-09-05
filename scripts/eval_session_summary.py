#!/usr/bin/env python3
"""离线评测滚动会话摘要（M3.6 第三步的开关证据）。

运行：
    # 只跑不需要模型的部分（结构性事实丢失证明），零成本
    python scripts/eval_session_summary.py

    # 跑完整摘要评测（需要生成后端）
    python scripts/eval_session_summary.py --model glm:glm-4-flash

docs/experiments/m36-session-summary.md 定义了打开 HISTORY_SUMMARY_ENABLED
默认值需要的三类证据：事实丢失（收益）、摘要漂移（代价）、引用正确性。这个
脚本只测前两类——引用正确性由 build_history_block 里的 `_strip_citations`
机械保证（背景段里的 [n] 一律被抹掉），属于代码层面的强约束，比一次评测的
"大概率成立"更可靠，不需要重复用实验去验证一个已经由代码保证的性质。

为什么断言仍是确定性的字符串包含/排除，而不是 LLM 评委：和 eval_multiturn.py
同样的理由——摘要的产物是一段文本，它"有没有提到某个事实""有没有编造某个
事实"都可以被机械验证，引入评委只会把一个可判定的问题变成又一个需要校准的
问题（M3.5 的教训：Judge 自己的一致率只有 67.9%）。

两层证据：

    结构性证明（零成本，不调模型）
        不开摘要时，超出 HISTORY_MAX_TURNS 窗口的早期事实在组装好的「对话
        背景」文本里**必然不存在**——这是 build_history_block 的截断逻辑
        决定的，不用跑模型就能证明"没有摘要 = 事实必然丢失"。

    真实摘要评测（需要模型，见 --model）
        对同一批早期轮次跑一次 build_summary，检查产出的摘要：
        - expect_contains：早期事实必须被摘要记住（收益）
        - expect_absent：摘要不能提到 early_turns 里没说过的内容（代价，
          防止模型把"还没发生的情节"脑补成摘要的一部分）
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config import HISTORY_MAX_TURNS  # noqa: E402
from generation_mixin import build_history_block  # noqa: E402
from session_summary import build_summary  # noqa: E402

TEST_SET = ROOT / "tests" / "session_summary_test_set.json"


def build_generate_fn(model: str):
    """按模型前缀路由到生成后端。与 eval_multiturn.py 同一套模式。"""
    from backend.dotenv_lite import load_env

    load_env(ROOT / ".env")
    if model.startswith("glm:"):
        from backend import zhipu

        return lambda prompt: zhipu.generate_stream(prompt, model)
    if model.startswith("claude:"):
        from backend import claude_cli

        return lambda prompt: claude_cli.generate_stream(prompt, model)
    raise SystemExit(f"不支持的模型前缀：{model}（只支持 glm:/claude:）")


def check(text: str, case: dict) -> list[str]:
    problems = []
    for token in case.get("expect_contains", []):
        if token not in text:
            problems.append(f"缺少「{token}」")
    for token in case.get("expect_absent", []):
        if token in text:
            problems.append(f"不该出现「{token}」")
    return problems


def report_structural_only(cases: list[dict]) -> int:
    """零成本证明：不开摘要时，早期事实必然不在组装好的背景里。

    这不是"大概率丢失"，是 build_history_block 的截断逻辑决定的必然结果——
    窗口只保留最近 HISTORY_MAX_TURNS 轮，早期轮次一旦被 recent_turns 挤出窗口，
    组装出的文本里不可能再出现它。用真实的 build_history_block 跑一遍，
    确认这条推理没有因为代码改动而失效。
    """
    print("只做结构性证明（未调用模型）。加 --model 跑完整摘要评测。\n")
    ok = 0
    for case in cases:
        all_turns = case["early_turns"] + case["recent_turns"]
        assert len(case["recent_turns"]) >= HISTORY_MAX_TURNS, (
            f"[{case['id']}] recent_turns 少于 HISTORY_MAX_TURNS={HISTORY_MAX_TURNS}，"
            "early_turns 不一定会被挤出窗口，测试前提不成立"
        )
        text, _ = build_history_block(all_turns)  # summary=None：不开摘要
        lost = [tok for tok in case["expect_contains"] if tok in text]
        if lost:
            print(f"  ⚠️ [{case['id']}] 早期事实仍在窗口内（{lost}），检查测试用例长度")
        else:
            ok += 1
    print(f"\n{ok}/{len(cases)} 条用例确认：不开摘要时早期事实必然从背景里消失")
    print("（这正是滚动摘要要解决的问题——收益能有多大，还要看 --model 那一半）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="摘要用的模型（glm:/claude: 前缀）。不传则只做零成本的结构性证明",
    )
    args = parser.parse_args()

    data = json.loads(TEST_SET.read_text(encoding="utf-8"))
    cases = data["cases"]

    if args.model is None:
        return report_structural_only(cases)

    generate_fn = build_generate_fn(args.model)
    rows = []
    for case in cases:
        errors: list[str] = []
        summary = build_summary(None, case["early_turns"], generate_fn, errors=errors)
        summary = summary or ""
        rows.append(
            {
                "case": case,
                "summary": summary,
                "problems": check(summary, case),
                "errors": errors,
            }
        )
    return report(rows, args.model)


def report(rows: list[dict], model: str) -> int:
    passed = [r for r in rows if not r["problems"]]
    print(f"摘要模型：{model}")
    print(f"通过率：{len(passed)}/{len(rows)} = {len(passed) / len(rows):.1%}\n")

    failures = [r for r in rows if r["problems"]]
    if failures:
        print(f"未通过（{len(failures)} 条）：")
        for row in failures:
            case = row["case"]
            print(f"\n  [{case['id']}] {case['book']}")
            print(f"    早期事实：{case['early_turns'][-1]['content']}")
            print(f"    生成的摘要：{row['summary']!r}")
            print(f"    问题：{'；'.join(row['problems'])}")
            print(f"    这条在测什么：{case['note']}")
            if row["errors"]:
                print(f"    调用错误：{row['errors']}")

    call_errors = [r for r in rows if r["errors"]]
    if call_errors:
        print(f"\n⚠️ {len(call_errors)} 条发生过调用/解析失败")

    print(
        "\n判读建议（写进 docs/experiments/m36-session-summary.md）：\n"
        "  收益为正（expect_contains 大量通过）且代价接近 0（expect_absent 无违反）\n"
        "  才把 HISTORY_SUMMARY_ENABLED 的默认值改成 1；任何一条 expect_absent 违反\n"
        "  都是真实发生过的摘要漂移，不能被总体通过率稀释掉。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
