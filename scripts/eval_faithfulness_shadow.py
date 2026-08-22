#!/usr/bin/env python3
"""忠实度影子评测：对比规则基线 / LLM Judge 与人工标注的差异。

用法：
    python scripts/eval_faithfulness_shadow.py                # 只跑规则基线，零成本
    python scripts/eval_faithfulness_shadow.py --model glm:glm-4-flash

**影子评测结果不阻塞任何线上回答**：本脚本只读 tests/citation_shadow_set.json，
把自动判断（规则基线 + 可选 LLM Judge）与人工标签的混淆矩阵和误判清单打印出来。
第一阶段只记录差异、积累对 Judge 行为的直觉；达到明确阈值之前，忠实度判断绝不
进入问答主链路（见 docs/roadmap.md「M3.5」）。

标注集里的证据全部来自仓库原创语料（tests/ci_corpus/ 两篇短篇和 data/novels/
雾隐山庄.txt），不含任何版权小说原文。人工标签是 supported/partial/unsupported
三档；LLM Judge 输出 supported/unsupported/uncertain——uncertain 不是错误，
它会被单独列进矩阵，供后续决定"不确定提示/拒答"策略时参考。
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# backend.* 的模型客户端按项目根导入；独立跑脚本时还要先加载 .env，
# 否则 ZHIPU_API_KEY 缺失会让 glm: 路径全部失败——这是 ingest.py 踩过的坑
sys.path.insert(0, str(ROOT))

from citation_eval import judge_support, rule_support  # noqa: E402

# 人工标签全集（partial 只出现在人工标注里）+ Judge 的 uncertain
HUMAN_LABELS = ("supported", "partial", "unsupported")
PREDICTED_LABELS = ("supported", "partial", "unsupported", "uncertain")

_SHADOW_SET_PATH = ROOT / "tests" / "citation_shadow_set.json"


def load_cases(path: Path = _SHADOW_SET_PATH) -> list[dict]:
    """读取标注集；跳过 _readme 之类的说明字段。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["cases"]


def build_confusion_matrix(rows: list[tuple[str, str]]) -> dict[str, Counter]:
    """从 (predicted, human) 对构建混淆矩阵。

    返回 {predicted: Counter({human_label: count})}，行是预测、列是人工标签。
    抽成纯函数方便单测（测试用 mock 数据，不跑任何模型）。
    """
    matrix: dict[str, Counter] = {label: Counter() for label in PREDICTED_LABELS}
    for predicted, human in rows:
        matrix.setdefault(predicted, Counter())[human] += 1
    return matrix


def _format_matrix(matrix: dict[str, Counter]) -> str:
    header = "预测 \\ 人工".ljust(14) + "".join(label.ljust(13) for label in HUMAN_LABELS)
    lines = [header]
    for predicted in PREDICTED_LABELS:
        counter = matrix.get(predicted, Counter())
        cells = "".join(str(counter.get(label, 0)).ljust(13) for label in HUMAN_LABELS)
        total = sum(counter.values())
        lines.append(f"{predicted:<14}{cells}{total}")
    return "\n".join(lines)


def _build_generate_fn(model: str):
    """按模型名前缀路由到生成后端（与 backend/main.py 的路由保持同一约定）。"""
    from backend.dotenv_lite import load_env

    load_env(ROOT / ".env")
    if model.startswith("glm:"):
        from backend import zhipu

        return lambda prompt: zhipu.generate_stream(prompt, model)
    if model.startswith("claude:"):
        from backend import claude_cli

        return lambda prompt: claude_cli.generate_stream(prompt, model)
    raise SystemExit(f"不支持的模型前缀：{model}（影子评测只支持 glm:/claude: 前缀）")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="可选的 LLM Judge 模型（glm:/claude: 前缀）。默认只跑规则基线，不调用任何模型",
    )
    args = parser.parse_args()

    cases = load_cases()
    print("=" * 64)
    print("【影子评测】结果仅用于离线观察，不阻塞任何线上回答")
    print("=" * 64)

    generate_fn = None
    if args.model:
        generate_fn = _build_generate_fn(args.model)
        print(f"LLM Judge 模型：{args.model}（每条一次真实调用，注意费用）")
    else:
        print("未指定 --model：只跑规则基线，不调用任何 LLM")

    method = "judge" if generate_fn is not None else "rule"
    rows: list[tuple[str, str]] = []
    mismatches: list[dict] = []
    errors: list[str] = []
    for case in cases:
        rule_result = rule_support(case["statement"], case["evidence"])
        human = case["human_label"]
        if generate_fn is not None:
            result = judge_support(
                case["statement"], case["evidence"], generate_fn, errors=errors
            )
        else:
            result = rule_result
        rows.append((result["label"], human))
        if result["label"] != human:
            mismatches.append({**case, "predicted": result["label"], "reason": result["reason"], "rule": rule_result["label"]})

    # ---- 混淆矩阵：行=预测，列=人工标签 ----
    matrix = build_confusion_matrix(rows)
    print(f"\n样本数：{len(cases)}   方法：{method}")
    print("\n混淆矩阵（行=自动判断，列=人工标签）：")
    print(_format_matrix(matrix))

    correct = sum(1 for predicted, human in rows if predicted == human)
    print(f"\n与人工标签一致：{correct}/{len(rows)} = {correct / len(rows):.1%}")

    if mismatches:
        print(f"\n误判清单（{len(mismatches)} 条）：")
        for item in mismatches:
            evidence_preview = (item["evidence"][0][:60] + "…") if item["evidence"] else ""
            print(f"\n  [{item['id']}] 类别={item['category']}")
            print(f"    陈述：{item['statement']}")
            print(f"    证据：{evidence_preview}")
            print(
                f"    预测={item['predicted']}  人工={item['human_label']}"
                f"  规则基线={item['rule']}"
            )
            if item.get("note"):
                print(f"    标注备注：{item['note']}")
    else:
        print("\n没有误判。（规则基线也能全对时，反而要警惕标注集太简单）")

    if errors:
        print(f"\nJudge 调用异常（已降级为 uncertain，共 {len(errors)} 条）：")
        for reason in errors[:5]:
            print(f"  - {reason[:160]}")

    # 影子评测永远返回 0：误判多是预期内的观察结果，不该让 CI 变红。
    print("\n再次提醒：影子评测结果不阻塞任何线上回答；阈值成熟前不接入问答主链路。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
