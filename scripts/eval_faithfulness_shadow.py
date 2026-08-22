#!/usr/bin/env python3
"""忠实度影子评测：对比规则基线 / 多个 LLM Judge 与人工标注的差异。

用法：
    python scripts/eval_faithfulness_shadow.py                # 只跑规则基线，零成本
    python scripts/eval_faithfulness_shadow.py --model glm:glm-4-flash
    python scripts/eval_faithfulness_shadow.py \
        --model rule --model glm:glm-4-flash --model glm:glm-4.5-air   # 多 Judge 对比

**影子评测结果不阻塞任何线上回答**：本脚本只读 tests/citation_shadow_set.json，
把每个方法（规则基线和/或若干 LLM Judge）的自动判断与人工标签的混淆矩阵、
跨方法汇总指标表和方法间分歧样本打印出来。第一阶段只记录差异、积累对 Judge
行为的直觉；达到明确阈值之前，忠实度判断绝不进入问答主链路
（见 docs/roadmap.md「M3.5」）。

标注集里的证据全部来自仓库原创语料（tests/ci_corpus/ 两篇短篇和 data/novels/
雾隐山庄.txt），不含任何版权小说原文。人工标签是 supported/partial/unsupported
三档；LLM Judge 输出 supported/unsupported/uncertain——uncertain 不是错误，
它会被单独列进矩阵，供后续决定"不确定提示/拒答"策略时参考。
"""

import argparse
import json
import sys
import time
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

# LLM 调用保护：任意两次调用之间至少间隔 0.5s（限速）；单条最多重试 2 次，
# 重试后仍失败则降级为 uncertain 并计数——影子评测宁可"不确定"也不编造标签。
_MIN_CALL_INTERVAL_S = 0.5
_MAX_RETRIES = 2
# judge_support 返回这两种前缀时说明是可重试故障（网络异常 / 输出不可解析）；
# 正常返回的 uncertain 不重试，那是模型自己的判断。
_RETRYABLE_PREFIXES = ("Judge 调用失败：", "Judge 输出解析失败：")


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


_last_call_at = 0.0


def _throttle() -> None:
    """限速闸：任意两次 LLM 调用之间至少间隔 _MIN_CALL_INTERVAL_S 秒。"""
    global _last_call_at
    wait = _MIN_CALL_INTERVAL_S - (time.monotonic() - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()


def judge_with_retry(
    case: dict, generate_fn, max_retries: int = _MAX_RETRIES
) -> tuple[str, str, bool]:
    """带限速与重试的单条 Judge 调用。

    返回 ``(label, reason, failed)``。只有 judge_report 报告的**调用级故障**
    （生成异常 / 输出解析不出 JSON，见 _RETRYABLE_PREFIXES）才会重试；模型正常
    回答的 uncertain 是它的真实判断，直接采纳。重试耗尽仍失败则降级为
    uncertain 并置 ``failed=True`` 供调用方计数。
    """
    result: dict = {"label": "uncertain", "reason": ""}
    for attempt in range(max_retries + 1):
        _throttle()
        result = judge_support(case["statement"], case["evidence"], generate_fn)
        if not result["reason"].startswith(_RETRYABLE_PREFIXES):
            return result["label"], result["reason"], False
        if attempt < max_retries:
            # 线性退避，避免连续撞上同一故障窗口
            time.sleep(0.5 * (attempt + 1))
    return result["label"], result["reason"], True


def binary_pr(rows: list[tuple[str, str]], positive: str) -> tuple[float | None, float | None]:
    """one-vs-rest 二元精确率/召回率，供跨方法汇总表使用。

    partial 只出现在人工端：预测 supported 而人工 partial 记为假阳性——这是刻意
    设计，Judge 从不输出 partial，把它算进 supported 召回只会虚高指标。
    分母为 0 时返回 None（表格里显示为 —），与 0.0 区分开。
    """
    tp = sum(1 for p, h in rows if p == positive and h == positive)
    fp = sum(1 for p, h in rows if p == positive and h != positive)
    fn = sum(1 for p, h in rows if p != positive and h == positive)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return precision, recall


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "评测方法，可重复传入做横向对比：rule 表示规则基线，其余为 "
            "glm:/claude: 前缀的 LLM Judge 模型名。不传该参数时只跑规则基线"
            "（与旧版单模型用法向后兼容）"
        ),
    )
    args = parser.parse_args()

    specs = list(dict.fromkeys(args.model)) if args.model else []
    for spec in specs:
        if spec != "rule" and not spec.startswith(("glm:", "claude:")):
            raise SystemExit(f"不支持的方法：{spec}（可选 rule 或 glm:/claude: 前缀模型）")

    cases = load_cases()
    print("=" * 64)
    print("【影子评测】结果仅用于离线观察，不阻塞任何线上回答")
    print("=" * 64)

    llm_specs = [spec for spec in specs if spec != "rule"]
    if llm_specs:
        print(
            f"LLM Judge 模型：{'、'.join(llm_specs)}"
            f"（每条最多 {1 + _MAX_RETRIES} 次真实调用，间隔≥{_MIN_CALL_INTERVAL_S}s，注意费用）"
        )
    else:
        print("未指定 --model：只跑规则基线，不调用任何 LLM")

    # ---- 规则基线永远先算（零成本）：无论是否被要求展示，都作为对照参考列 ----
    rule_labels: dict[str, str] = {}
    rows_rule: list[tuple[str, str]] = []
    mismatches_rule: list[dict] = []
    for case in cases:
        result = rule_support(case["statement"], case["evidence"])
        label = result["label"]
        rule_labels[case["id"]] = label
        rows_rule.append((label, case["human_label"]))
        if label != case["human_label"]:
            mismatches_rule.append({**case, "predicted": label})

    def show_method(spec: str, rows: list[tuple[str, str]], mismatches: list[dict]) -> None:
        """打印单个方法的混淆矩阵 / 一致率 / 与人工标签不符的样本清单。"""
        matrix = build_confusion_matrix(rows)
        correct = sum(1 for predicted, human in rows if predicted == human)
        print(f"\n{'-' * 64}\n方法：{spec}   样本数:{len(rows)}")
        print("\n混淆矩阵（行=自动判断，列=人工标签）：")
        print(_format_matrix(matrix))
        print(f"\n与人工标签一致：{correct}/{len(rows)} = {correct / len(rows):.1%}")

        if mismatches:
            print(f"\n误判清单（{len(mismatches)} 条）：")
            for item in mismatches:
                evidence_preview = (item["evidence"][0][:60] + "…") if item["evidence"] else ""
                print(f"\n  [{item['id']}] 类别={item['category']}")
                print(f"    陈述:{item['statement']}")
                print(f"    证据:{evidence_preview}")
                print(
                    f"    预测={item['predicted']}  人工={item['human_label']}"
                    f"  规则基线={item.get('rule', rule_labels[item['id']])}"
                )
                reason = item.get("reason")
                if reason:
                    print(f"    Judge 理由:{reason[:120]}")
                if item.get("note"):
                    print(f"    标注备注:{item['note']}")
        else:
            print("\n没有误判。（规则基线也能全对时，反而要警惕标注集太简单）")

    # ---- 逐方法评测并输出各自的混淆矩阵 ----
    labels_by_spec: dict[str, dict[str, str]] = {"rule": rule_labels}
    failed_counts: Counter = Counter()
    failure_notes: list[str] = []

    for spec in specs or ["rule"]:
        if spec == "rule":
            show_method(spec, rows_rule, mismatches_rule)
            continue
        generate_fn = _build_generate_fn(spec)
        print(f"\n>>> 开始评测方法：{spec}")
        rows: list[tuple[str, str]] = []
        mismatches: list[dict] = []
        labels_by_spec[spec] = {}
        for index, case in enumerate(cases, start=1):
            label, reason, failed = judge_with_retry(case, generate_fn)
            if failed:
                failed_counts[spec] += 1
                failure_notes.append(f"{spec} [{case['id']}] {reason}")
            labels_by_spec[spec][case["id"]] = label
            human = case["human_label"]
            rows.append((label, human))
            if label != human:
                mismatches.append({**case, "predicted": label, "reason": reason})
            progress = "!" if failed else ""
            print(f"    [{index}/{len(cases)}] {case['id']} -> {label}{progress}")
        show_method(spec, rows, mismatches)

    # ---- 跨方法汇总表 ----
    table_specs = specs or ["rule"]
    print(f"\n{'=' * 64}\n跨方法汇总\n{'=' * 64}")
    columns = ("一致率", "支持P", "支持R", "反对P", "反对R", "uncertain%", "调用失败")
    header = "方法".ljust(20) + "".join(col.ljust(11) for col in columns)
    print(header)
    print("-" * len(header))
    for spec in table_specs:
        labels_map = labels_by_spec[spec]
        rows = [(labels_map[c["id"]], c["human_label"]) for c in cases]
        correct = sum(1 for predicted, human in rows if predicted == human)
        sup_p, sup_r = binary_pr(rows, "supported")
        uns_p, uns_r = binary_pr(rows, "unsupported")
        uncertain_ratio = sum(1 for predicted, _ in rows if predicted == "uncertain") / len(
            rows
        )

        def fmt(value: float | None) -> str:
            return f"{value:.0%}" if value is not None else "—"

        cells = [
            f"{correct / len(rows):.1%}",
            fmt(sup_p),
            fmt(sup_r),
            fmt(uns_p),
            fmt(uns_r),
            f"{uncertain_ratio:.1%}",
            str(failed_counts.get(spec, 0)),
        ]
        print(spec.ljust(20) + "".join(cell.ljust(11) for cell in cells))

    # ---- 方法间分歧样本 ----
    if len(table_specs) > 1:
        disputes: list[tuple[dict, dict[str, str]]] = []
        for case in cases:
            votes = {spec: labels_by_spec[spec][case["id"]] for spec in table_specs}
            if len(set(votes.values())) > 1:
                disputes.append((case, votes))
        print(f"\n方法间分歧样本（{len(disputes)} 条）：")
        for case, votes in disputes:
            vote_text = "  ".join(f"{spec}={label}" for spec, label in votes.items())
            evidence_preview = (case["evidence"][0][:50] + "…") if case["evidence"] else ""
            print(f"\n  [{case['id']}] 类别={case['category']}  人工={case['human_label']}")
            print(f"    陈述:{case['statement']}")
            print(f"    各方法:{vote_text}")
            print(f"    证据:{evidence_preview}")
            if case.get("rationale"):
                print(f"    标注理由:{case['rationale'][:120]}")
            elif case.get("note"):
                print(f"    标注备注:{case['note']}")
    else:
        print("\n只评测了一个方法，无方法间分歧可列。")

    if failure_notes:
        print(
            f"\nJudge 调用异常（重试 {_MAX_RETRIES} 次后仍失败，已降级 uncertain，"
            f"共 {sum(failed_counts.values())} 条）："
        )
        for note in failure_notes[:5]:
            print(f"  - {note[:160]}")

    # 影子评测永远返回 0：误判多是预期内的观察结果，不该让 CI 变红。
    print("\n再次提醒：影子评测结果不阻塞任何线上回答；阈值成熟前不接入问答主链路。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
