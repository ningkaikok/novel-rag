#!/usr/bin/env python3
"""忠实度影子评测：对比规则基线 / 多个 LLM Judge 与人工标注的差异。

用法：
    python scripts/eval_faithfulness_shadow.py                # 只跑规则基线，零成本
    python scripts/eval_faithfulness_shadow.py --model glm:glm-4-flash
    python scripts/eval_faithfulness_shadow.py \\
        --model rule --model glm:glm-4-flash --model glm:glm-4.5-air   # 多 Judge 对比
    python scripts/eval_faithfulness_shadow.py \\
        --model glm:glm-4-flash --model two:glm:glm-4-flash   # 同一模型单步 vs 两步走
    python scripts/eval_faithfulness_shadow.py --two-step --model claude:sonnet

**影子评测结果不阻塞任何线上回答**：本脚本只读 tests/citation_shadow_set.json，
把每个方法（规则基线和/或若干 LLM Judge）的自动判断与人工标签的混淆矩阵、
跨方法汇总指标表和方法间分歧样本打印出来。第一阶段只记录差异、积累对 Judge
行为的直觉；达到明确阈值之前，忠实度判断绝不进入问答主链路
（见 docs/roadmap.md「M3.5」）。

标注集里的证据全部来自仓库原创语料（tests/ci_corpus/ 两篇短篇和 data/novels/
雾隐山庄.txt），不含任何版权小说原文。人工标签是 supported/partial/unsupported
三档；单步 Judge 输出 supported/unsupported/uncertain，两步走 Judge 靠逐条
断言判定的机械聚合还能产出 partial——uncertain 不是错误，它会被单独列进
矩阵，供后续决定"不确定提示/拒答"策略时参考。
"""

import argparse
import json
import os
import signal
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# backend.* 的模型客户端按项目根导入；独立跑脚本时还要先加载 .env，
# 否则 ZHIPU_API_KEY 缺失会让 glm: 路径全部失败——这是 ingest.py 踩过的坑
sys.path.insert(0, str(ROOT))

from citation_eval import (  # noqa: E402
    judge_support,
    judge_support_two_step,
    rule_support,
)

# 人工标签全集（partial 只出现在人工标注里）+ Judge 的 uncertain
HUMAN_LABELS = ("supported", "partial", "unsupported")
PREDICTED_LABELS = ("supported", "partial", "unsupported", "uncertain")

# 两步走 Judge 的方法名前缀：two:glm:glm-4-flash 表示用两步走路径评测 glm:glm-4-flash。
# 与 --two-step / FAITHFULNESS_JUDGE_MODE 的区别是粒度——前缀可以只给某一个方法开两步走，
# 同一次运行里就能对比同一模型的单步 vs 两步。
TWO_STEP_PREFIX = "two:"

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

        # GLM 用流级看门狗（见 zhipu.generate_stream 的 ZHIPU_STREAM_DEADLINE）：
        # SIGALRM 打不断 macOS 上 _ssl 在 C 层的 poll 重试循环，只有从其他线程
        # 强制关闭响应才能可靠打断僵尸 SSE 流。setdefault 允许环境显式覆盖。
        os.environ.setdefault("ZHIPU_STREAM_DEADLINE", str(int(_CALL_DEADLINE_S)))
        return lambda prompt: zhipu.generate_stream(prompt, model)
    if model.startswith("claude:"):
        from backend import claude_cli

        return _with_deadline(lambda prompt: claude_cli.generate_stream(prompt, model))
    raise SystemExit(f"不支持的模型前缀：{model}（影子评测只支持 glm:/claude: 前缀）")


def parse_method_spec(spec: str) -> tuple[str, str, bool]:
    """解析方法描述符，返回 ``(展示名, 路由用的模型 spec, 是否两步走)``。

    规则：
    - ``rule``：规则基线，永远单步；
    - ``two:<model>``：显式给该方法开两步走；
    - 其余 glm:/claude: 前缀模型按环境变量 ``FAITHFULNESS_JUDGE_MODE``
      （取值 two_step/two/1/true）决定默认模式——env 提供全局开关，
      two: 前缀提供逐方法覆盖。
    """
    mode_env = os.environ.get("FAITHFULNESS_JUDGE_MODE", "").strip().lower()
    env_two_step = mode_env in ("two_step", "two", "1", "true")
    if spec.startswith(TWO_STEP_PREFIX):
        base = spec[len(TWO_STEP_PREFIX) :]
        if base == "rule" or not base.startswith(("glm:", "claude:")):
            raise SystemExit(f"两步走只支持 glm:/claude: 前缀模型，收到：{spec}")
        return spec, base, True
    if spec != "rule" and not spec.startswith(("glm:", "claude:")):
        raise SystemExit(f"不支持的方法：{spec}（可选 rule 或 [two:]glm:/claude: 前缀模型）")
    return spec, spec, env_two_step


_last_call_at = 0.0


def _throttle() -> None:
    """限速闸：任意两次 LLM 调用之间至少间隔 _MIN_CALL_INTERVAL_S 秒。"""
    global _last_call_at
    wait = _MIN_CALL_INTERVAL_S - (time.monotonic() - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()


# 单次 LLM 调用的墙钟上限。requests 的读超时只约束「两个数据块之间」的间隔——
# 本机代理（Surge/Clash fake-ip）的隧道卡死时仍会间歇性喂入 TLS 心跳，把
# SSE 流挂成永远读不完也永不超时的僵尸连接（实测一次挂 25 分钟）。墙钟
# 看门狗不依赖对端行为，到点强制打断。
_CALL_DEADLINE_S = float(os.environ.get("FAITHFULNESS_CALL_DEADLINE", "120"))


class _CallTimeout(Exception):
    """单次生成调用超过墙钟上限（区别于网络层异常，但同属可重试故障）。"""


@contextmanager
def _call_deadline(seconds: float = _CALL_DEADLINE_S):
    """SIGALRM 墙钟看门狗：限定单次生成调用的真实耗时，超时抛 _CallTimeout。

    只能在主线程使用（signal 限制）——影子评测脚本恰好是单线程主循环，
    够用且零依赖。退出时无条件恢复原定时器与原处理器。
    """

    def _on_alarm(signum, frame):
        raise _CallTimeout(f"单次调用超过 {seconds:g}s 墙钟上限（疑似代理/网络挂起）")

    old_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _with_deadline(generate_fn):
    """给生成函数套上墙钟看门狗。

    抛出的 _CallTimeout 会被 citation_eval 两层 except Exception 捕获并降级为
    带「Judge 调用失败：」前缀的 uncertain——正好落在既有可重试语义里，
    上层 judge_with_retry 的退避重试无需任何改动即可接管。
    """

    def run(prompt: str):
        with _call_deadline():
            return generate_fn(prompt)

    return run


def judge_with_retry(
    case: dict,
    generate_fn,
    max_retries: int = _MAX_RETRIES,
    two_step: bool = False,
    batch_verdicts: bool = False,
) -> tuple[dict, bool]:
    """带限速与重试的单条 Judge 调用（单步/两步走共用）。

    返回 ``(result, failed)``：result 是 judge_support /
    judge_support_two_step 的完整返回（至少含 label/reason；两步走另附
    claims/verdicts 供误判分析）。只有报告的**调用级故障**（生成异常 /
    输出解析不出 JSON，见 _RETRYABLE_PREFIXES，两种模式同前缀）才会重试；
    模型正常回答的 uncertain 是它的真实判断，直接采纳。重试耗尽仍失败则
    保留最后一次结果并置 ``failed=True`` 供调用方计数。
    """
    judge_fn = judge_support_two_step if two_step else judge_support
    result: dict = {"label": "uncertain", "reason": ""}
    for attempt in range(max_retries + 1):
        _throttle()
        result = (
            judge_fn(case["statement"], case["evidence"], generate_fn, batch_verdicts=True)
            if two_step and batch_verdicts
            else judge_fn(case["statement"], case["evidence"], generate_fn)
        )
        if not result["reason"].startswith(_RETRYABLE_PREFIXES):
            return result, False
        if attempt < max_retries:
            # 线性退避，避免连续撞上同一故障窗口
            time.sleep(0.5 * (attempt + 1))
    return result, True


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


def per_label_agreement(rows: list[tuple[str, str]]) -> dict[str, tuple[int, int]]:
    """逐人工标签统计一致数：partial 类的改善是两步走的核心观察点。

    返回 ``{human_label: (correct, total)}``。整体一致率会被多数类稀释，
    这里按标签拆开看——尤其 partial：单步 Judge 全部退化成二分类器时，
    这一列接近 0/14；两步走若有效，partial 列应该明显抬升。
    """
    stats: dict[str, tuple[int, int]] = {}
    for human in HUMAN_LABELS:
        subset = [(p, h) for p, h in rows if h == human]
        correct = sum(1 for p, h in subset if p == human)
        stats[human] = (correct, len(subset))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "评测方法，可重复传入做横向对比：rule 表示规则基线，其余为 "
            "[two:]glm:/claude: 前缀的模型名——加 two: 前缀表示该方法走"
            "「先抽断言再逐条核对」的两步走路径（如 two:glm:glm-4-flash）。"
            "不传该参数时只跑规则基线（与旧版单模型用法向后兼容）"
        ),
    )
    parser.add_argument(
        "--two-step",
        action="store_true",
        help="把本次所有未带 two: 前缀的 LLM 方法都切换为两步走 Judge"
        "（也可用环境变量 FAITHFULNESS_JUDGE_MODE=two_step 达到同样效果）",
    )
    parser.add_argument(
        "--dump-json",
        metavar="PATH",
        default=None,
        help="把逐条自动判定（含两步走的断言与逐条核对结果）写入 JSON 文件，"
        "供聚合规则离线重算——改聚合逻辑不需要重跑模型调用",
    )
    args = parser.parse_args()

    specs = list(dict.fromkeys(args.model)) if args.model else []
    if args.two_step:
        os.environ["FAITHFULNESS_JUDGE_MODE"] = "two_step"
    # 提前解析全部方法描述符：非法前缀在花钱之前就报错。
    # 每个 plan 是 (展示名, 路由模型名, 是否两步走)
    method_plans = [parse_method_spec(spec) for spec in specs or ["rule"]]

    cases = load_cases()
    print("=" * 64)
    print("【影子评测】结果仅用于离线观察，不阻塞任何线上回答")
    print("=" * 64)

    llm_plans = [plan for plan in method_plans if plan[0] != "rule"]
    if llm_plans:
        mode_note = "、".join(
            f"{display}(两步走)" if two else display for display, _, two in llm_plans
        )
        print(
            f"LLM Judge 模型：{mode_note}"
            f"（每条最多 {1 + _MAX_RETRIES} 次真实调用，两步走按断言数翻倍，"
            f"间隔≥{_MIN_CALL_INTERVAL_S}s，注意费用）"
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
        label_stats = per_label_agreement(rows)
        stats_text = "  ".join(
            f"{label} {correct_n}/{total}"
            for label, (correct_n, total) in label_stats.items()
            if total
        )
        print(f"逐人工标签一致：{stats_text}")

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
                claims = item.get("claims") or []
                verdict_map = {v["claim"]: v["label"] for v in item.get("verdicts") or []}
                if claims:
                    # 两步走的中间过程是误判分析的核心素材：断言怎么拆的、
                    # 每条拆出来的断言被判成什么
                    print(f"    断言拆解（{len(claims)} 条）：")
                    for order, claim in enumerate(claims, start=1):
                        print(f"      {order}. [{verdict_map.get(claim, '?')}] {claim}")
                if item.get("note"):
                    print(f"    标注备注:{item['note']}")
        else:
            print("\n没有误判。（规则基线也能全对时，反而要警惕标注集太简单）")

    # ---- 逐方法评测并输出各自的混淆矩阵 ----
    labels_by_spec: dict[str, dict[str, str]] = {"rule": rule_labels}
    results_by_spec: dict[str, dict[str, dict]] = {}
    failed_counts: Counter = Counter()
    failure_notes: list[str] = []

    for display, base, two in method_plans:
        if display == "rule":
            show_method(display, rows_rule, mismatches_rule)
            continue
        generate_fn = _build_generate_fn(base)
        print(f"\n>>> 开始评测方法：{display}{'（两步走）' if two else ''}", flush=True)
        rows: list[tuple[str, str]] = []
        mismatches: list[dict] = []
        labels_by_spec[display] = {}
        results_by_spec[display] = {}
        for index, case in enumerate(cases, start=1):
            result, failed = judge_with_retry(
                case, generate_fn, two_step=two, batch_verdicts=batch_env
            )
            label = result["label"]
            if failed:
                failed_counts[display] += 1
                failure_notes.append(f"{display} [{case['id']}] {result['reason']}")
            labels_by_spec[display][case["id"]] = label
            results_by_spec[display][case["id"]] = result
            human = case["human_label"]
            rows.append((label, human))
            if label != human:
                mismatches.append({**case, "predicted": label, "reason": result["reason"]})
            progress = "!" if failed else ""
            extra = ""
            claims = result.get("claims")
            if two and claims:
                extra = f"（断言x{len(claims)}）"
            print(
                f"    [{index}/{len(cases)}] {case['id']} -> {label}{extra}{progress}",
                flush=True,
            )
        show_method(display, rows, mismatches)

    # ---- 跨方法汇总表 ----
    table_specs = [plan[0] for plan in method_plans]
    print(f"\n{'=' * 64}\n跨方法汇总\n{'=' * 64}")
    columns = ("一致率", "支持P", "支持R", "反对P", "反对R", "uncertain%", "调用失败")
    header = "方法".ljust(20) + "".join(col.ljust(11) for col in columns)
    print(header)
    print("-" * len(header))
    agreement_by_spec: dict[str, dict[str, tuple[int, int]]] = {}
    for spec in table_specs:
        labels_map = labels_by_spec[spec]
        rows = [(labels_map[c["id"]], c["human_label"]) for c in cases]
        correct = sum(1 for predicted, human in rows if predicted == human)
        sup_p, sup_r = binary_pr(rows, "supported")
        uns_p, uns_r = binary_pr(rows, "unsupported")
        uncertain_ratio = sum(1 for predicted, _ in rows if predicted == "uncertain") / len(
            rows
        )
        agreement_by_spec[spec] = per_label_agreement(rows)

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

    # ---- 逐人工标签一致率（partial 列是两步走是否突破盲区的关键观察点）----
    print(f"\n{'=' * 64}\n逐人工标签一致率\n{'=' * 64}")
    label_header = "方法".ljust(20) + "".join(label.ljust(14) for label in HUMAN_LABELS)
    print(label_header)
    print("-" * len(label_header))
    for spec in table_specs:
        cells = [
            f"{correct_n}/{total}"
            for correct_n, total in (agreement_by_spec[spec][label] for label in HUMAN_LABELS)
        ]
        print(spec.ljust(20) + "".join(cell.ljust(14) for cell in cells))

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

    if args.dump_json:
        dump = {
            method: {
                case_id: {**result, "human_label": human}
                for case_id, result in results.items()
            }
            for method, results in results_by_spec.items()
        }
        Path(args.dump_json).write_text(
            json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"逐条判定已写入 {args.dump_json}（聚合规则可离线重算，不必重跑模型）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
