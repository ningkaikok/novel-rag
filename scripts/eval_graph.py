#!/usr/bin/env python3
"""人物关系抽取评测（M4 质量闭环）：共现基线 vs LLM 抽取的 precision/recall。

用法：
    python scripts/eval_graph.py                     # 只跑共现基线，零成本、零调用
    python scripts/eval_graph.py --model glm:glm-4-flash   # 追加 LLM 抽取对比（真实调用）

标注集在 tests/graph_eval_set.json：16 条人工标注，全部取自仓库原创语料
tests/ci_corpus/ 两篇短篇，刻意混入两类困难样本——
- 「同段高频共现但关系不成立」（如 修表客户被当成师徒）→ 考验精确率，
  这是共现基线的结构性假边来源；
- 「真关系但全文没有触发词」（如 教沙漠规矩却没有'师父'二字）→ 考验召回率，
  是关键词门控共现法原理上就够不到的部分。

两个方法共用同一条候选生成链路（只看含关系词的片段），差别只在判断环节：
基线用"出现过即算有关系"，LLM 方法要求输出 explicit 且置信度达标的判断——
与线上 GRAPH_REQUIRE_EXPLICIT 的门槛语义一致。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# backend.* 的模型客户端按项目根导入；独立跑脚本时还要先加载 .env，
# 否则 ZHIPU_API_KEY 缺失会让 glm: 路径全部失败（ingest.py 踩过的坑）
sys.path.insert(0, str(ROOT))

from graph import (  # noqa: E402
    build_edge_records,
    chunks_with_relation,
    extract_relations_llm,
)

_EVAL_SET_PATH = ROOT / "tests" / "graph_eval_set.json"


def load_cases(path: Path = _EVAL_SET_PATH) -> tuple[list[dict], dict[str, list[str]]]:
    """读取标注集，返回 (cases, 每本书的人物名单)。_readme 等说明字段自动跳过。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["cases"], payload["cast"]


def load_corpus(novels: list[str]) -> dict:
    """用正式切分器加载评测语料，保证和线上建图看到的是同样的片段边界。"""
    from loader import load_novel_file

    return {
        novel: load_novel_file(ROOT / "tests" / "ci_corpus" / f"{novel}.txt")
        for novel in novels
    }


def normalize_pair(a: str, b: str) -> tuple[str, str]:
    """人物对无向化：按字典序排序，与建图侧的存储约定一致。"""
    return min(a, b), max(a, b)


def _grouped_cases(cases: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """按 (书, 关系类型) 分组——评测和建图一样按这个粒度组织候选片段。"""
    groups: dict[tuple[str, str], list[dict]] = {}
    for case in cases:
        groups.setdefault((case["novel"], case["relation"]), []).append(case)
    return groups


def predict_cooccurrence(
    cases: list[dict], corpus: dict, cast: dict[str, list[str]]
) -> dict[str, bool]:
    """共现基线：含关系词的片段里两个人物同时出现，就预测 positive。

    这正是 M4 之前线上建图的全部逻辑（build_edge_records 的纯共现路径），
    作为对照基线它应该表现出「精确率高、召回率低」的特征——语料越干净
    假边越少，但触发词没出现的真关系一条也抓不到。
    """
    predictions = {case["id"]: False for case in cases}
    for (novel, relation), group in _grouped_cases(cases).items():
        chunks = corpus.get(novel, [])
        matched = chunks_with_relation(chunks, relation)
        records = build_edge_records(matched, set(cast.get(novel, [])), relation)
        found = {normalize_pair(record["person_a"], record["person_b"]) for record in records}
        for case in group:
            predictions[case["id"]] = normalize_pair(case["a"], case["b"]) in found
    return predictions


def predict_llm(
    cases: list[dict],
    corpus: dict,
    cast: dict[str, list[str]],
    generate_fn,
    min_confidence: float,
    errors: list[str],
) -> dict[str, bool]:
    """LLM 抽取方法：模型判断 explicit 且置信度达标才算 positive。

    与在线查询的质量门槛（GRAPH_REQUIRE_EXPLICIT + GRAPH_MIN_CONFIDENCE）
    同一种语义：co_occurrence 边即使被抽出来也不计入正类，因为它们默认
    只进审核队列、不进问答结果。
    某组抽取彻底失败（返回 None）时该组全部记 negative 并收集原因——
    评测如实反映降级后的表现，而不是悄悄给 LLM 记满分或零分。
    """
    predictions = {case["id"]: False for case in cases}
    for (novel, relation), group in _grouped_cases(cases).items():
        chunks = corpus.get(novel, [])
        matched = chunks_with_relation(chunks, relation)
        records = extract_relations_llm(
            matched, set(cast.get(novel, [])), relation, generate_fn, errors=errors
        )
        if records is None:
            continue
        found = {
            normalize_pair(record["person_a"], record["person_b"])
            for record in records
            if record["evidence_type"] == "explicit" and record["confidence"] >= min_confidence
        }
        for case in group:
            predictions[case["id"]] = normalize_pair(case["a"], case["b"]) in found
    return predictions


def confusion_counts(cases: list[dict], predictions: dict[str, bool]) -> Counter:
    """统计混淆四格。人工标签是唯一真值，predictions 只有 True/False 两档。"""
    counts: Counter = Counter()
    for case in cases:
        predicted = predictions[case["id"]]
        human = case["label"] == "positive"
        if predicted and human:
            counts["tp"] += 1
        elif predicted and not human:
            counts["fp"] += 1
        elif not predicted and human:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return counts


def prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    """precision / recall / F1。分母为 0 时返回 None（表格里显示 —）。"""
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return precision, recall, f1


def show_method(name: str, cases: list[dict], predictions: dict[str, bool]) -> None:
    """打印单个方法的混淆四格、P/R/F1 和误判清单。"""
    counts = confusion_counts(cases, predictions)
    precision, recall, f1 = prf(counts["tp"], counts["fp"], counts["fn"])

    def fmt(value: float | None) -> str:
        return f"{value:.1%}" if value is not None else "—"

    print(f"\n{'-' * 64}\n方法：{name}")
    print(f"  TP={counts['tp']}  FP={counts['fp']}  FN={counts['fn']}  TN={counts['tn']}")
    print(f"  精确率={fmt(precision)}  召回率={fmt(recall)}  F1={fmt(f1)}")
    mistakes = [c for c in cases if predictions[c["id"]] != (c["label"] == "positive")]
    if mistakes:
        print(f"  误判清单（{len(mistakes)} 条）：")
        for case in mistakes:
            verdict = "多判为正" if predictions[case["id"]] else "漏判为负"
            print(
                f"    [{case['id']}] {case['a']}–{case['b']} "
                f"{case['relation']}：{verdict}（{case.get('note', '')}）"
            )
    else:
        print("  没有误判。（全对时也要警惕标注集太简单）")


def _build_generate_fn(model: str):
    """按模型名前缀路由到生成后端（与 eval_faithfulness_shadow.py 同一约定）。"""
    from backend.dotenv_lite import load_env

    load_env(ROOT / ".env")
    if model.startswith("glm:"):
        from backend import zhipu

        return lambda prompt: zhipu.generate_stream(prompt, model)
    if model.startswith("claude:"):
        from backend import claude_cli

        return lambda prompt: claude_cli.generate_stream(prompt, model)
    raise SystemExit(f"不支持的模型前缀：{model}（只支持 glm:/claude: 前缀）")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="可选：glm:/claude: 前缀的模型名，追加一轮 LLM 关系抽取对比。"
        "不传时只跑共现基线，零成本零调用",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="LLM 边进入正类的最低置信度（默认 0.7，与 GRAPH_MIN_CONFIDENCE 一致）",
    )
    args = parser.parse_args()

    cases, cast = load_cases()
    corpus = load_corpus(sorted({case["novel"] for case in cases}))

    print("=" * 64)
    print("【关系抽取评测】对比「共现基线」与「LLM 抽取」的 P/R/F1")
    print("=" * 64)
    print(f"标注：{len(cases)} 条，来自 {sorted(corpus)}")

    # ---- 共现基线永远先算：零成本，且是任何新方法必须打败的对照组 ----
    baseline = predict_cooccurrence(cases, corpus, cast)
    show_method("共现基线（出现即算有关系）", cases, baseline)

    if args.model:
        print(f"\n>>> 开始 LLM 抽取：{args.model}（真实调用，注意费用）")
        generate_fn = _build_generate_fn(args.model)
        errors: list[str] = []
        predictions = predict_llm(
            cases, corpus, cast, generate_fn, args.min_confidence, errors
        )
        show_method(
            f"LLM 抽取（{args.model}，explicit≥{args.min_confidence}）", cases, predictions
        )
        if errors:
            print(f"\n抽取异常（已按降级计入该组结果，共 {len(errors)} 条）：")
            for reason in errors[:5]:
                print(f"  - {reason[:160]}")
    else:
        print("\n未指定 --model：只跑共现基线，不调用任何 LLM。")

    print("\n说明：评测只用于观察方法差异；是否默认开启图检索仍由 GRAPH_ENABLED 决定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
