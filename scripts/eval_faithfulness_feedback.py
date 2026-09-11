#!/usr/bin/env python3
"""用真实用户引用反馈评估影子 Judge，默认只输出聚合统计。

``helpful`` 作为 supported 的弱标签，``incorrect`` 作为 unsupported 的弱标签；
这不是金标准，尤其不能把一次反馈直接升级为自动拒答门槛。脚本的作用是观察
不同 Judge 方法/模型与真实使用信号的分歧，决定是否需要继续人工标注。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from postgres import load_citation_calibration_rows  # noqa: E402


def aggregate(rows: list[dict]) -> dict:
    groups: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in rows:
        human = "supported" if row["feedback"] == "helpful" else "unsupported"
        groups[(str(row["method"]), str(row["model"]))][(str(row["label"]), human)] += 1
    output = {}
    for (method, model), counts in sorted(groups.items()):
        total = sum(counts.values())
        agreement = sum(value for (predicted, human), value in counts.items() if predicted == human)
        output[f"{method}|{model}"] = {
            "samples": total,
            "agreement": agreement / total if total else None,
            "matrix": {f"{predicted}|{human}": value for (predicted, human), value in counts.items()},
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = aggregate(load_citation_calibration_rows(args.limit))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("method|model\tsamples\tagreement")
        for key, item in result.items():
            agreement = item["agreement"]
            rendered = "—" if agreement is None else f"{agreement:.1%}"
            print(f"{key}\t{item['samples']}\t{rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
