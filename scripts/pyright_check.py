#!/usr/bin/env python3
"""Pyright 增量门禁：只拦截「新增」的类型错误，存量进基线。

为什么需要这个脚本（P1 工程化）
------------------------------
7000+ 行存量的项目直接把 pyright 设成 CI 硬门槛会报出上百个历史问题，
一次修完既不现实也容易引入行为变更；但完全不设门槛又会让新代码继续欠债。
折中方案是工业界常用的「错误基线」：

- 首次运行 `--update-baseline` 把当前全部诊断记入 `.pyright-baseline.json`；
- 之后每次检查，只有 **基线之外的新诊断** 才算失败——存量可以慢慢还，
  新增零容忍；
- 修掉一个基线里的旧错后跑 `--update-baseline`，基线同步收缩（只减不增）。

用法
----
    python scripts/pyright_check.py                    # 增量检查（CI/本地通用）
    python scripts/pyright_check.py --update-baseline # 重新生成基线
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / ".pyright-baseline.json"


def run_pyright() -> list[dict]:
    proc = subprocess.run(
        ["pyright", "--outputjson", "src", "backend"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(proc.stdout[-2000:], proc.stderr[-500:], file=sys.stderr)
        raise SystemExit("pyright 输出解析失败") from None
    return data["generalDiagnostics"]


def fingerprint(diag: dict) -> dict:
    """诊断的稳定指纹：文件 + 规则 + 消息前缀（**不含行号**）。

    早期版本把行号也编进指纹，结果任何无关编辑导致代码上下平移，
    一批旧错误就被误判成「新增」而挡住合并——行号信息噪声大于信号。
    消息只取前 120 字符：同一处错误的措辞基本稳定，微小的表达变化
    不至于让整条基线失效；代价是同一文件同一规则的同文消息无法区分
    出现多次的情况——对「只拦新增」这个目的来说可接受。
    """
    return {
        "file": diag["file"].split("novel-rag/")[-1],
        "rule": diag.get("rule", ""),
        "message": diag["message"][:120],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pyright 增量门禁")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="把当前全部诊断写入基线文件（修复旧错后用来收缩基线）",
    )
    args = parser.parse_args()

    current = [fingerprint(d) for d in run_pyright()]

    if args.update_baseline:
        BASELINE.write_text(
            json.dumps(current, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"基线已更新：{len(current)} 条诊断 -> {BASELINE.relative_to(ROOT)}")
        return

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline_set = {json.dumps(item, sort_keys=True) for item in baseline}
    new_errors = [
        item for item in current if json.dumps(item, sort_keys=True) not in baseline_set
    ]

    if not new_errors:
        print(f"pyright 增量检查通过：{len(current)} 条诊断全部在基线内")
        return

    print(f"发现 {len(new_errors)} 条基线外的新类型错误：", file=sys.stderr)
    for item in new_errors[:30]:
        print(
            f"  {item['file']}:{item['line']} [{item['rule']}] {item['message'][:90]}",
            file=sys.stderr,
        )
    sys.exit(1)


if __name__ == "__main__":
    main()
