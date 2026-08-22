#!/usr/bin/env python3
"""检查小说索引输入质量并输出不含原文的 JSON 报告。

用法：
    python scripts/check_index_quality.py --novel data/novels/雾隐山庄.txt

脚本只做“切分 + tokenizer”检查，不写 PostgreSQL、不计算向量，适合在正式索引前
快速发现编码、空片段和 embedding 长度问题。完整索引时同一门禁会在 ``ingest.py``
中再次执行。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from embedder import load_embedder  # noqa: E402
from index_quality import make_quality_report  # noqa: E402
from ingest import _file_hash  # noqa: E402
from loader import load_novel_file, read_text_with_metadata  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="生成小说索引质量报告")
    parser.add_argument("--novel", type=Path, required=True, help="小说 txt 路径")
    parser.add_argument("--model", default=None, help="可选 embedding 模型名")
    args = parser.parse_args()
    path = args.novel.expanduser().resolve()
    if not path.is_file():
        parser.error(f"文件不存在：{path}")

    chunks = load_novel_file(path)
    _raw, source = read_text_with_metadata(path)
    source["byte_count"] = path.stat().st_size
    model = load_embedder(args.model) if args.model else load_embedder()
    texts = [chunk.text for chunk in chunks]
    report = make_quality_report(
        novel=path.stem,
        source_hash=_file_hash(path),
        source=source,
        chunks=chunks,
        model=model,
        embedding_inputs={"chunk": texts},
        lineage={"mode": "preflight", "model": args.model or "configured default"},
    )
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
