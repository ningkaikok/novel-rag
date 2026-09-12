#!/usr/bin/env python3
"""Legacy V1 → V2 dry-run 工具。

输入是由 V1 查询导出的 JSON snapshot；默认只生成并校验迁移计划，不连接 PostgreSQL。
示例：

    uv run python scripts/migrate_legacy_to_v2.py --input /tmp/legacy-snapshot.json --dry-run
    uv run python scripts/migrate_legacy_to_v2.py --input /tmp/legacy-snapshot.json \
        --schema-sql --embedding-dimension 384

本脚本没有数据写入或切换选项，避免误把 dry-run 变成生产迁移。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from v2_migration import LegacyNovelSnapshot, build_v2_migration_plan  # noqa: E402
from v2_schema import render_v2_schema_sql  # noqa: E402


def _load_snapshots(path: Path) -> list[LegacyNovelSnapshot]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    novels = payload.get("novels")
    if not isinstance(novels, list):
        raise ValueError("snapshot 顶层必须包含 novels 数组")
    snapshots: list[LegacyNovelSnapshot] = []
    for item in novels:
        terms = {
            int(str(chunk_id)): {str(term): int(tf) for term, tf in values.items()}
            for chunk_id, values in item.get("terms", {}).items()
        }
        snapshots.append(
            LegacyNovelSnapshot(
                novel=str(item["novel"]),
                source_hash=str(item["source_hash"]),
                pipeline_hash=str(item["pipeline_hash"]),
                chunks=tuple(item.get("chunks", [])),
                terms=terms,
            )
        )
    return snapshots


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 LegacyNovel → V2 迁移计划")
    parser.add_argument("--input", type=Path, required=True, help="V1 snapshot JSON 文件")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="显式声明只生成计划；脚本当前始终为 dry-run",
    )
    parser.add_argument("--schema-sql", action="store_true", help="额外打印 V2 DDL")
    parser.add_argument("--embedding-dimension", type=int, default=384)
    args = parser.parse_args()

    try:
        plan = build_v2_migration_plan(_load_snapshots(args.input))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"迁移计划校验失败：{exc}", file=sys.stderr)
        return 2

    print(json.dumps({"mode": "dry-run", **plan.summary()}, ensure_ascii=False, indent=2))
    if args.schema_sql:
        print("\n-- V2 schema DDL --")
        print(render_v2_schema_sql(args.embedding_dimension))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
