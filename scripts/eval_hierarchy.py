#!/usr/bin/env python3
"""M3 层级检索的结构评测：检查跨书公平覆盖和章节多样性。

“主题是否概括得好”没有一个能靠关键词自动判定的唯一答案。这个脚本不伪造精度，
只自动验证层级检索必须满足的结构条件，并打印命中的章节标题供人工复核：

- 每本目标书都进入候选，而不是被另一本文本量更大的书挤掉；
- 每本书至少覆盖多个章节，而不是仍退化成单片段问答；
- 摘要命中已经映射回若干原文 SourceChunk。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from embedder import load_embedder  # noqa: E402
from rag import NovelRAG  # noqa: E402


def _short_title(novel: str) -> str:
    if "《" in novel and "》" in novel:
        return novel.split("《", 1)[1].split("》", 1)[0]
    return novel


def main() -> None:
    cases = json.loads(
        (ROOT / "tests" / "hierarchy_test_set.json").read_text(encoding="utf-8")
    )
    service = NovelRAG(embedder=load_embedder())
    failed = 0
    print(f"层级结构评测：{len(cases)} 条\n")
    for case in cases:
        named = service._named_novels(case["question"])
        sources, hits = service.hierarchy_retrieve(case["question"], named_novels=named)
        chapters = [hit for hit in hits if hit["level"] == "chapter"]
        counts = {
            expected: len(
                {hit["node_id"] for hit in chapters if expected in _short_title(hit["novel"])}
            )
            for expected in case["expect_novels"]
        }
        passed = bool(sources) and all(
            count >= case["min_chapters_per_novel"] for count in counts.values()
        )
        failed += not passed
        titles = "；".join(f"{_short_title(hit['novel'])}:{hit['title']}" for hit in chapters)
        print(
            f"{'✅' if passed else '❌'} {case['id']} {case['category']} "
            f"章节覆盖={counts} 原文候选={len(sources)}"
        )
        print(f"   {titles}")

    print(f"\n通过 {len(cases) - failed}/{len(cases)}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
