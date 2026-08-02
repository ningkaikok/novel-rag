#!/usr/bin/env python3
"""把 git-cliff 生成的内容合并进 CHANGELOG.md 的「未发布」段，只替换这一段。

为什么不用 git-cliff 的 --prepend：
--prepend 是往文件最前面硬插一整块，它不认识「哪里是手写说明、哪里该更新」。
本项目 CHANGELOG.md 开头有手写的格式说明和历史记录，用 --prepend 会：
  1. 重复插入 git-cliff 自己生成的「# 更新日志」标题（一次跑一个）；
  2. 把手写说明挤到下面，跑几次结构就乱了。
改成只定位并替换 `## [未发布]` 到下一个 `## [` 之间的内容，手写部分一字不动。

用法：merge_changelog.py 生成的片段.md CHANGELOG.md
"""
import sys
from pathlib import Path

MARKER = "## [未发布]"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("用法：merge_changelog.py GENERATED CHANGELOG")

    generated_path = Path(sys.argv[1])
    changelog_path = Path(sys.argv[2])

    if not generated_path.exists():
        raise SystemExit(f"找不到生成的片段文件：{generated_path}")

    generated = generated_path.read_text(encoding="utf-8").strip()
    current = changelog_path.read_text(encoding="utf-8")

    start = current.find(MARKER)
    if start < 0:
        raise SystemExit(f"{changelog_path} 里没有 {MARKER}，无法定位要替换的段落")

    # 「未发布」段的终点：下一个 ## [ 开头的版本标题；没有就到文件末尾
    next_release = current.find("\n## [", start + len(MARKER))
    if next_release < 0:
        next_release = len(current)

    # git-cliff 加了 --strip header，所以生成内容以 ## [未发布] 开头；
    # 没有任何符合规则的新提交时它会输出空内容，此时只保留一个空的段标题。
    replacement = generated if generated.startswith(MARKER) else MARKER

    before = current[:start]
    after = current[next_release:].lstrip("\n")

    parts = [before, replacement.rstrip(), "\n"]
    if after:
        # 段与段之间留一个空行，保持 Markdown 结构清晰
        parts.append("\n" + after)
    changelog_path.write_text("".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
