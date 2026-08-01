"""极简 .env 加载器：把项目根目录 .env 里的 KEY=VALUE 注入 os.environ。

不引入额外依赖（不需要 python-dotenv）。

规则：
- 忽略空行和 # 注释行，容忍行首的 `export `
- 值两侧的引号会被去掉
- **已存在的环境变量优先**：临时 `export FOO=bar` 仍然能覆盖 .env 里的值
"""
import os
from pathlib import Path


def load_env(path: Path) -> list[str]:
    """加载 .env，返回本次真正注入的变量名列表（不含值，避免把密钥写进日志）。"""
    if not path.exists():
        return []

    loaded: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # 已在真实环境里设置过的不覆盖
        if not key or key in os.environ:
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded
