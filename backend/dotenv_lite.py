"""极简 .env 加载器：把项目根目录 .env 里的 KEY=VALUE 注入 os.environ。

不引入额外依赖（不需要 python-dotenv）。

为什么自己写而不用 python-dotenv
--------------------------------
本项目实际需要的 .env 功能只有"KEY=VALUE、去引号、忽略注释"这三条，二十行
就能覆盖；为此引入一个带完整解析器（多行值、变量插值、转义序列等）的第三方
依赖，收益几乎为零，却增加了安装面和供应链风险。这是个学习项目，依赖树保持
越小越好。如果将来真的需要插值/多行等高级特性，再换回 python-dotenv 也不迟
——本模块的对外行为（只注入、返回键名列表）和它是对齐的。

优先级设计（真实环境变量 > .env）
--------------------------------
已存在于 os.environ 的变量**不会被 .env 覆盖**。这样：
- 部署环境（shell 里 export 的密钥、CI 的 secrets）永远赢过文件里的旧值；
- 临时调试 `export ZHIPU_API_KEY=xxx` 不用改 .env 就能生效；
- 这也是 docker-compose / Heroku 等工具对 .env 类文件的通行语义。

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
        # .env 是可选的：没有它也能跑（全靠真实环境变量），不报错
        return []

    loaded: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # 容忍 `export KEY=...`：.env 常直接从 shell 脚本/终端历史里复制过来
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        # 简单去引号即可满足本项目（密钥、URL 都不含引号）；不支持转义引号、
        # 多行值这些 python-dotenv 的高级语法——见模块 docstring 的取舍说明。
        value = value.strip().strip('"').strip("'")
        # 已在真实环境里设置过的不覆盖（优先级设计见模块 docstring）
        if not key or key in os.environ:
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded
