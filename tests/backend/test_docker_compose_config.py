"""Docker 部署配置的轻量回归检查（#3：重建容器后仍能找到上传的文档源文件）。

只读文本断言，不依赖 PyYAML、不启动 Docker、不重建任何容器。断言的是「路径三方
对齐」这一件事：

    仓库内目录  data/knowledge
    容器内目录  /app/data/knowledge            （Dockerfile 运行时阶段的 WORKDIR + 相对路径）
    应用默认值  ROOT_DIR/data/knowledge        （src/config.py 的 KNOWLEDGE_DIR）

任何一边改了另一边没跟着改，上传的 TXT/Markdown/PDF 源文件在容器重建后就随容器
消失，之后想重建这些文档的索引就没有原文可用了。
"""

import os
import re
from pathlib import Path

import pytest

from config import KNOWLEDGE_DIR, ROOT_DIR

COMPOSE_FILE = Path(ROOT_DIR) / "docker-compose.yml"
DOCKERFILE = Path(ROOT_DIR) / "Dockerfile"
KNOWLEDGE_MOUNT_SOURCE = "./data/knowledge"


def _mounts() -> dict[str, str]:
    """从 compose 文本里抠出 ``- 宿主机路径:容器路径`` 形式的 bind mount。"""
    compose = COMPOSE_FILE.read_text(encoding="utf-8")
    return {
        source: target
        for source, target in re.findall(r"-\s*(\./[^:\s]+):(/[^:\s]+)", compose)
    }


def _container_workdir() -> str:
    """取**运行时阶段**的 WORKDIR。

    Dockerfile 是多阶段构建，第一个 ``WORKDIR /app/frontend`` 属于前端构建阶段，
    真正决定运行镜像目录结构的是最后出现的那个（运行时阶段）。
    """
    workdirs = re.findall(
        r"^WORKDIR\s+(\S+)",
        DOCKERFILE.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert workdirs, "Dockerfile 必须有 WORKDIR，否则路径对齐全靠猜"
    return workdirs[-1]


def test_compose_persists_both_novels_and_knowledge_dirs():
    """小说目录和通用文档源目录都要挂在宿主机上。"""
    mounts = _mounts()

    assert mounts.get("./data/novels") == "/app/data/novels"
    assert mounts.get(KNOWLEDGE_MOUNT_SOURCE) == "/app/data/knowledge"
    assert "hf-cache:/cache/hf" in COMPOSE_FILE.read_text(encoding="utf-8"), (
        "模型缓存卷不能被顺手删掉（重建容器要重新下 1.2GB）"
    )


def test_knowledge_mount_target_matches_app_default_path():
    """容器目标目录必须等于「WORKDIR + KNOWLEDGE_DIR 的默认相对路径」。

    这条只在**默认配置**下成立：``KNOWLEDGE_DIR`` 是合法的环境变量覆盖点（生产
    多租户部署会指到 ``/tenant/knowledge`` 之类的位置）。本机若设了覆盖值，这条
    断言就没有可比对的对象，直接跳过而不是拿末尾目录名去硬凑。
    """
    if os.environ.get("KNOWLEDGE_DIR"):
        pytest.skip("本机覆盖了 KNOWLEDGE_DIR，无法据此校验 compose 的默认路径映射")

    relative = Path(KNOWLEDGE_DIR).relative_to(ROOT_DIR).as_posix()
    assert relative == "data/knowledge", "默认源目录必须是 data/knowledge"
    assert _mounts()[KNOWLEDGE_MOUNT_SOURCE] == f"{_container_workdir()}/{relative}"
