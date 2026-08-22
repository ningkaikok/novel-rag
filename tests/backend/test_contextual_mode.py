"""CONTEXTUAL_MODE 三档决策（M3.4）的单元测试。

_contextual_decision 是 Contextual Retrieval auto 成本分级的核心：off 短路、
大书跳过、auto 档要求生成后端可用。这里用 monkeypatch 控制 config 导入值和
环境变量，不碰真实数据库和模型。
"""
import pytest

import ingest
from ingest import _contextual_decision


@pytest.fixture(autouse=True)
def backend_available(monkeypatch):
    """默认假设生成后端可用，各测试里按需覆盖。"""
    monkeypatch.setattr(ingest, "_generation_backend_available", lambda: True)


def test_off_mode_short_circuits_without_checks(monkeypatch):
    """off 档直接不做，连后端可用性检查都不该发生。"""
    monkeypatch.setattr(ingest, "CONTEXTUAL_MODE", "off")
    monkeypatch.setattr(
        ingest, "_generation_backend_available", lambda: (_ for _ in ()).throw(AssertionError("不应检查后端"))
    )
    build, reason = _contextual_decision("书", 10)
    assert build is False
    assert reason == ""


def test_auto_builds_small_books_when_backend_ready(monkeypatch):
    monkeypatch.setattr(ingest, "CONTEXTUAL_MODE", "auto")
    build, reason = _contextual_decision("小书", 100)
    assert build is True
    assert reason == ""


def test_auto_skips_large_books_by_chunk_gate(monkeypatch):
    monkeypatch.setattr(ingest, "CONTEXTUAL_MODE", "auto")
    monkeypatch.setattr(ingest, "CONTEXTUAL_MAX_CHUNKS_PER_BOOK", 2000)
    build, reason = _contextual_decision("大部头", 19501)
    assert build is False
    assert "19501" in reason and "2000" in reason


def test_on_mode_still_blocked_by_hard_chunk_gate(monkeypatch):
    """on 是强制生成，但体积上限是硬闸门——手滑 on 了也不许跑一整夜。"""
    monkeypatch.setattr(ingest, "CONTEXTUAL_MODE", "on")
    monkeypatch.setattr(ingest, "CONTEXTUAL_MAX_CHUNKS_PER_BOOK", 2000)
    build, reason = _contextual_decision("大部头", 3000)
    assert build is False
    assert "上限" in reason


def test_auto_requires_generation_backend(monkeypatch):
    monkeypatch.setattr(ingest, "CONTEXTUAL_MODE", "auto")
    monkeypatch.setattr(ingest, "_generation_backend_available", lambda: False)
    build, reason = _contextual_decision("小书", 10)
    assert build is False
    assert "生成后端" in reason


def test_on_mode_does_not_need_backend_probe(monkeypatch):
    """on 档由用户显式负责：即使探测函数坏了/没配 key 也照常构建（失败会走静默降级）。"""
    monkeypatch.setattr(ingest, "CONTEXTUAL_MODE", "on")
    monkeypatch.setattr(
        ingest,
        "_generation_backend_available",
        lambda: (_ for _ in ()).throw(AssertionError("on 模式不应探测")),
    )
    build, reason = _contextual_decision("小书", 10)
    assert build is True


def test_config_rejects_unknown_mode(monkeypatch):
    """非法模式值要在启动时立刻报错，而不是运行中静默当作 off。"""
    import subprocess
    import sys

    code = (
        "import os, sys; os.environ['CONTEXTUAL_MODE']='sometimes';"
        "sys.path.insert(0,'src'); import config"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode != 0
    assert "CONTEXTUAL_MODE" in result.stderr
