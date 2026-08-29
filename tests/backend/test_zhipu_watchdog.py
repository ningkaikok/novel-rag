"""ZHIPU_STREAM_DEADLINE 流级看门狗的行为测试。

背景：本机代理隧道卡死时，SSE 流会被挂成「读超时被心跳不断重置」的僵尸
连接——requests 的 (10, 300) 超时形同虚设。看门狗用后台线程定时器到点强制
关闭响应对象，让阻塞中的 SSL read 立刻抛错（SIGALRM 实测打不断 macOS 上
_ssl 在 C 层的 poll 重试循环，所以必须走线程关响应这条路）。
"""

import threading
import time

import pytest

from backend import zhipu


class _NeverEndingStream:
    """模拟代理卡死：iter_lines 永远不产出也不结束，直到响应被外部 close。"""

    def __init__(self):
        self.status_code = 200
        self._closed = threading.Event()

    def close(self):
        self._closed.set()

    def raise_for_status(self):  # requests 响应接口的占位
        return None

    def iter_lines(self, decode_unicode=True):
        while not self._closed.is_set():
            time.sleep(0.05)
        raise ConnectionError("响应被看门狗关闭")


def test_stream_deadline_breaks_zombie_connection(monkeypatch):
    monkeypatch.setattr(zhipu, "_api_key", lambda: "test-key")
    resp = _NeverEndingStream()
    captured = {}

    def fake_post(url, **kwargs):
        captured["called"] = True

        class _Ctx:
            def __enter__(self):
                return resp

            def __exit__(self, *exc_info):
                resp.close()
                return False

        return _Ctx()

    monkeypatch.setattr(zhipu.requests, "post", fake_post)
    monkeypatch.setenv("ZHIPU_STREAM_DEADLINE", "0.5")

    start = time.monotonic()
    with pytest.raises((ConnectionError, OSError)):
        list(zhipu.generate_stream("glm:x", "glm-4-flash"))
    elapsed = time.monotonic() - start

    assert captured.get("called")
    # 关键断言：不是等满 iter_lines 的自然时长（这里等于永不返回），
    # 而是被 0.5s 的墙钟强制打断——留一点调度余量
    assert elapsed < 5.0, f"看门狗未生效，耗时 {elapsed:.1f}s"


def test_stream_deadline_disabled_by_default(monkeypatch):
    """不设环境变量时行为与旧版完全一致：没有定时器介入。"""
    monkeypatch.delenv("ZHIPU_STREAM_DEADLINE", raising=False)

    events: list[str] = [
        'data: {"choices": [{"delta": {"content": "你好"}}]}',
        "data: [DONE]",
    ]

    class _OneShotStream:
        status_code = 200

        def close(self):
            raise AssertionError("默认路径不应触发任何关闭")

        def iter_lines(self, decode_unicode=True):
            yield from events

    def fake_post(url, **kwargs):
        class _Ctx:
            def __enter__(self):
                return _OneShotStream()

            def __exit__(self, *exc_info):
                return False

        return _Ctx()

    monkeypatch.setattr(zhipu.requests, "post", fake_post)
    monkeypatch.setattr(zhipu, "_api_key", lambda: "test-key")

    text = "".join(zhipu.generate_stream("glm:x", "glm-4-flash"))
    assert text == "你好"
