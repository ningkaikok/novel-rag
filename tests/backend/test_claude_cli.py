"""claude_cli.generate_stream 的中断/终止逻辑单元测试。

不依赖真实 claude CLI（不需要本机装了 CLI、也不需要真实 OAuth 登录）——
用一个假的 Popen 对象模拟四种场景：正常结束、被中断（GeneratorExit）、
terminate 无效需要 kill 兜底、CLI 自己失败退出。
"""
import io
import json
import subprocess

import pytest

from backend import claude_cli


class FakeProc:
    """模拟 subprocess.Popen 的最小接口：stdout/stderr/poll/terminate/kill/wait/returncode。"""

    def __init__(self, lines, terminate_effective=True):
        self.stdout = io.StringIO("\n".join(lines) + "\n" if lines else "")
        self.stderr = io.StringIO("")
        self.returncode = None
        self.terminate_called = False
        self.kill_called = False
        self._terminate_effective = terminate_effective
        self._wait_calls = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_called = True
        if self._terminate_effective:
            self.returncode = -15  # 模拟收到 SIGTERM 后退出

    def kill(self):
        self.kill_called = True
        self.returncode = -9

    def wait(self, timeout=None):
        self._wait_calls += 1
        # terminate 后第一次 wait 如果还没生效（模拟进程没反应），抛超时，
        # 逼调用方走 kill() 兜底分支
        if not self._terminate_effective and self._wait_calls == 1:
            raise subprocess.TimeoutExpired(cmd="claude", timeout=timeout)
        return self.returncode


def _sse_line(text: str) -> str:
    return json.dumps(
        {
            "type": "stream_event",
            "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}},
        }
    )


def test_normal_completion_no_terminate(monkeypatch):
    """正常读完所有输出：不应该调用 terminate/kill，也不该抛异常。"""
    lines = [_sse_line("你"), _sse_line("好")]
    proc = FakeProc(lines)
    proc.returncode = 0  # 模拟 for 循环读完 stdout 时进程已经正常退出

    monkeypatch.setattr(claude_cli.subprocess, "Popen", lambda *a, **k: proc)

    chunks = list(claude_cli.generate_stream("你好吗", "claude:sonnet"))

    assert chunks == ["你", "好"]
    assert proc.terminate_called is False
    assert proc.kill_called is False


def test_interrupted_calls_terminate_and_does_not_raise(monkeypatch):
    """消费者中途 close() 生成器（对应用户点「停止」）：必须调用 terminate()，
    且这是预期中的中断，不应该抛异常掩盖 close() 本身。
    """
    lines = [_sse_line("正"), _sse_line("在"), _sse_line("生成")]
    proc = FakeProc(lines)
    # 中断时进程还没退出（poll() 应为 None），直到 terminate() 生效才变成 -15
    monkeypatch.setattr(claude_cli.subprocess, "Popen", lambda *a, **k: proc)

    gen = claude_cli.generate_stream("讲讲", "claude:sonnet")
    first = next(gen)
    assert first == "正"

    gen.close()  # 模拟 backend/main.py 里 token_iter.close()

    assert proc.terminate_called is True
    assert proc.kill_called is False  # terminate 生效了，不需要 kill 兜底


def test_interrupted_escalates_to_kill_when_terminate_ineffective(monkeypatch):
    """terminate() 之后进程还是没退出（wait 超时）：必须兜底调用 kill()。"""
    lines = [_sse_line("正"), _sse_line("在")]
    proc = FakeProc(lines, terminate_effective=False)
    monkeypatch.setattr(claude_cli.subprocess, "Popen", lambda *a, **k: proc)

    gen = claude_cli.generate_stream("讲讲", "claude:sonnet")
    next(gen)
    gen.close()

    assert proc.terminate_called is True
    assert proc.kill_called is True


def test_real_cli_failure_still_raises(monkeypatch):
    """CLI 自己失败退出（比如认证过期，正的非零返回码）：这才是真失败，要报错。"""
    proc = FakeProc([])
    proc.returncode = 1
    proc.stderr = io.StringIO("Error: not authenticated")
    monkeypatch.setattr(claude_cli.subprocess, "Popen", lambda *a, **k: proc)

    with pytest.raises(RuntimeError, match="claude CLI 调用失败"):
        list(claude_cli.generate_stream("讲讲", "claude:sonnet"))
