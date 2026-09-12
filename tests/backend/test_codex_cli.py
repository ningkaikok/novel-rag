"""codex_cli.generate_stream 的中断/终止/错误提取逻辑单元测试。

和 test_claude_cli.py 同一套思路：不依赖真实 codex CLI（不需要本机装了 CLI、
也不需要真实 ChatGPT 登录），用一个假的 Popen 对象模拟四种场景：正常结束、
被中断（GeneratorExit）、terminate 无效需要 kill 兜底、CLI 自己失败退出。
"""

import io
import json
import subprocess

import pytest

from backend import codex_cli


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
            self.returncode = -15

    def kill(self):
        self.kill_called = True
        self.returncode = -9

    def wait(self, timeout=None):
        self._wait_calls += 1
        if not self._terminate_effective and self._wait_calls == 1:
            raise subprocess.TimeoutExpired(cmd="codex", timeout=timeout)
        return self.returncode


def _agent_message_line(text: str) -> str:
    return json.dumps(
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": text},
        }
    )


def test_normal_completion_no_terminate(monkeypatch):
    """正常读完所有输出：不应该调用 terminate/kill，也不该抛异常。"""
    lines = [_agent_message_line("你好")]
    proc = FakeProc(lines)
    proc.returncode = 0
    monkeypatch.setattr(codex_cli.subprocess, "Popen", lambda *a, **k: proc)

    chunks = list(codex_cli.generate_stream("你好吗", "codex:gpt-5-codex"))

    assert chunks == ["你好"]
    assert proc.terminate_called is False
    assert proc.kill_called is False


def test_skips_non_agent_message_items(monkeypatch):
    """reasoning / command_execution 等中间步骤不应该被当作正文 yield 出去。"""
    lines = [
        json.dumps(
            {"type": "item.completed", "item": {"type": "reasoning", "text": "思考中"}}
        ),
        json.dumps(
            {"type": "item.completed", "item": {"type": "command_execution", "text": "ls"}}
        ),
        _agent_message_line("最终答案"),
    ]
    proc = FakeProc(lines)
    proc.returncode = 0
    monkeypatch.setattr(codex_cli.subprocess, "Popen", lambda *a, **k: proc)

    assert list(codex_cli.generate_stream("问题", "codex:gpt-5-codex")) == ["最终答案"]


def test_interrupted_calls_terminate_and_does_not_raise(monkeypatch):
    """消费者中途 close() 生成器（对应用户点「停止」）：必须调用 terminate()。"""
    lines = [_agent_message_line("生成中")]
    proc = FakeProc(lines)
    monkeypatch.setattr(codex_cli.subprocess, "Popen", lambda *a, **k: proc)

    gen = codex_cli.generate_stream("讲讲", "codex:gpt-5-codex")
    next(gen)
    gen.close()

    assert proc.terminate_called is True
    assert proc.kill_called is False


def test_interrupted_escalates_to_kill_when_terminate_ineffective(monkeypatch):
    """terminate() 之后进程还是没退出（wait 超时）：必须兜底调用 kill()。"""
    lines = [_agent_message_line("生成中")]
    proc = FakeProc(lines, terminate_effective=False)
    monkeypatch.setattr(codex_cli.subprocess, "Popen", lambda *a, **k: proc)

    gen = codex_cli.generate_stream("讲讲", "codex:gpt-5-codex")
    next(gen)
    gen.close()

    assert proc.terminate_called is True
    assert proc.kill_called is True


def test_real_cli_failure_still_raises(monkeypatch):
    """CLI 自己失败退出（正的非零返回码）：这才是真失败，要报错。"""
    proc = FakeProc([])
    proc.returncode = 1
    proc.stderr = io.StringIO("Error: not authenticated")
    monkeypatch.setattr(codex_cli.subprocess, "Popen", lambda *a, **k: proc)

    with pytest.raises(RuntimeError, match="codex CLI 调用失败"):
        list(codex_cli.generate_stream("讲讲", "codex:gpt-5-codex"))


def test_cli_failure_with_empty_stderr_surfaces_json_error(monkeypatch):
    """登录过期这类失败：stderr 常常是空的，真正的原因在 JSON 的 error 事件里。"""
    error_event = json.dumps({"type": "error", "message": "Not logged in"})
    proc = FakeProc([error_event])
    proc.returncode = 1
    proc.stderr = io.StringIO("")
    monkeypatch.setattr(codex_cli.subprocess, "Popen", lambda *a, **k: proc)

    with pytest.raises(RuntimeError, match="Not logged in"):
        list(codex_cli.generate_stream("讲讲", "codex:gpt-5-codex"))
