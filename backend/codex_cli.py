"""通过本地已登录的 OpenAI Codex CLI（`codex`）复用用户自己的 ChatGPT 订阅生成回答。

和 ``claude_cli.py`` 是同一种模式：不需要单独配置 API Key，直接复用 `codex`
命令行已经登录的 OAuth 会话；两条相同的告知义务同样适用——这条路径会把检索到的
原文片段和问题发送到 OpenAI 的服务器，调用计入用户自己 ChatGPT 订阅的用量/额度。

**和 claude_cli.py 的关键差异**（Codex CLI 未装在本仓库开发机上，以下依据
`codex exec --json` 的公开文档整理，未在真实登录环境验证过，模型别名等信息
如与用户实际可用的模型不一致，用 ``CODEX_MODEL_ALIASES`` 环境变量覆盖）：

1. `codex exec --json` 不是逐 token 增量输出，而是任务完成后一次性给出完整
   ``item.completed`` / ``agent_message`` 事件——因此这里每次调用只会 yield
   一次完整文本，不是多次增量片段。前端的打字机效果由 useChatStream 自己
   按字节匀速吐出，不依赖后端真的是逐字流式，所以体验上没有区别。
2. `codex exec` 默认沙箱只读、审批策略 never（不会因为等待交互式审批而挂起），
   这里仍显式传参，避免行为随 CLI 版本升级的默认值变化而漂移。
3. 没有独立的 --system-prompt 参数，改成把角色设定拼进 prompt 正文最前面。
"""

import json
import os
import shutil
import subprocess
from collections.abc import Iterator

MODEL_PREFIX = "codex:"
# Codex CLI 当前公开文档列出的模型别名；如果用户账号下的实际可用模型不同，
# 用 CODEX_MODEL_ALIASES（逗号分隔）覆盖，不需要改代码。
CODEX_MODEL_ALIASES = [
    alias.strip()
    for alias in os.environ.get("CODEX_MODEL_ALIASES", "gpt-5-codex,gpt-5,o3").split(",")
    if alias.strip()
]

# 没有专门的系统提示参数，直接拼进 prompt 最前面，效果等价于 claude_cli 那份。
_SYSTEM_PROMPT = "你是一个只做问答的助手，不使用任何工具，只依据用户消息里提供的内容作答。"


def is_available() -> bool:
    return shutil.which("codex") is not None


def codex_model_options() -> list[str]:
    """返回可选的 codex:xxx 模型名列表；CLI 未安装时返回空列表。"""
    return (
        [f"{MODEL_PREFIX}{alias}" for alias in CODEX_MODEL_ALIASES] if is_available() else []
    )


def generate_stream(prompt: str, model_name: str) -> Iterator[str]:
    """model_name 形如 'codex:gpt-5-codex'，去掉前缀后作为 -m 的模型名传给 CLI。

    `codex exec --json` 一次任务只有一条 agent_message，所以这里通常只 yield
    一次；调用方（model_gateway/agent_lab）按 Iterator[str] 消费，多一次少一次
    不影响协议。
    """
    alias = model_name.removeprefix(MODEL_PREFIX)
    full_prompt = f"{_SYSTEM_PROMPT}\n\n{prompt}"
    cmd = [
        "codex",
        "exec",
        full_prompt,
        "--json",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "-c",
        "approval_policy=never",
        "-m",
        alias,
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
    )
    assert proc.stdout is not None and proc.stderr is not None
    # 同 claude_cli：失败原因（比如登录过期）经常在 stdout 的 JSON 事件里，
    # 不在 stderr——事先记下来，退出码非零时才有信息量的报错可用。
    cli_error_text: str | None = None
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_type = event.get("type")
            if event_type == "error":
                text = event.get("message") or event.get("text")
                if isinstance(text, str) and text:
                    cli_error_text = text
                continue
            if event_type != "item.completed":
                continue
            item = event.get("item") or {}
            item_type = item.get("type")
            if item_type == "error":
                text = item.get("text") or item.get("message")
                if isinstance(text, str) and text:
                    cli_error_text = text
                continue
            # 跳过 reasoning / command_execution / file_change 等中间步骤，
            # 只把最终答案文本交给上层。
            if item_type == "agent_message":
                text = item.get("text", "")
                if text:
                    yield text
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            # 还在跑——大概率是调用方 token_iter.close() 触发了 GeneratorExit
            # （用户点了「停止」），处理方式与 claude_cli 完全一致。
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        returncode = proc.returncode
        if returncode is not None and returncode > 0:
            err = proc.stderr.read().strip() or cli_error_text or "未知错误（stderr 为空）"
            raise RuntimeError(f"codex CLI 调用失败（exit {returncode}）：{err}")
