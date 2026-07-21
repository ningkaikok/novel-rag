"""通过本地已登录的 Claude Code CLI 调用用户自己的 Claude 订阅来生成回答。

不需要单独配置 ANTHROPIC_API_KEY —— 直接复用 `claude` 命令行已经登录的 OAuth 会话。

务必注意两点，并在界面上如实告知用户：
1. 这条路径会把检索到的原文片段和问题发送到 Anthropic 的服务器，不再是"完全本地"。
2. 调用计入用户自己 Claude 订阅的用量/额度。
"""
import json
import shutil
import subprocess
from collections.abc import Iterator

MODEL_PREFIX = "claude:"
# claude CLI --model 支持的别名；对应各档位，费用和速度依次递增
CLAUDE_MODEL_ALIASES = ["haiku", "sonnet", "opus"]

# 完全替换 CLI 默认的系统提示，避免带入"编码助手"人设或尝试调用工具
_SYSTEM_PROMPT = "你是一个只做问答的助手，不使用任何工具，只依据用户消息里提供的内容作答。"


def is_available() -> bool:
    return shutil.which("claude") is not None


def claude_model_options() -> list[str]:
    """返回可选的 claude:xxx 模型名列表；CLI 未安装时返回空列表。"""
    return [f"{MODEL_PREFIX}{alias}" for alias in CLAUDE_MODEL_ALIASES] if is_available() else []


def generate_stream(prompt: str, model_name: str) -> Iterator[str]:
    """model_name 形如 'claude:sonnet'，去掉前缀后作为 --model 的别名传给 CLI。逐段 yield 文本增量。"""
    alias = model_name.removeprefix(MODEL_PREFIX)
    cmd = [
        "claude",
        "--print",
        prompt,
        "--model",
        alias,
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--verbose",  # --print + stream-json 必须搭配此项
        "--system-prompt",
        _SYSTEM_PROMPT,
        "--allowedTools",
        "",
        "--disable-slash-commands",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
    )
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "stream_event":
                continue
            inner = event.get("event", {})
            if inner.get("type") != "content_block_delta":
                continue
            delta = inner.get("delta", {})
            # 跳过 thinking_delta（推理过程），只要最终答案的 text_delta
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                if text:
                    yield text
    finally:
        proc.stdout.close()
        returncode = proc.wait(timeout=10)
        if returncode != 0:
            err = proc.stderr.read()
            raise RuntimeError(f"claude CLI 调用失败（exit {returncode}）：{err.strip()}")
