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
    # stdout/stderr 分开接管：stdout 走 JSON 事件流，stderr 只在 CLI 失败退出时
    # 读一次、拼进异常消息，平时不消费（避免和事件流混在一起解析）。
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
    )
    try:
        # 逐行读子进程 stdout：stream-json 模式下每行是一个独立 JSON 事件，
        # 行即消息边界，不需要自己攒缓冲区。非 stream_event / 非 text_delta 的
        # 事件（初始化、system、结果汇总等）全部跳过，只透出正文增量。
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
        if proc.poll() is None:
            # 还在跑——大概率是调用方 token_iter.close() 触发了 GeneratorExit
            # （用户点了「停止」）。只关我们这端的读管道不够：子进程如果当时
            # 不在写 stdout（比如正等着 Anthropic 的 API 响应），不会立刻收到
            # SIGPIPE 退出，可能继续跑到自然结束——那样用户点了停止，
            # 还在悄悄消耗 Claude 订阅额度。必须主动发信号终止。
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        returncode = proc.returncode
        # >0 才是 CLI 自己失败退出；0 是正常结束，负数是被信号杀死
        # （很可能就是上面我们自己发的 terminate/kill）——这两种都不该在这里
        # 抛异常：中断是预期行为，不是失败，抛出去只会掩盖调用方的 GeneratorExit，
        # 让 token_iter.close() 意外抛错。
        if returncode is not None and returncode > 0:
            err = proc.stderr.read()
            raise RuntimeError(f"claude CLI 调用失败（exit {returncode}）：{err.strip()}")
