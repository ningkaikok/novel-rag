"""通过本地已登录的 OpenAI Codex CLI（`codex`）复用用户自己的 ChatGPT 订阅生成回答。

和 ``claude_cli.py`` 是同一种模式：不需要单独配置 API Key，直接复用 `codex`
命令行已经登录的 OAuth 会话；两条相同的告知义务同样适用——这条路径会把检索到的
原文片段和问题发送到 OpenAI 的服务器，调用计入用户自己 ChatGPT 订阅的用量/额度。

以下行为已用真实的 codex-cli 0.153.4（ChatGPT 账号登录）实测验证过，不是照文档猜的：

1. `codex exec --json` 不是逐 token 增量输出，而是任务完成后一次性给出完整的
   ``{"type":"item.completed","item":{"type":"agent_message","text":"..."}}``
   事件——因此这里每次调用只会 yield 一次完整文本，不是多次增量片段。前端的
   打字机效果由 useChatStream 自己按字节匀速吐出，不依赖后端真的是逐字流式，
   所以体验上没有区别。
2. **ChatGPT 账号登录时，``-m`` 传具体模型名（比如 gpt-5 / gpt-5-codex / o3）会被
   直接拒绝**：`"The 'xxx' model is not supported when using Codex with a ChatGPT
   account."`（这类模型名只对 API Key 登录开放）。不传 `-m` 才会用账号当前默认
   模型正常回答。所以这里默认只暴露一个 ``codex:default``（不传 -m），如果你是
   用 API Key 登录、账号支持指定模型，用 ``CODEX_MODEL_ALIASES``（逗号分隔）
   覆盖成你需要的具体模型名。
3. `codex exec` 默认沙箱只读、审批策略 never（不会因为等待交互式审批而挂起），
   这里仍显式传参，避免行为随 CLI 版本升级的默认值变化而漂移。
4. **`codex exec` 在检测到 stdin 是管道时会尝试把 stdin 内容当成额外输入读一遍**
   （stderr 会打一行 "Reading additional input from stdin..."）。后端作为子进程
   启动它时，父进程的 stdin 未必是已关闭的——不显式传 ``stdin=DEVNULL`` 会有
   实际卡死风险（这正是 claude_cli.py 那次排查过的"卡在思考"问题的同类隐患，
   这里趁实现时一次性堵上，不留给未来复现）。
5. 没有独立的 --system-prompt 参数，改成把角色设定拼进 prompt 正文最前面。
"""

import json
import os
import shutil
import subprocess
from collections.abc import Iterator

MODEL_PREFIX = "codex:"
# 表示"不传 -m，用账号当前默认模型"——ChatGPT 账号登录下这是唯一确认可用的选项
# （见上方模块说明第 2 点）。用 CODEX_MODEL_ALIASES 覆盖成具体模型名前，
# 请先确认你的登录方式支持指定模型，否则会在真正回答前就被 CLI 拒绝。
DEFAULT_ALIAS = "default"
CODEX_MODEL_ALIASES = [
    alias.strip()
    for alias in os.environ.get("CODEX_MODEL_ALIASES", DEFAULT_ALIAS).split(",")
    if alias.strip()
]

# CLI 可执行文件路径；不在 PATH 里（比如只在某个项目目录下装了 vendored 版本）时，
# 用这个环境变量指向具体路径，不用手动软链到 PATH 里。
CODEX_BIN = os.environ.get("CODEX_BIN", "").strip() or "codex"

# 没有专门的系统提示参数，直接拼进 prompt 最前面，效果等价于 claude_cli 那份。
_SYSTEM_PROMPT = "你是一个只做问答的助手，不使用任何工具，只依据用户消息里提供的内容作答。"


def is_available() -> bool:
    return shutil.which(CODEX_BIN) is not None


def codex_model_options() -> list[str]:
    """返回可选的 codex:xxx 模型名列表；CLI 未安装时返回空列表。"""
    return (
        [f"{MODEL_PREFIX}{alias}" for alias in CODEX_MODEL_ALIASES] if is_available() else []
    )


def generate_stream(prompt: str, model_name: str) -> Iterator[str]:
    """model_name 形如 'codex:default' 或 'codex:<具体模型名>'。

    别名等于 ``DEFAULT_ALIAS`` 时不传 ``-m``，让 CLI 用账号当前默认模型；
    否则原样透传给 ``-m``（前提是登录方式支持指定模型，见模块说明第 2 点）。

    `codex exec --json` 一次任务只有一条 agent_message，所以这里通常只 yield
    一次；调用方（model_gateway/agent_lab）按 Iterator[str] 消费，多一次少一次
    不影响协议。
    """
    alias = model_name.removeprefix(MODEL_PREFIX)
    full_prompt = f"{_SYSTEM_PROMPT}\n\n{prompt}"
    cmd = [
        CODEX_BIN,
        "exec",
        full_prompt,
        "--json",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "-c",
        "approval_policy=never",
    ]
    if alias != DEFAULT_ALIAS:
        cmd += ["-m", alias]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,  # 见模块说明第 4 点：不关掉会有真实卡死风险
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None and proc.stderr is not None
    # 同 claude_cli：失败原因（比如登录过期、模型名不被登录方式支持）经常在
    # stdout 的 JSON 事件里，不在 stderr——事先记下来，退出码非零时才有信息量
    # 的报错可用。
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
            if event_type in ("error", "turn.failed"):
                text = event.get("message") or (event.get("error") or {}).get("message")
                if isinstance(text, str) and text:
                    cli_error_text = text
                continue
            if event_type != "item.completed":
                continue
            item = event.get("item") or {}
            item_type = item.get("type")
            if item_type == "error":
                text = item.get("message") or item.get("text")
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
            # 和 claude_cli 相反的优先级，是实测出来的：codex exec 的 stderr 几乎
            # 总是非空——不管成功失败都会打一行 "Reading additional input from
            # stdin..." 的噪音（见模块说明第 4 点）——先读 stderr 只会把这行噪音
            # 当成"错误原因"，把真正有用的 JSON 失败原因盖掉。cli_error_text
            # 才是实测里真正可读的那部分（比如具体的模型名不被支持）。
            err = cli_error_text or proc.stderr.read().strip() or "未知错误（stderr 为空）"
            raise RuntimeError(f"codex CLI 调用失败（exit {returncode}）：{err}")
