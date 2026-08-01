"""通过智谱开放平台（BigModel）的 GLM 系列模型生成回答。

API 是 OpenAI 兼容格式：POST /api/paas/v4/chat/completions，SSE 流式。

Key 只从环境变量 ZHIPU_API_KEY 读取，**绝不硬编码到代码里**：
    export ZHIPU_API_KEY=xxxxxxxx.yyyyyyyy
未设置时 is_available() 返回 False，界面上不会出现 GLM 选项。

同 claude_cli.py，这条路径也不是"完全本地"，需要在界面上如实告知用户：
1. 检索到的原文片段和问题会发送到智谱的服务器。
2. 调用计入用户自己的智谱账号额度。
"""
import json
import os
from collections.abc import Iterator

import requests

MODEL_PREFIX = "glm:"
API_URL = os.environ.get(
    "ZHIPU_API_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions"
)
# 常用几档，费用/能力依次递增；glm-4-flash 免费
GLM_MODELS = ["glm-4-flash", "glm-4.5-air", "glm-4.5", "glm-4.6"]

_SYSTEM_PROMPT = "你是一个只做问答的助手，只依据用户消息里提供的内容作答。"

_TIMEOUT = (10, 300)  # (连接, 读取)：流式回答可能持续较久


def _api_key() -> str | None:
    key = os.environ.get("ZHIPU_API_KEY", "").strip()
    return key or None


def is_available() -> bool:
    return _api_key() is not None


def model_options() -> list[str]:
    """返回可选的 glm:xxx 模型名列表；没配 key 时返回空列表。"""
    return [f"{MODEL_PREFIX}{m}" for m in GLM_MODELS] if is_available() else []


def generate_stream(prompt: str, model_name: str) -> Iterator[str]:
    """model_name 形如 'glm:glm-4.6'，去掉前缀后作为真实模型名。逐段 yield 文本增量。"""
    key = _api_key()
    if key is None:
        raise RuntimeError("未设置环境变量 ZHIPU_API_KEY，无法调用智谱 GLM")

    body = {
        "model": model_name.removeprefix(MODEL_PREFIX),
        "stream": True,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    with requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
        stream=True,
        timeout=_TIMEOUT,
    ) as resp:
        if resp.status_code != 200:
            # 不回显 key，只透出服务端的错误说明
            raise RuntimeError(
                f"智谱 GLM 调用失败（HTTP {resp.status_code}）：{resp.text[:300]}"
            )
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                # 只取最终答案；GLM 的推理过程在 reasoning_content 里，跳过
                text = delta.get("content")
                if text:
                    yield text
