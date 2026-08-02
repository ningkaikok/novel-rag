#!/usr/bin/env python3
"""CI 跑完后把构建结果推送到飞书群自定义机器人。

只依赖标准库（urllib/hmac/hashlib/base64），不需要在这个 job 里额外装依赖。

需要的环境变量：
  FEISHU_WEBHOOK_URL     必需。飞书群「自定义机器人」的 Webhook 地址。
                         没设置时直接跳过（exit 0），不会让 CI 失败——
                         这样在还没配置飞书机器人之前，合并这个 workflow 不会破坏 CI。
  FEISHU_WEBHOOK_SECRET  可选。机器人开启了「签名校验」时提供的签名密钥。
  FRONTEND_RESULT / E2E_RESULT / BACKEND_RESULT
                         对应 workflow 里三个 job 的 needs.*.result（success/failure/...）。

GitHub 自动注入的 GITHUB_REPOSITORY / GITHUB_REF_NAME / GITHUB_SHA / GITHUB_ACTOR /
GITHUB_EVENT_NAME / GITHUB_SERVER_URL / GITHUB_RUN_ID 直接从环境读取，不需要额外传参。
"""
import base64
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.request


def sign(timestamp: str, secret: str) -> str:
    """飞书自定义机器人签名算法：把 "{timestamp}\\n{secret}" 当作 HMAC-SHA256 的 key，
    对空字节串求 MAC，再 base64 编码。这是飞书官方文档给出的固定算法，不能改动。
    """
    string_to_sign = f"{timestamp}\n{secret}"
    hmac_code = hmac.new(
        string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
    ).digest()
    return base64.b64encode(hmac_code).decode("utf-8")


def build_message() -> str:
    results = {
        "前端类型检查": os.environ.get("FRONTEND_RESULT", "unknown"),
        "前端 e2e 测试": os.environ.get("E2E_RESULT", "unknown"),
        "后端导入检查": os.environ.get("BACKEND_RESULT", "unknown"),
    }
    ok = all(v == "success" for v in results.values())
    overall = "✅ 全部通过" if ok else "❌ 有检查未通过"

    status_icon = {"success": "✅", "failure": "❌", "cancelled": "⏹️", "skipped": "⏭️"}
    lines = [f"{status_icon.get(v, '❓')} {name}：{v}" for name, v in results.items()]

    repo = os.environ.get("GITHUB_REPOSITORY", "")
    ref = os.environ.get("GITHUB_REF_NAME", "")
    sha = os.environ.get("GITHUB_SHA", "")[:7]
    actor = os.environ.get("GITHUB_ACTOR", "")
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    run_url = (
        f"{os.environ.get('GITHUB_SERVER_URL', '')}/{repo}/actions/runs/"
        f"{os.environ.get('GITHUB_RUN_ID', '')}"
    )

    text = (
        f"【{repo}】CI {overall}\n"
        f"分支/PR：{ref}　提交：{sha}　触发人：{actor}（{event}）\n"
        + "\n".join(lines)
        + f"\n详情：{run_url}"
    )
    return text


def send(webhook_url: str, secret: str, text: str) -> None:
    payload: dict = {"msg_type": "text", "content": {"text": text}}
    if secret:
        timestamp = str(int(time.time()))
        payload["timestamp"] = timestamp
        payload["sign"] = sign(timestamp, secret)

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp_body = resp.read().decode("utf-8")
        result = json.loads(resp_body)
        # 飞书 webhook 返回 200 但可能在 body 里报业务错误（code != 0），例如签名不对、
        # 机器人被移出群等——必须检查 code，否则 CI 会误以为通知发送成功。
        if result.get("code", 0) != 0:
            print(f"飞书返回错误：{resp_body}", file=sys.stderr)
            sys.exit(1)
        print(f"飞书通知发送成功：{resp_body}")


def main() -> None:
    webhook_url = os.environ.get("FEISHU_WEBHOOK_URL", "").strip()
    if not webhook_url:
        print("未配置 FEISHU_WEBHOOK_URL，跳过飞书通知（不影响 CI 结果）。")
        return

    secret = os.environ.get("FEISHU_WEBHOOK_SECRET", "").strip()
    text = build_message()
    try:
        send(webhook_url, secret, text)
    except urllib.error.URLError as e:
        # 通知失败不应该让整条 CI 变红——构建本身的结果已经由前面几个 job 决定了。
        print(f"发送飞书通知失败（不影响 CI 结果）：{e}", file=sys.stderr)


if __name__ == "__main__":
    main()
