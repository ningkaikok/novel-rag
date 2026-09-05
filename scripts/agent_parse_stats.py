"""统计 Agent Lab 动作解析的真实失败率（路线图 M3.2.1 的前置埋点）。

为什么需要这个脚本
------------------
M3.2.1 想把 JSON 动作协议换成首行标签协议，理由是「JSON 收完才能解析、正则兜底
是猜、解析失败没有回路」。三条在道理上都成立，但**在自家模型上到底多久出一次，
从来没测过**。路线图对此写得很明白：先用真实失败率决定排期，不要凭"这个设计
更好"就抢跑。

这个脚本读 `chat_turns.agent_steps` 里已经落库的 `parse_mode`，把它聚合成一张表：

    strict          裸 JSON 一次成功——理想情况
    fenced          剥掉 ```json 围栏后成功——无害，但说明模型没遵守"只输出 JSON"
    regex           正则抓第一个 {...} 兜底成功——**最值得看的一档**
    failed:<类别>   彻底失败，走了降级

`regex` 比 `failed` 更值得警惕：它意味着解析成功了、但成功得很可疑。正则抓的是
第一个花括号，模型在 JSON 前后多写一句带花括号的话就可能抓错对象，而且抓错了
不会报错，会安安静静地执行一个错误的动作。`failed` 至少还会在界面上写明降级原因。

用法
----
    uv run python scripts/agent_parse_stats.py             # 全部历史
    uv run python scripts/agent_parse_stats.py --days 7    # 最近 7 天
    uv run python scripts/agent_parse_stats.py --model     # 按当轮模型分组

判读建议（路线图 M3.2.1 的排期依据）：`regex + failed` 合计占比低于 ~2% 说明
当前 JSON 协议在自家模型上够用，M3.2.1 可以继续排在后面；超过 ~10% 则说明步数
预算正在被解析问题稳定吃掉，值得提前。中间地带按绝对次数和用户可见影响判断。
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from postgres import connect  # noqa: E402


def collect(days: int | None = None, by_model: bool = False) -> tuple[Counter, Counter]:
    """返回 (parse_mode 计数, 按模型分组的 "模型|parse_mode" 计数)。"""
    where = "WHERE agent_steps IS NOT NULL"
    params: tuple = ()
    if days:
        where += " AND created_at >= now() - make_interval(days => %s)"
        params = (days,)
    with connect() as conn:
        rows = conn.execute(
            f"SELECT agent_steps, run_config FROM chat_turns {where}", params
        ).fetchall()

    modes: Counter = Counter()
    per_model: Counter = Counter()
    for row in rows:
        model = (row.get("run_config") or {}).get("generate_model") or "（未记录）"
        for step in row["agent_steps"] or []:
            mode = step.get("parse_mode")
            if not mode:
                # 规划器没参与的步骤（强制收尾、目录门禁）不计入——把它们算进来
                # 会把失败率冲淡，得出"看起来很健康"的假结论。
                continue
            modes[mode] += 1
            if by_model:
                per_model[f"{model}|{mode}"] += 1
    return modes, per_model


def _report(modes: Counter) -> str:
    total = sum(modes.values())
    if not total:
        return (
            "没有可统计的步骤。可能是：还没有人用过 Agent Lab，或者这些对话是\n"
            "本次埋点上线之前落库的（旧记录没有 parse_mode 字段）。"
        )
    lines = [f"共 {total} 个由规划器产出的步骤", ""]
    lines.append(f"{'parse_mode':<24}{'次数':>8}{'占比':>10}")
    for mode, count in modes.most_common():
        lines.append(f"{mode:<24}{count:>8}{count / total:>9.1%}")
    suspicious = sum(c for m, c in modes.items() if m == "regex" or m.startswith("failed"))
    lines += [
        "",
        f"可疑 + 失败合计：{suspicious} / {total} = {suspicious / total:.1%}",
        "（regex 属于「成功了但成功得可疑」：抓错对象不会报错，会静默执行错误动作）",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 Agent Lab 动作解析的真实失败率")
    parser.add_argument("--days", type=int, help="只统计最近 N 天")
    parser.add_argument("--model", action="store_true", help="额外按当轮生成模型分组")
    args = parser.parse_args()

    modes, per_model = collect(days=args.days, by_model=args.model)
    print(_report(modes))
    if args.model and per_model:
        print("\n按模型分组：")
        for key, count in sorted(per_model.items()):
            model, mode = key.split("|", 1)
            print(f"  {model:<28}{mode:<24}{count:>6}")


if __name__ == "__main__":
    main()
