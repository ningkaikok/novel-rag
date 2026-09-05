"""滚动会话摘要：接住那些掉出「最近几轮」窗口的对话（M3.6）。

要解决什么问题
--------------
``build_history_block`` 只把最近几轮原文带进 prompt，更早的整轮丢弃。这对指代
解析够用——"他"几乎总指上一两轮里的人——但长会话里另一类信息会跟着一起丢掉：

    第 1 轮：用户在读《雾隐山庄》
    第 3 轮：确认了顾长风是庄主、中的是蚀骨散
    ...
    第 15 轮：用户问"那他后来解了吗？"     ← 前面那些全都不在窗口里了

摘要就是用来接住这部分的：**不是替代原文证据，只是帮模型知道"我们在聊什么"**。

三条设计约束
------------
1. **只在超过阈值时更新**（路线图原话）。摘要要多调一次 LLM，如果每轮都更新，
   等于给每一次追问都加延迟。这里攒够 ``every`` 轮掉出窗口的对话才更新一次，
   短会话则一次都不会触发——那是绝大多数会话。
2. **失败不阻塞回答**。摘要生成失败就沿用上一版摘要，再不行就当作没有摘要，
   退回到"只有最近几轮原文"，也就是引入摘要之前的行为。
3. **摘要不是证据**。它由模型压缩而来，本身可能漂移；prompt 里它和逐字历史一样
   属于「对话背景」，明确禁止被引用（见 generation_mixin 的模板注释）。
"""

from config import HISTORY_SUMMARY_MAX_CHARS

# 摘要提示词。刻意要求"只输出摘要正文"：任何"好的，摘要如下："之类的前缀都会被
# 原样拼进下一轮的 prompt，而且会在滚动更新里被当成正文一路累积下去。
#
# 「不确定的就不要写」这条是本文件里最重要的一句约束。摘要是**在压缩里丢信息**，
# 而模型倾向于把丢掉的部分脑补圆满——一旦补出"顾长风已经解毒了"这种原文里没有的
# 结论，它会在之后每一轮都以背景的身份出现，比一次答错严重得多。
_PROMPT = """下面是一段小说问答对话。请把它压缩成一段简短的背景摘要，
供后续追问时理解"我们在聊什么"。

要求：
- 只写这些：在聊哪本书、出现过哪些人物、用户已经确认或关心的结论、当前话题
- **不确定的一律不写**。对话里没有明确说过的，绝对不要推断或补全
- 不要写"根据片段[1]"这类引用编号——摘要不是证据，编号在这里没有意义
- 不要复述完整情节，也不要评价
- 控制在 {max_chars} 字以内
- 只输出摘要正文，不要任何解释、标题或前后缀

{previous_block}对话内容：
{conversation}"""

_PREVIOUS_BLOCK = """已有的背景摘要（请在它的基础上更新，不要丢掉里面仍然有效的信息）：
{previous}

新增的"""


def turns_to_summarize(
    turns: list[dict], covered_through: int, window_turns: int
) -> list[dict]:
    """挑出「已经掉出逐字窗口、但还没进摘要」的那些轮次。

    ``covered_through`` 是上一版摘要覆盖到的最大 ``turn_index``（没有摘要时传 -1）。
    逐字窗口是最后 ``window_turns`` 轮，摘要负责的正是它前面、且尚未被覆盖的部分——
    两边合起来不重不漏。空内容的轮次先剔除，口径必须和 ``build_history_block``
    一致，否则同一轮可能既进摘要又进逐字历史。
    """
    usable = [t for t in turns if t.get("content")]
    outside = usable[:-window_turns] if window_turns > 0 else usable
    return [t for t in outside if int(t.get("turn_index", -1)) > covered_through]


def should_update(pending: list[dict], every: int) -> bool:
    """攒够 ``every`` 轮才值得为这次更新多花一次模型调用。"""
    return len(pending) >= max(1, every)


def format_conversation(turns: list[dict], per_turn_chars: int = 300) -> str:
    lines = []
    for turn in turns:
        role = "用户" if turn.get("role") == "user" else "助手"
        content = (turn.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        if len(content) > per_turn_chars:
            content = content[:per_turn_chars] + "…"
        lines.append(f"{role}：{content}")
    return "\n".join(lines)


def build_summary(
    previous: str | None,
    pending: list[dict],
    generate_fn,
    max_chars: int = HISTORY_SUMMARY_MAX_CHARS,
    errors: list | None = None,
) -> str | None:
    """滚动更新摘要：拿上一版摘要 + 新掉出窗口的几轮，产出新一版。

    返回 ``None`` 表示这次更新没成交（没有新内容、模型失败、结果不可用）。
    调用方应当**沿用上一版摘要**而不是清空——一次调用失败不该让长会话突然失忆。
    """
    conversation = format_conversation(pending)
    if not conversation:
        return None

    previous_block = (
        _PREVIOUS_BLOCK.format(previous=previous.strip())
        if previous and previous.strip()
        else ""
    )
    prompt = _PROMPT.format(
        max_chars=max_chars, previous_block=previous_block, conversation=conversation
    )

    try:
        summary = "".join(generate_fn(prompt)).strip()
    except Exception as exc:
        if errors is not None:
            errors.append(f"{type(exc).__name__}: {exc}")
        return None

    if not summary:
        if errors is not None:
            errors.append("模型返回空摘要")
        return None
    # 硬截断兜底：提示词里的字数要求是软约束，模型经常超。这里宁可截断也不能让
    # 摘要无限膨胀——它是每轮都要进 prompt 的固定开销，涨上去就再也下不来。
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "…（略）"
    return summary
