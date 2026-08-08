"""多轮对话的查询改写：把带指代的追问补全成能独立检索的问题。

要解决什么问题
--------------
这个项目的 prompt 一直是**完全无状态**的：每次提问只把检索到的片段和当前问题
拼给模型，历史对话虽然存在 `chat_turns` 表里，但**从来没有参与过检索**。

于是这样的对话必然失败：

    用户：韩立的师父是谁？
    助手：墨大夫……
    用户：他后来怎么样了？        ← 拿「他后来怎么样了」去检索，什么也搜不到

「他」指的是谁，只有看历史才知道。而检索是**在生成之前**发生的，所以不能指望
生成模型去理解——必须先把问题补全，再拿补全后的问题去检索。

    "他后来怎么样了"  ──改写──→  "墨大夫后来怎么样了"  ──→ 检索

为什么不直接把历史拼进检索的查询里
------------------------------------
最省事的做法是把最近几轮对话原样拼在问题前面一起做 embedding。但那样会把大量
无关内容混进查询向量——上一轮的完整回答可能几百字，会把「他后来怎么样了」这句
真正的问题稀释掉，检索结果反而更差。

改写成一句独立完整的问题，既解决了指代，又不引入噪声。

成本控制
--------
改写要多调一次 LLM，会给**每一次追问**都加上延迟。所以：

1. **第一轮不改写**——没有历史，没有指代可解析
2. **看起来不需要改写的就跳过**——问题里没有指代词、且长度足够自成一句时，
   直接用原问题（见 needs_rewrite）
3. **用便宜的小模型**——这只是个句子改写任务，不需要推理能力
4. **失败降级为用原问题**——改写不该成为提问的阻塞点
"""

# 指代性表达：出现这些说明问题可能依赖上文
_REFERRING = (
    "他", "她", "它", "他们", "她们", "它们",
    "这", "那", "其", "此", "对方", "两人",
    "后来", "然后", "接着", "之后", "继续",
)

_PROMPT = """下面是一段对话历史，以及用户最新的问题。
请把最新的问题改写成一句**不依赖上文也能读懂**的完整问题。

要求：
- 把「他」「她」「那个」这类指代替换成历史里对应的具体名字
- 保持原问题的意图和信息量，不要增加历史里没有的内容
- 如果原问题本来就完整，原样输出即可
- 只输出改写后的问题，不要任何解释、前后缀或引号

对话历史：
{history}

最新的问题：{question}"""


# 比较两个问题是否"实质相同"时要忽略的字符：标点 + 不影响检索的虚词。
# 这些字符在 BM25 分词时本来就会被当停用词过滤掉，改不改都不影响检索结果。
_IGNORABLE = set("的了是在有和与吗呢啊吧？?！!。，,、 　\n")


def _normalize(text: str) -> str:
    return "".join(c for c in text if c not in _IGNORABLE)


def needs_rewrite(question: str, has_history: bool) -> bool:
    """判断这个问题值不值得花一次 LLM 调用去改写。

    两个条件都要满足：
    - **有历史**：第一轮没有上文，无从解析指代
    - **像是依赖上文**：含指代词，或者短到不像一句自足的问题

    长度阈值取 12 个字：比这更短的问题（"后来呢"、"还有吗"、"为什么"）
    基本都依赖上文；更长的问题通常已经把主语说清楚了。
    """
    if not has_history:
        return False
    text = question.strip()
    if len(text) <= 8:
        return True
    return any(word in text for word in _REFERRING)


def format_history(turns: list[dict], max_turns: int = 4, max_chars: int = 200) -> str:
    """把最近几轮对话压成一段紧凑的历史文本。

    **回答要截断**：助手的回答可能几百上千字，全部塞进改写提示词里既慢又容易
    喧宾夺主。改写只需要知道"上文提到了哪些人和事"，开头两三句就够了。
    """
    recent = turns[-max_turns * 2 :] if max_turns else turns
    lines = []
    for turn in recent:
        role = "用户" if turn.get("role") == "user" else "助手"
        content = (turn.get("content") or "").strip().replace("\n", " ")
        if len(content) > max_chars:
            content = content[:max_chars] + "…"
        if content:
            lines.append(f"{role}：{content}")
    return "\n".join(lines)


def rewrite_query(
    question: str, turns: list[dict], generate_fn, errors: list | None = None
) -> str:
    """把带指代的追问改写成独立完整的问题；不需要或失败时返回原问题。

    **失败必须降级为原问题**，并且（和 contextualizer 里同样的教训）
    **不能把失败原因也吞掉**——静默降级会让人完全无从排查为什么改写没生效。
    """
    if not needs_rewrite(question, has_history=bool(turns)):
        return question

    history = format_history(turns)
    if not history:
        return question

    try:
        rewritten = "".join(
            generate_fn(_PROMPT.format(history=history, question=question))
        ).strip()
    except Exception as exc:
        if errors is not None:
            errors.append(f"{type(exc).__name__}: {exc}")
        return question

    # 模型可能返回空串、或者啰嗦了一大段解释。两种情况都退回原问题——
    # 一个坏的改写比不改写更糟，它会把检索引到完全错误的方向。
    if not rewritten or len(rewritten) > len(question) * 5 + 50:
        return question

    # 改写结果和原问题实质相同时，当作没改。
    # 踩过的坑：问"雾隐山庄的庄主得了什么病"（本来就自足）被误判成需要改写，
    # 模型只把「的」去掉了，返回"雾隐山庄庄主得了什么病"——字符串不同，于是
    # 界面上显示"补全指代后按…检索"，可用户根本没写任何指代，纯属误导。
    # 判据是去掉标点和常见虚词后是否相等。
    if _normalize(rewritten) == _normalize(question):
        return question
    return rewritten
