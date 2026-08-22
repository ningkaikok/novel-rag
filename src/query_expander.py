"""自适应查询扩展（路线图 M3.4）：低置信度时生成改写变体补充检索。

和 query_rewriter 的区别（路线图刻意要求两者分开）
--------------------------------------------------
query_rewriter 解决的是**多轮指代**：把「他后来怎么样了」补全成
「墨大夫后来怎么样了」——那是把问题修完整，正常问题直接跳过。

本模块解决的是**单轮检索失败**：问题本身是完整的，但主链路检索出来的结果
置信度很低（信号见 ``confidence.py``）。这时换 2~3 种措辞再各查一次，
相当于从不同角度再摸几遍口袋——原措辞召回不了的东西，换个词可能就召回了。

为什么默认关闭、且最多补救一次
------------------------------
1. **成本与延迟**：每个变体都要完整跑一遍混合检索 + 重排，3 个变体就是
   3 倍检索开销，再加一次 LLM 调用。对所有问题默认扩展纯属浪费。
2. **语义漂移**：改写不总是中性的——小模型可能把「庄主的病」改成
   「庄主的身世」，越改离原问题越远。变体只做补充，原始问题的结果
   永远参与最终重排，坏变体最多浪费一次检索，不会污染原有结果。
3. **绝不循环**：「补救后还是低置信」就认输，交给现有的无证据拒答逻辑。
   在线系统没有真值可依赖，循环补救只会无限烧钱还掩盖问题。

实现方式刻意复用 query_rewriter 的模式：一次 LLM 调用、便宜的小模型、
失败降级为空列表（调用方看到没有变体就不补救）、解析时对模型输出的
各种花式格式（编号、引号、前缀客套话）做清洗。
"""
import re

# 和 query_rewriter 共用同一套「实质相同」判定：去掉标点和虚词后比较。
# 改写变体如果只是标点不同，拿去检索结果必然一样，纯属浪费一次全链路。
from query_rewriter import _normalize

_PROMPT = """下面这个问题在小说原文库里检索效果不佳，召回的内容和问题对不上。
请生成 {max_variants} 个改写变体，用于换一种说法再次检索同一个信息需求。

要求：
- 每个变体独立一行，行首不要编号、不要任何符号前缀
- 只改变表达方式（同义词、语序、详略），不得改变问题在问什么
- 不要回答问题，也不要解释，只输出改写后的问句

原问题：{question}"""


def _clean_variant(line: str) -> str:
    """清掉模型输出里常见的包装：行首编号/圆点、成对引号、首尾空白。"""
    text = line.strip()
    # "1. "/"2、"/*/- 等前缀——prompt 明令禁止但模型偶尔不听话
    text = re.sub(r"^(?:\d{1,2}[.、)）]|[-*•·])\s*", "", text)
    # 引号包装："..." 「...」 “...” '...'
    for left, right in (('"', '"'), ("「", "」"), ("\u201c", "\u201d"), ("'", "'")):
        if len(text) >= 2 and text.startswith(left) and text.endswith(right):
            text = text[1:-1].strip()
    return text


def expand_query_variants(
    question: str,
    generate_fn,
    max_variants: int = 3,
    errors: list | None = None,
) -> list[str]:
    """为低置信度问题生成至多 max_variants 个检索改写变体；失败返回空列表。

    **失败必须降级为空列表**而不是抛异常：扩展只是锦上添花的补救路径，
    模型限流/超时不应让整个问答挂掉——调用方拿到空列表就直接跳过补救，
    用原始检索结果继续走（行为退化为未开启扩展，完全安全）。
    失败原因会追加进 errors（如果调用方给了），避免静默失效无从排查
    （Contextual Retrieval 那边踩过的坑）。
    """
    try:
        raw = "".join(generate_fn(_PROMPT.format(max_variants=max_variants, question=question))).strip()
    except Exception as exc:
        if errors is not None:
            errors.append(f"{type(exc).__name__}: {exc}")
        return []

    variants: list[str] = []
    seen = {_normalize(question)}
    for line in raw.splitlines():
        variant = _clean_variant(line)
        # 太短的行基本是模型输出的残渣（"好的""如下"）；太长的行说明模型
        # 把解释性文字混进来了——一个坏变体比少一个变体更糟。
        if len(variant) < 4 or len(variant) > len(question) * 3 + 50:
            continue
        key = _normalize(variant)
        if not key or key in seen:
            continue  # 和原问题实质相同、或彼此重复的变体没有检索价值
        seen.add(key)
        variants.append(variant)
        if len(variants) >= max_variants:
            break
    return variants
