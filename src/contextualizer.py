"""Contextual Retrieval：给"看不出在讲谁"的片段补一句上下文说明。

要解决什么问题
--------------
片段被切开后就失去了上下文。本项目里的真实例子（《凡人修仙传》chunk 2）：

    "二愣子睁大着双眼，直直望着茅草和烂泥糊成的黑屋顶…"

整段只有绰号「二愣子」，**没有「韩立」两个字**。用户问"韩立的绰号是什么"时，
这段话既匹配不上「韩立」，向量也不知道它在讲主角——正确答案就在眼前却召回不到。

做法：入库前用 LLM 给这类片段生成一句定位说明，拼在前面再做 embedding 和
BM25 索引；**但送给大模型回答时仍然用原片段**，不把生成的说明混进正文。

    索引的是：  "本段出自《凡人修仙传》开篇，讲主角韩立…" + 原片段
    回答用的是： 原片段

为什么这里必须"选择性 + 增量"，不能照搬原始方案
--------------------------------------------------
Anthropic 的原始做法是把**整篇文档**连同片段一起发给 LLM（靠 prompt caching
摊薄成本，官方数字是 $1.02/百万文档 token）。那套算法假设文档是 8K token 量级
的技术文档、财报、法律文件。

本项目的实际情况完全不同：

    《凡人修仙传》903 万字 ≈ 600 万 token
    GLM-4-Flash 上下文窗口   12.8 万 token      ← 差 47 倍

**整本小说根本塞不进上下文**，prompt caching 无从谈起。所以这里退而求其次，
用片段周围的窗口当"文档"——效果会比原方法差（模型看不到全书的人物关系），
但这是长篇小说场景下唯一可行的做法。

成本实测（GLM-4-Flash，本机）：单次调用约 4.4 秒。全库 32730 个片段串行要
40 小时。所以必须做两件事：

1. **选择性**：只给真正缺上下文的片段做（判据见 is_context_poor），
   实测《降龙》1278 个片段里只有 35% 需要处理。
2. **增量**：生成结果按「片段原文的哈希」存起来，重建索引时能复用。
   否则用户在界面上随手点一次「重新整理书架」就要重跑几小时。
"""
import collections
import hashlib
from concurrent.futures import ThreadPoolExecutor

import jieba.posseg as pseg

# 指代性词语：出现这些说明片段在讲"某个人"，但可能没说是谁
_PRONOUNS = ("他", "她", "它", "他们", "她们", "它们", "此人", "对方", "两人", "自己")

# Prompt 的设计思路（对照模块 docstring 里「二愣子」的失败案例）：
# - 给出书名和前后文窗口：长篇小说塞不进模型上下文，窗口是模型判断
#   “这段在讲谁”的唯一线索；
# - 明确要求点明绰号/代称对应的人物本名——这正是召回失败的根本原因，
#   不写清楚模型容易只复述片段内容而不做指代消解；
# - 限定一句话、50 字以内、禁止前后缀和引号：生成结果要拼在原片段前面
#   一起做 embedding 和 BM25 索引，太长会挤占原片段的输入预算，
#   客套话还会稀释 BM25 词频。
_PROMPT = """下面是一本小说《{novel}》里的一个片段，以及它前后的原文。
请用一句话（50 字以内）说明这个片段在讲什么、涉及哪些人物，用于给检索系统补充上下文。

要求：
- 如果片段里用的是绰号、代称或代词，请点明对应的人物本名
- 只输出这一句说明，不要任何前后缀、不要引号

前后文：
{window}

要说明的片段：
{chunk}"""


def extract_main_characters(texts: list[str], top_n: int = 20) -> set[str]:
    """从一本书的所有片段里统计出主要人物名。

    用 jieba 的词性标注取 nr（人名）标签再按出现次数排序。**这个名单一定会有
    噪声**——实测《降龙》的结果里混进了「闻言」「明白」「和尚」这类非人名。
    但噪声只会让判据**更保守**（把本来该处理的片段判成"不缺上下文"），
    不会造成错误处理，属于可以接受的误差。

    出现次数的门槛**随书的长度自适应**，不能写死。踩过的坑：门槛固定为 20 时，
    《雾隐山庄》（只有 3 个片段）里没有任何名字能出现 20 次，于是名单是空集合，
    `is_context_poor` 里的"不含任何人物名"恒为真——**所有片段都被判成缺上下文**。
    判据在小书上静默失效了。改成按片段数取比例（至少 2 次）之后，
    《雾隐山庄》能正常提取出「顾长风」「沈砚之」。
    """
    min_count = max(2, len(texts) // 50)
    counter: collections.Counter = collections.Counter()
    for text in texts:
        for word in pseg.cut(text):
            if word.flag == "nr" and len(word.word) >= 2:
                counter[word.word] += 1
    return {name for name, count in counter.most_common(top_n) if count >= min_count}


def is_context_poor(text: str, main_characters: set[str]) -> bool:
    """判断这个片段是不是"看不出在讲谁"。

    判据：**有指代性词语，但不含任何主要人物名**。
    这样的片段满篇"他""那汉子""老人"，读者和检索系统都无从判断主角是谁。

    实测《降龙》1278 个片段里 35.3% 命中这个判据，抽查确认都是真的缺上下文
    （"他们既然来过"、"那汉子来到一户人家门前"、"老人进了里屋"）。
    """
    has_pronoun = any(p in text for p in _PRONOUNS)
    has_name = any(name in text for name in main_characters)
    return has_pronoun and not has_name


def text_hash(text: str) -> str:
    """片段原文的哈希，用作生成结果的缓存键。

    用内容哈希而不是 (书名, chunk_id)：切分参数一变，同一个 chunk_id 对应的
    文本就变了，用位置做键会取到过期的上下文。用内容哈希则天然正确——
    文本没变就复用，变了就重新生成。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_window(chunks: list, index: int, neighbors: int = 2) -> str:
    """取片段前后各若干段作为"文档上下文"。

    这是对 Anthropic 原方案的降级——原方案给整篇文档，这里只能给一个窗口
    （原因见模块 docstring：长篇小说塞不进上下文窗口）。
    """
    start = max(0, index - neighbors)
    end = min(len(chunks), index + neighbors + 1)
    return "\n".join(c.text for c in chunks[start:end])


def generate_context(
    novel: str, chunk_text: str, window: str, generate_fn, errors: list | None = None
) -> str:
    """调 LLM 生成一句上下文说明。生成失败返回空串。

    **失败必须降级而不是阻断**：某个片段超时/限流不该让整个入库流程卡住，
    直接索引原文即可——那只是回到没有 Contextual Retrieval 的状态，不是故障。

    但**降级不能把失败原因也一起吞掉**。踩过的坑：第一版只 return ""，
    结果 451 个片段全部生成失败，日志里只有一句"451 条生成失败"，完全看不出
    为什么——真实原因是 ingest 独立运行时没加载 .env、拿不到 API key。
    静默降级 + 不报原因 = 让人调试半天。所以这里把异常收集到 errors 里，
    由调用方汇总展示。
    """
    prompt = _PROMPT.format(novel=novel, window=window[:2000], chunk=chunk_text)
    try:
        return "".join(generate_fn(prompt)).strip()
    except Exception as exc:
        if errors is not None:
            errors.append(f"{type(exc).__name__}: {exc}")
        return ""


def generate_contexts_parallel(
    tasks: list[tuple[str, str, str]],
    generate_fn,
    max_workers: int = 8,
    progress_every: int = 50,
) -> tuple[list[str], list[str]]:
    """并发生成多个片段的上下文说明。

    tasks 是 (书名, 片段原文, 窗口) 三元组列表。
    返回 (说明列表, 失败原因列表)——说明列表与输入等长，失败的位置是空串；
    失败原因单独返回，供调用方汇总展示（见 generate_context 里关于
    "降级不能吞掉原因"的说明）。

    并发是必需的：实测单次调用 4.4 秒，451 个片段串行要 33 分钟，
    8 路并发降到约 4 分钟。
    """
    results: list[str] = [""] * len(tasks)
    errors: list[str] = []
    done = 0

    def work(item):
        i, (novel, chunk_text, window) = item
        return i, generate_context(novel, chunk_text, window, generate_fn, errors)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, context in pool.map(work, enumerate(tasks)):
            results[i] = context
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"  已生成 {done}/{len(tasks)} 条上下文说明…")
    return results, errors
