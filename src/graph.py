"""GraphRAG：把人物关系抽成一张图，回答"全书范围的关系聚合"问题。

要解决什么问题
--------------
有一类问题，**靠 top-k 片段拼凑本质上就答不好**：

    "韩立有哪些伴侣？"

答案分散在全书 19501 个片段的几十处，每处只提到一个人。无论怎么调检索，
top-3 或 top-10 都只能捞到其中一两处——**这不是检索质量问题，是方法不匹配**。
向量检索和 BM25 回答的是"哪几段最相关"，而这个问题问的是"把全书扫一遍、
汇总出一个列表"。

图检索换了个思路：**预先把关系抽出来存成结构化数据**，查询时直接做图遍历，
一次拿到完整列表，不受 top-k 限制。

关键设计：只从「含关系词的片段」里抽
--------------------------------------
最省钱的做法本来是用 jieba 的词性标注（nr = 人名）免费抽人名。**试过，不行**：

    jieba.posseg.cut("南宫婉点了点头")
      → [('南宫', 'ns'), ('婉', 'ag'), ...]        ← 复姓被切成地名
    jieba.posseg.cut("厉飞雨大笑")
      → [('厉', 'nr'), ('飞雨', 'n'), ...]         ← 三字名被切开

**复姓和三字名根本切不对**——不是标注错，是分词就错了，后面所有基于 nr 的
方案都建立在沙子上。实测《凡人修仙传》按 nr 频次取 top-40，一个真人名都没有，
全是「闻言」「白光」「修仙」这类词。

改用 LLM 抽人名，但**采样方式是关键**：

- ❌ 全书均匀采样 60 段 → 只覆盖 0.3% 的内容，南宫婉这类角色抽不到
- ✅ **只采「含关系词的片段」** → 关系本来就只存在于这些片段里

实测后者（《凡人修仙传》「伴侣」关系，88 个片段、11 次调用、57 秒）：

    韩立 12 次、南宫婉 5 次、紫灵 3 次、元瑶 2 次、董萱儿 1 次

又准又省——比全书采样准，比逐片段抽取省几个数量级。

实测结论：原理跑通了，但没达到可用质量（必须如实记录）
--------------------------------------------------------
查「韩立有哪些伴侣」的真实结果：

    陈巧倩 10、南宫婉 9、紫灵 7、雪虹 6、李化元 5、宋蒙 5、曲魂 5
                                  ↑师父    ↑男性  ↑魔头

三个明显的假边，而真正的核心伴侣南宫婉只排第二。端到端问答时模型选择了拒答
（"根据提供的片段无法确定"）——**好的一面**是"统计推断"这个标注起了作用、
没有产生"把师父说成伴侣"的幻觉；**坏的一面**是也没能给出正确答案。

根因：共现只知道「这两个名字出现在一段含"道侣"的文字里」，**分不清
"他们是道侣"还是"他们在讨论别人的道侣"**。

> **GraphRAG 的难点不在图结构**——存边、查邻居都很简单。
> **难点在关系抽取的质量**，而抽取质量和成本直接挂钩。
> 要真正可用得让 LLM 在抽取时判断关系的方向和真假，那就回到全库级成本
> （本项目实测约 40 小时）。规则法省下的钱，正好就是精度。

所以 GRAPH_ENABLED 默认关闭，这份代码的定位是**原理演示 + 天花板实测**。

局限（必须说清楚）
------------------
边是**共现推断**出来的，不是真正的关系抽取：两个人同时出现在一段提到「师父」
的话里，不代表他们是师徒。所以：

- 用**共现次数**做权重，查询时按权重排序、要求达到最小阈值，过滤偶然同框
- 结果明确标注为"根据共现统计推断"，不当作确定事实
- 图结果**和原文片段一起**给模型，让模型自己核对，而不是直接当答案输出
"""

import collections
import re

# 关系类型 → 触发词。含这些词的片段才会被拿去抽人名和建边。
#
# 词表是**手工的、面向中文网络小说的**，不追求通用性——GraphRAG 的关系抽取
# 本来就高度依赖领域。换个语料（比如企业文档）要换一套完全不同的词表，
# 这正是规则法相比全 LLM 抽取的主要短板。
RELATION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "伴侣": ("双修伴侣", "道侣", "结为夫妻", "结成好事"),
    "师徒": ("师父", "师尊", "拜师", "恩师"),
    "亲属": ("父亲", "母亲", "哥哥", "弟弟", "姐姐", "妹妹"),
    "敌对": ("仇人", "死敌", "宿敌"),
}

# 问题里出现这些词，就认为在问对应的关系（用于决定要不要走图检索）
_QUESTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "伴侣": ("伴侣", "道侣", "妻子", "老婆", "夫人", "情侣", "感情"),
    "师徒": ("师父", "师尊", "徒弟", "师承", "拜师"),
    "亲属": ("家人", "亲人", "父母", "兄弟姐妹", "家里"),
    "敌对": ("仇人", "敌人", "对手"),
}

_EXTRACT_PROMPT = """下面是小说里的几个片段。列出其中出现的**人物姓名**。
只要真实姓名（例如「韩立」「南宫婉」这样的具体名字），
不要「师兄」「老者」「白衣修士」这类泛称，也不要「白光」「金光」这类非人名。
只输出人名，用顿号分隔，不要任何解释。

{batch}"""


def detect_relation_question(question: str) -> str | None:
    """判断这个问题是不是在问某种人物关系；不是则返回 None。

    只有命中时才走图检索——图检索是**补充**而不是替代，普通问题走原来的
    多路召回就好，没必要多查一次图。
    """
    for relation, words in _QUESTION_PATTERNS.items():
        if any(w in question for w in words):
            return relation
    return None


def _parse_names(raw: str) -> list[str]:
    """从模型输出里解析出人名列表，并做基本清洗。

    **句号也要当分隔符**，不能只当作可剥离的尾字符。踩过的坑：模型有时会
    在名单后面加一句话（"韩立、南宫婉、紫灵。这些是主要人物"），如果只按顿号
    切分，「紫灵。这些是主要人物」会变成一个整体——strip 只能去掉首尾的标点，
    去不掉中间的句号，于是这个片段因为超长被整个丢弃，**连带把「紫灵」也丢了**。
    """
    names = []
    for piece in re.split(r"[、,，。；;\s]+", raw.strip()):
        name = piece.strip(".！!：:（）()「」《》")
        # 2~4 个纯汉字才当人名：中文人名基本都在这个范围，
        # 加长度限制能挡掉模型偶尔输出的解释性短语
        if 2 <= len(name) <= 4 and re.fullmatch(r"[一-龥]+", name):
            names.append(name)
    return names


def extract_characters_from_chunks(
    chunks: list, generate_fn, batch_size: int = 8, errors: list | None = None
) -> collections.Counter:
    """用 LLM 从给定片段里抽人名，返回 {人名: 被抽到的批次数}。

    批次数就是置信度：一个词在多个批次里都被认作人名，才更可能是真人名。
    单次出现的往往是模型偶然把泛称当成了名字。

    失败降级为跳过这一批（不阻断整个建图），但把原因收集起来——
    Contextual Retrieval 那边踩过"静默降级查不出原因"的坑。
    """
    names: collections.Counter = collections.Counter()
    # 成本靠双重截断封顶：每个片段只取前 400 字（人名和关系句通常出现在
    # 片段开头），每批再整体截到 4500 字。单批 LLM 输入长度因此有确定上界，
    # 不会因为个别超长片段把一次调用的 token 数顶爆。
    for i in range(0, len(chunks), batch_size):
        batch = "\n---\n".join(c.text[:400] for c in chunks[i : i + batch_size])
        try:
            out = "".join(generate_fn(_EXTRACT_PROMPT.format(batch=batch[:4500])))
        except Exception as exc:
            if errors is not None:
                errors.append(f"{type(exc).__name__}: {exc}")
            continue
        for name in set(_parse_names(out)):
            names[name] += 1
    return names


def chunks_with_relation(chunks: list, relation: str) -> list:
    """挑出含某种关系触发词的片段。"""
    words = RELATION_KEYWORDS.get(relation, ())
    return [c for c in chunks if any(w in c.text for w in words)]


def build_edges(chunks: list, characters: set[str], relation: str) -> list[tuple]:
    """在给定片段里做人物共现统计，产出关系边。

    返回 (书名, 人物A, 人物B, 关系类型, 共现次数)，A/B 按字典序排好——
    关系是无向的，排序后同一对人不会因为出现顺序不同被记成两条边。
    """
    counter: collections.Counter = collections.Counter()
    # 长名字优先匹配，避免「韩立」抢在「韩立仙师」前面造成误判
    ordered = sorted(characters, key=len, reverse=True)

    for chunk in chunks:
        present = [name for name in ordered if name in chunk.text]
        if len(present) < 2:
            continue
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = sorted((present[i], present[j]))
                counter[(chunk.novel, a, b, relation)] += 1

    return [(novel, a, b, rel, n) for (novel, a, b, rel), n in counter.items()]


def format_graph_hint(subject: str, relation: str, neighbors: list[tuple[str, int]]) -> str:
    """把图查询结果拼成一段给模型看的提示。

    **刻意写明"根据共现统计推断"**：这是启发式结果，可能有假边。让模型知道
    这是线索而不是定论，它才会去核对一起给过去的原文片段，而不是直接照抄。
    """
    if not neighbors:
        return ""
    items = "、".join(f"{name}（共现 {n} 次）" for name, n in neighbors)
    return (
        f"[人物关系线索] 根据全书共现统计推断，与「{subject}」可能存在"
        f"「{relation}」关系的有：{items}。\n"
        f"这是统计推断而非确定事实，请结合下面的原文片段判断。"
    )
