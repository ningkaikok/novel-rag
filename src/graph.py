"""
GraphRAG：把人物关系抽成一张图，回答“全书范围的关系聚合”问题。

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

M4 质量闭环：区分「明确陈述」和「同段共现」
------------------------------------------
实测结论：共现推断跑通了原理，但没达到可用质量（必须如实记录）。查"韩立有
哪些伴侣"时混进了师父、男性和魔头三个假边——共现只知道「两个名字出现在一段
提到'道侣'的文字里」，分不清"他们是道侣"还是"他们在讨论别人的道侣"。

所以 M4 给每条边补上了质量字段，把"怎么来的"如实记下来：

- ``evidence_type``：explicit（LLM 判断是明确的关系陈述）/ co_occurrence（仅同段出现）
- ``confidence``：0~1。共现边固定给低分（CO_OCCURRENCE_CONFIDENCE），LLM 边由模型自评
- ``direction``：方向（如 师父→徒弟），只有 explicit 边可能有
- ``source_chunk_ids``：来源片段定位，审核界面据此展示原文摘录
- ``review_status``：pending/approved/rejected，人工审核结果

在线查询默认只放行 explicit 且置信度达标的边（见 config.GRAPH_REQUIRE_EXPLICIT）；
共现边留在库里供审核。没有生成后端时整条链路自动降级为纯共现，行为与 M4 之前
一致（GRAPH_ENABLED=0 时则完全不建图）。

局限（必须说清楚）
------------------
即使有 LLM 把关，「明确陈述」的判断本身也是模型的意见，仍可能出错——这正是
保留人工审核环节的原因：机器判断 → 门槛过滤 → 人工复核，三层递进而不是
互相替代。图结果始终**和原文片段一起**给模型，让模型自己核对。
"""

import collections
import json
import re

# 共现推断边的固定置信度。共现只能说明「两个名字出现在同一段含关系词的文字里」，
# 分不清"他们是"还是"他们在谈论别人"。给一个保守低分，让它在质量门槛前
# 如实暴露自己的不可靠，同时保留在库里供人工审核。
CO_OCCURRENCE_CONFIDENCE = 0.3

# evidence_type 的两个取值。用常量而不是散落的字符串字面量，
# 避免"explicit"/"co_occurrence"拼错一个字母就静默失配。
EVIDENCE_EXPLICIT = "explicit"
EVIDENCE_CO_OCCURRENCE = "co_occurrence"

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
    # M4：评测语料（原创短篇）里的核心关系是"同一支队伍的队友"，网文式的
    # 四类覆盖不了；补一个「同伴」类。「队伍」是团队叙事里比"队友"更高频的
    # 指称，实测原创语料里靠它才采到点名队友的段落。
    "同伴": ("队友", "同伴", "战友", "队伍"),
}

# 问题里出现这些词，就认为在问对应的关系（用于决定要不要走图检索）
_QUESTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "伴侣": ("伴侣", "道侣", "妻子", "老婆", "夫人", "情侣", "感情"),
    "师徒": ("师父", "师尊", "徒弟", "师承", "拜师"),
    "亲属": ("家人", "亲人", "父母", "兄弟姐妹", "家里"),
    "敌对": ("仇人", "敌人", "对手"),
    "同伴": ("队友", "同伴", "战友"),
}

_EXTRACT_PROMPT = """下面是小说里的几个片段。列出其中出现的**人物姓名**。
只要真实姓名（例如「韩立」「南宫婉」这样的具体名字），
不要「师兄」「老者」「白衣修士」这类泛称，也不要「白光」「金光」这类非人名。
只输出人名，用顿号分隔，不要任何解释。

{batch}"""

# 关系抽取 prompt：让人名抽取再往前走一步，直接判断「有没有关系、什么关系」。
# 要求输出 JSON 数组而不是自由文本，是为了解析可靠——自由文本的格式漂移
# 在 Contextual Retrieval 那边已经踩过坑。每个字段都给了判定标准，
# 尤其 explicit/co_occurrence 的分界：有明确陈述句才算 explicit。
_RELATION_PROMPT = """下面是小说里的几个片段，每段开头标了编号（如 [片段2]）。
候选人物名单：{names}

请判断候选人物之间是否存在「{relation}」关系。判断标准：
- kind="explicit"：片段里有明确的关系陈述（例如"A是B的师父""两人结为道侣"）。
  介绍身份的句子也算：如"队长A""队伍里有司机B和实习生C"对「同伴」关系就是明确陈述
- kind="co_occurrence"：两个人只是出现在同一段文字里，没有明确说明他们的关系

对每一对有关系的人物输出一个 JSON 对象：
{{"a": "人名A", "b": "人名B", "direction": "a→b 或 b→a 或 none",
"kind": "explicit 或 co_occurrence", "confidence": 0到1的小数, "chunks": [片段编号]}}

direction 表示关系的主动方指向另一方（如 师父→徒弟、兄→弟）；对称或无法判断写 none。
confidence 是你对这条关系的把握：有明确陈述才给 0.7 以上，仅凭同段出现不超过 0.5。
chunks 填支持这个判断的片段编号。不要复述、翻译或总结片段内容。
只输出一个 JSON 数组（没有就输出 []），不要任何解释文字。

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


def build_edge_records(chunks: list, characters: set[str], relation: str) -> list[dict]:
    """共现统计的完整版（M4）：除计数外还记录来源片段，产出可直接入库的边记录。

    返回 dict 列表，字段与 character_relations 表的 v2 schema 一一对应：
    共现边的 evidence_type 固定为 co_occurrence、置信度固定为
    CO_OCCURRENCE_CONFIDENCE（低分如实反映"这只是统计推断"）、方向未知。

    兼容性说明：片段对象只要有 .text / .novel 就能参与统计；.chunk_id 通过
    getattr 取——测试用的假片段没有这个属性时来源列表为空，不影响计数。
    """
    # stats 的键是 (novel, a, b)，值是 [共现次数, 来源 chunk_id 列表]
    stats: dict[tuple[str, str, str], list] = {}
    # 长名字优先匹配，避免「韩立」抢在「韩立仙师」前面造成误判
    ordered = sorted(characters, key=len, reverse=True)

    for chunk in chunks:
        present = [name for name in ordered if name in chunk.text]
        if len(present) < 2:
            continue
        chunk_id = getattr(chunk, "chunk_id", None)
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = sorted((present[i], present[j]))
                entry = stats.setdefault((chunk.novel, a, b), [0, []])
                entry[0] += 1
                # 同一片段里两个人名出现多次也只记一次来源，去重保持定位干净
                if chunk_id is not None and chunk_id not in entry[1]:
                    entry[1].append(chunk_id)

    return [
        {
            "novel": novel,
            "person_a": a,
            "person_b": b,
            "relation": relation,
            "weight": weight,
            # 共现无法判断关系方向，留空交给审核界面显示"未知"
            "direction": None,
            "confidence": CO_OCCURRENCE_CONFIDENCE,
            "evidence_type": EVIDENCE_CO_OCCURRENCE,
            "source_chunk_ids": source_ids,
        }
        for (novel, a, b), (weight, source_ids) in stats.items()
    ]


def build_edges(chunks: list, characters: set[str], relation: str) -> list[tuple]:
    """在给定片段里做人物共现统计，产出旧形状的关系边 (书, A, B, 关系, 次数)。

    M4 起内部实现委托给 build_edge_records（它额外带质量字段和来源定位）；
    这里保留旧的 5 元组返回值，既有调用方和测试不受影响。A/B 按字典序排好，
    同一对人不会因为出现顺序不同被记成两条边。
    """
    return [
        (
            record["novel"],
            record["person_a"],
            record["person_b"],
            record["relation"],
            record["weight"],
        )
        for record in build_edge_records(chunks, characters, relation)
    ]


def _parse_relation_payload(
    raw: str, valid_names: set[str], batch_chunk_ids: list[int | None]
) -> list[dict] | None:
    """解析 LLM 输出的 JSON 数组为边记录；解析不出返回 None（调用方降级为共现）。

    解析规则刻意宽容（每一条都对应实测踩过的输出形态）：
    - **容忍模型复读输入**：小模型有时会把「[片段1]」这类输入标记原样抄进
      输出，直接拿第一个 '[' 当数组起点必然解析失败——先把片段标记从文本里
      删掉再解析；
    - 从后往前尝试每个 '[' 作为候选起点，用 raw_decode 取第一个合法值：
      这样对象内部的 "chunks": [1,2] 数字数组会被"元素必须都是对象"的校验
      拒掉，继续向前找真正的外层数组；
    - confidence 截断到 [0,1]，缺省 0.5（中性值，不冒充高把握）；
    - kind 不是 explicit 的一律按 co_occurrence 处理（宁低勿高）；
    - chunks 编号越界/非法时回退为"整批都是来源"，宁可多记不可错记。
    人名不在候选名单里、或 a==b 的条目直接丢弃——那是模型幻觉，不能入库。
    """
    # 去掉可能被复读的片段标记和 markdown 围栏，只留真正的判断内容
    cleaned = re.sub(r"\[片段\s*\d+\]", "", raw.replace("```", ""))
    candidates = [i for i, ch in enumerate(cleaned) if ch == "["]
    decoder = json.JSONDecoder()

    items: list | None = None
    for start in reversed(candidates):
        try:
            value, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, list):
            continue
        # 输出被截断时（max_tokens 用尽）最后一个对象往往不完整，
        # raw_decode 会失败并继续往前试——宁可少收几条也不能整体报废
        if value and all(isinstance(entry, dict) for entry in value):
            items = value
            break

    if items is None:
        return None

    records: dict[tuple[str, str], dict] = {}
    for item in items:
        a, b = item.get("a"), item.get("b")
        # 人名必须来自候选名单：模型偶尔会输出名单外的名字（泛称/幻觉），
        # 这些没有经过 GRAPH_MIN_NAME_HITS 降噪，直接放行会把噪音放进图里
        if not (isinstance(a, str) and isinstance(b, str)):
            continue
        if a not in valid_names or b not in valid_names or a == b:
            continue

        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        explicit = item.get("kind") == EVIDENCE_EXPLICIT

        # 方向归一化成「名字→名字」的字符串；b→a 时交换主动方，
        # 存储侧 person_a/person_b 保持字典序不变，方向只写在 direction 里
        direction = item.get("direction")
        if direction == f"{a}→{b}":
            arrow = f"{a}→{b}"
        elif direction == f"{b}→{a}":
            arrow = f"{b}→{a}"
        else:
            arrow = None

        raw_chunks = item.get("chunks")
        source_ids: list[int | None] = []
        if isinstance(raw_chunks, list):
            source_ids = [
                batch_chunk_ids[k - 1]
                for k in raw_chunks
                if isinstance(k, int) and 1 <= k <= len(batch_chunk_ids)
            ]
        if not source_ids:
            # 没给出有效编号：整批片段都算来源。宁可冗余也不能让边失去追溯
            source_ids = list(batch_chunk_ids)

        key = (min(a, b), max(a, b))
        prev = records.get(key)
        # 同一对人在同一批里被抽到多次，保留置信度更高的一条
        if prev is None or confidence > prev["confidence"]:
            records[key] = {
                "person_a": key[0],
                "person_b": key[1],
                "direction": arrow,
                "confidence": confidence,
                "evidence_type": EVIDENCE_EXPLICIT if explicit else EVIDENCE_CO_OCCURRENCE,
                "source_chunk_ids": [c for c in source_ids if c is not None],
            }
    return list(records.values())


def extract_relations_llm(
    chunks: list,
    characters: set[str],
    relation: str,
    generate_fn,
    batch_size: int = 4,
    errors: list | None = None,
) -> list[dict] | None:
    """用 LLM 从片段中抽取带质量标注的关系边；失败返回 None（调用方降级为共现）。

    与 extract_characters_from_chunks 相同的容错哲学：单批失败不阻断整体，
    但原因必须收集进 errors——静默降级是最难查的一种坏。所有批次全部失败
    （一次成功批都没有）时才返回 None，让上层完整退回纯共现结果。

    批量大小默认 4：每条关系的输出比人名列表长得多（JSON 对象），批太大
    容易顶爆输出长度导致 JSON 被截断、解析失败率上升。
    """
    ordered_names = sorted(characters)
    merged: list[dict] = []
    succeeded_batches = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        # 片段编号从 1 开始、只在本批内有意义；解析时按同样的偏移映射回
        # 真实 chunk_id，这样 LLM 引用的"[片段2]"能落到正确的原文片段上
        batch_text = "\n".join(
            f"[片段{k}] {chunk.text[:400]}" for k, chunk in enumerate(batch, start=1)
        )
        prompt = _RELATION_PROMPT.format(
            names="、".join(ordered_names),
            relation=relation,
            batch=batch_text[:4500],
        )
        try:
            out = "".join(generate_fn(prompt))
        except Exception as exc:
            if errors is not None:
                errors.append(f"{type(exc).__name__}: {exc}")
            continue
        parsed = _parse_relation_payload(
            out, characters, [getattr(c, "chunk_id", None) for c in batch]
        )
        if parsed is None:
            if errors is not None:
                errors.append(f"关系抽取输出解析失败：{out[:120]}")
            continue
        succeeded_batches += 1
        merged.extend(
            {**record, "novel": batch[0].novel, "relation": relation} for record in parsed
        )

    return _merge_relation_records(merged) if succeeded_batches else None


def _merge_relation_records(records: list[dict]) -> list[dict]:
    """按 (novel, 人物对, 关系) 合并重复边，并给每条边算出入库用的 weight。

    同一对人可能出现在多个批次里被抽到多次——数据库主键是
    (novel, person_a, person_b, relation)，不去重会直接撞主键。
    合并规则：保留置信度更高的一条判断；来源片段取并集（追溯宁多勿缺）；
    weight = 去重后的来源片段数，与共现边的"次数"保持同一种量纲，
    让查询端的 ORDER BY weight DESC 对两类边都公平。
    """
    best: dict[tuple, dict] = {}
    for record in records:
        key = (
            record["novel"],
            record["person_a"],
            record["person_b"],
            record["relation"],
        )
        prev = best.get(key)
        if prev is None:
            best[key] = {**record}
            continue
        # 来源片段先做并集——无论最后留下哪条判断，证据都不能丢
        seen = set(prev["source_chunk_ids"])
        for cid in record["source_chunk_ids"]:
            if cid is not None and cid not in seen:
                seen.add(cid)
                prev["source_chunk_ids"].append(cid)
        if record["confidence"] > prev["confidence"]:
            winner = {**record}
            # 换成置信度更高的判断，但保留刚算好的来源并集
            winner["source_chunk_ids"] = list(prev["source_chunk_ids"])
            best[key] = winner

    merged = []
    for record in best.values():
        record["source_chunk_ids"] = [
            cid for cid in dict.fromkeys(record["source_chunk_ids"]) if cid is not None
        ]
        record["weight"] = max(1, len(record["source_chunk_ids"]))
        merged.append(record)
    return merged


def format_graph_hint(subject: str, relation: str, neighbors: list[tuple[str, int]]) -> str:
    """把图查询结果拼成一段给模型看的提示。

    **刻意写明"推断"**：无论边来自 LLM 抽取还是共现统计，都是机器判断、
    可能有假边。让模型知道这是线索而不是定论，它才会去核对一起给过去的
    原文片段，而不是直接照抄。数字统一表述为"依据"——共现边的权重是
    共现次数，LLM 边是来源片段数，对模型来说语义等价（支持这条关系的证据量）。
    """
    if not neighbors:
        return ""
    items = "、".join(f"{name}（{n} 条依据）" for name, n in neighbors)
    return (
        f"[人物关系线索] 根据全书关系抽取推断，与「{subject}」可能存在"
        f"「{relation}」关系的有：{items}。\n"
        f"这是自动抽取的线索而非确定事实，请结合下面的原文片段核对后再回答。"
    )
